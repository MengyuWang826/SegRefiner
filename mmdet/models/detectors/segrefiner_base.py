from collections import OrderedDict
import torch
import torch.distributed as dist
import numpy as np
from mmcv.runner import BaseModule
from ..builder import DETECTORS, build_head, build_loss


def uniform_sampler(num_steps, batch_size, device):
    all_indices = np.arange(num_steps)
    indices_np = np.random.choice(all_indices, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices  

@DETECTORS.register_module()
class SegRefiner(BaseModule):
    """Base class for detectors."""
    def __init__(self,
                 task,
                 step,
                 denoise_model,
                 diffusion_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_texture=dict(type='TextureL1Loss', loss_weight=5.0),
                 init_cfg=None):
        super(SegRefiner, self).__init__(init_cfg)
        self.task = task
        self.denoise_model = build_head(denoise_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(diffusion_cfg)
        self.loss_mask = build_loss(loss_mask)
        self.loss_texture = build_loss(loss_texture)
        self.num_classes = 80
        self.step = step
    
    def _diffusion_init(self, diffusion_cfg):
        self.diff_iter = diffusion_cfg['diff_iter']
        betas = diffusion_cfg['betas']
        self.eps = 1.e-6
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'], 
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]
    
    def forward(self, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img_metas[0])

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            if self.task == 'instance':
                return self.simple_test_instance(img_metas, **kwargs)
            elif self.task == 'semantic':
                return self.simple_test_semantic(img_metas, **kwargs)
            else:
                raise ValueError(f'unsupported task type: {self.task}')
    
    def forward_train(self, **kwargs):
        target, x_last, img, current_device = self.get_train_input(**kwargs)
        t = uniform_sampler(self.num_timesteps, img.shape[0], current_device)
        x_t = self.q_sample(target, x_last, t, current_device)
        z_t = torch.cat((img, x_t), dim=1)
        pred_logits = self.denoise_model(z_t, t) 
        iou_pred = self.cal_iou(target, pred_logits)
        losses = dict()
        losses['loss_mask'] = self.loss_mask(pred_logits, target)
        losses['loss_texture'] = self.loss_texture(pred_logits, target)
        losses['iou'] = iou_pred.mean()
        return losses
    
    def get_train_input(self, object_img, object_gt_masks, object_coarse_masks,
                        patch_img=None, patch_gt_masks=None, patch_coarse_masks=None):
        current_device = object_img.device
        img = object_img
        target = self._bitmapmasks_to_tensor(object_gt_masks, current_device)
        x_last = self._bitmapmasks_to_tensor(object_coarse_masks, current_device)
        if patch_img is not None:
            img = torch.cat((img, patch_img), dim=0)
            target = torch.cat((target, self._bitmapmasks_to_tensor(patch_gt_masks, current_device)), dim=0)
            x_last = torch.cat((x_last, self._bitmapmasks_to_tensor(patch_coarse_masks, current_device)), dim=0)
        return target, x_last, img, current_device
    
    @torch.no_grad()
    def cal_iou(self, target, mask, eps=1e-3):
        target = target.clone().detach() >= 0.5
        mask = mask.clone().detach() >= 0
        si = (target & mask).sum(-1).sum(-1)
        su = (target | mask).sum(-1).sum(-1)
        return (si / (su + eps))
    
    def _bitmapmasks_to_tensor(self, bitmapmasks, current_device):
        tensor_masks = []
        for bitmapmask in bitmapmasks:
            tensor_masks.append(bitmapmask.masks)
        tensor_masks = np.stack(tensor_masks)
        tensor_masks = torch.tensor(tensor_masks, device=current_device, dtype=torch.float32)
        return tensor_masks
    
    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        return sample
    
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    def train_step(self, data, *args):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def p_sample_loop(self, xs, indices, current_device, use_last_step=True):
        res, fine_probs = [], []
        for data in xs:
            x_last, img, cur_fine_probs = data
            if cur_fine_probs is None:
                cur_fine_probs = torch.zeros_like(x_last)
            x = x_last
            for i in indices:
                t = torch.tensor([i] * x.shape[0], device=current_device)
                last_step_flag = (use_last_step and i==indices[-1])
                model_input = torch.cat((img, x), dim=1)
                x, cur_fine_probs = self.p_sample(model_input, cur_fine_probs, t)

                if last_step_flag:
                    x =  x.sigmoid()
                else:
                    sample_noise = torch.rand(size=x.shape, device=x.device)
                    fine_map = (sample_noise < cur_fine_probs).float()
                    pred_x_start = (x >= 0).float()
                    x = pred_x_start * fine_map + x_last * (1 - fine_map)
            res.append(x)
            fine_probs.append(cur_fine_probs)
        res = torch.cat(res, dim=0)
        fine_probs = torch.cat(fine_probs, dim=0)
        return res, fine_probs

    def p_sample(self, model_input, cur_fine_probs, t):
        pred_logits = self.denoise_model(model_input, t)
        t = t[0].item()
        x_start_fine_probs = 2 * torch.abs(pred_logits.sigmoid() - 0.5)
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f
        return pred_logits, cur_fine_probs
    
    def simple_test_instance(img_metas, **kwargs):
        raise NotImplementedError
    
    def simple_test_semantic(img_metas, **kwargs):
        raise NotImplementedError
    

    