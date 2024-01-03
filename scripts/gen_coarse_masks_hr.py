# -----------------------------------------------------------------------------------
# References:
# cascadepsp: https://github.com/hkchengrex/CascadePSP/blob/83cc3b8783b595b2e47c75016f93654eaddb7412/util/boundary_modification.py
# -----------------------------------------------------------------------------------
import os
import cv2
import numpy as np
import random
import math
import json
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image


def get_random_structure(size):
    # The provided model is trained with 
    #   choice = np.random.randint(4)
    # instead, which is a bug that we fixed here
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target = 0.8):
    # modifies boundary of the given mask.
    # remove consecutive vertice of the boundary by regional sample rate
    # ->
    # remove any vertice by sample rate
    # ->
    # move vertice by distance between vertice and center of the mask by move rate. 
    # input: np array of size [H,W] image
    # output: same shape as input
    
    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #only modified contours is needed actually. 
    sampled_contours = []   
    modified_contours = [] 

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)

        #remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)
        
        idx_dist = []
        for i in range(number_of_vertices-number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])
            
        idx_dist = sorted(idx_dist, key=lambda x:x[1])
        
        remove_start = random.choice(idx_dist[:math.ceil(0.1*len(idx_dist))])[0]
        
       #remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
        new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
        contour = new_contour
        

        #sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            #modify contours
            for idx, coor in enumerate(modified_contour):

                change = np.random.normal(0,move_rate) # 0.1 means change position of vertex to 10 percent farther from center
                x,y = coor[0]
                new_x = x + (x-center[0]) * change
                new_y = y + (y-center[1]) * change

                modified_contour[idx] = [new_x,new_y]
        modified_contours.append(modified_contour)
        
    #draw boundary
    gt = np.copy(image)
    image = np.zeros((image.shape[0], image.shape[1], 3))

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    if len(image.shape) == 3:
        image = image[:, :, 0]
    image = perturb_seg(image, iou_target)

    
    image = (image >= 128).astype(np.uint8) * 255
    
    return image

def run_inst(img_info):
    gt_mask = cv2.imread(data_root+img_info['maskname'], cv2.IMREAD_GRAYSCALE)
    assert gt_mask is not None
    coarse_mask = modify_boundary(gt_mask*255)
    coarse_filename = data_root + img_info['coarsename']
    Image.fromarray(coarse_mask).save(coarse_filename)
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expand_coarse_mask = np.zeros_like(gt_mask)
    cv2.drawContours(expand_coarse_mask, contours, contourIdx=-1, color=1, thickness=-1)
    expand_coarse_mask = modify_boundary(expand_coarse_mask*255)
    expand_coarse_filename = data_root + img_info['expandname']
    Image.fromarray(expand_coarse_mask).save(expand_coarse_filename)


if __name__ == '__main__':
    data_root = 'data/'
    thin_object_txt = 'data/thin_object/list/train.txt'
    thin_object_root = 'data/thin_object/images'
    dis_root = 'data/dis/DIS-TR/im'
    collection_json_file = 'data/collection_hr.json'
    collection = dict(dis=[], thin=[])

    print('----------start collecting dis---------------')
    dis_all_files = os.listdir(dis_root)
    with tqdm(total=len(dis_all_files)) as p:
        for filename in dis_all_files:
            mask_file = filename.replace('jpg', 'png')
            mask_name = 'dis/DIS-TR/gt/' + mask_file
            img_name = 'dis/DIS-TR/im/' + filename
            coarsename = 'dis/DIS-TR/coarse/' + mask_file
            expandname = 'dis/DIS-TR/coarse_expand/' + mask_file
            img = cv2.imread(os.path.join(dis_root, filename))
            assert img is not None
            h, w = img.shape[:2]
            img_info = {'filename': img_name,
                        'maskname': mask_name,
                        'coarsename': coarsename,
                        'expandname': expandname,
                        'height': h, 'width': w}
            collection['dis'].append(img_info)
            p.update()

    print('----------start tansforming dis---------------')
    os.makedirs(data_root + 'dis/DIS-TR/coarse', exist_ok=True)
    os.makedirs(data_root + 'dis/DIS-TR/coarse_expand', exist_ok=True)
    with mp.Pool(processes=20) as p:
        with tqdm(total=len(collection['dis'])) as pbar:
            for _ in p.imap_unordered(run_inst, collection['dis']):
                pbar.update()
    
    print('----------start collecting thin---------------')
    f=open(thin_object_txt)
    thin_all_files=[]
    for line in f:
        thin_all_files.append(line.strip())
    with tqdm(total=len(dis_all_files)) as p:
        for filename in thin_all_files:
            mask_file = filename
            mask_name = 'thin_object/masks/' + mask_file
            filename = filename.replace('png', 'jpg')
            img_name = 'thin_object/images/' + filename
            coarsename = 'thin_object/coarse/' + mask_file
            expandname = 'thin_object/coarse_expand/' + mask_file
            img = cv2.imread(os.path.join(thin_object_root, filename))
            assert img is not None
            h, w = img.shape[:2]
            img_info = {'filename': img_name,
                        'maskname': mask_name,
                        'coarsename': coarsename,
                        'expandname': expandname,
                        'height': h, 'width': w}
            collection['thin'].append(img_info)
            p.update()

    print('----------start tansforming thin---------------')
    os.makedirs(data_root + 'thin_object/coarse', exist_ok=True)
    os.makedirs(data_root + 'thin_object/coarse_expand', exist_ok=True)
    with mp.Pool(processes=20) as p:
        with tqdm(total=len(collection['thin'])) as pbar:
            for _ in p.imap_unordered(run_inst, collection['thin']):
                pbar.update()
    
    print('----------writing json file---------------')
    with open(collection_json_file, 'w') as f:
        json.dump(collection, f)