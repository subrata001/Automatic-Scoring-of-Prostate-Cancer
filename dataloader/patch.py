import numpy as np
import cv2
import math
import warning

THRESHOLD = 210
BLACK_THRESHOLD = 5
PATCH_SIZE = 256

def pad_image(img, len:int, val:str):
    assert img is not None and img.ndim > 1

    padding = [(len, len), (len, len)]
    if img.ndim == 3: 
        padding.append((0,0))

    return np.pad(img, tuple(padding), val)

def transpose_image(img):
    assert img is not None and img.ndim > 1

    transposed_dim = [1,0]
    if img.ndim == 3:
        transposed_dim.append(2) # 1,0,2

    return np.transpose(img, tuple(transposed_dim)).copy()

def get_tissue_parts_indices(tissue, min_consec_info):
    split_points = np.where(np.diff(tissue) != 1)[0]+1
    tissue_parts = np.split(tissue, split_points)

    return [tp for tp in tissue_parts if len(tp) >= min_consec_info]

def get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end-start)/patch_size

    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts*patch_size) - (end-start)
    shift = excess // 2

    return [i * patch_size + start - shift for i in range(num_subparts)]

def eval_and_append_xy_coords(coords,
                               image, 
                               mask, 
                               patch_size, 
                               x, y,
                               min_patch_info,
                               transposed,
                               precompute):

    patch_1d = (image[y:y+patch_size, x:x+patch_size, :].mean(axis=2).reshape(-1))
    idx_tissue = np.where(patch_1d <= THRESHOLD)[0]
    idx_black = np.where(patch_1d < BLACK_THRESHOLD)[0]
    idx_background = np.where(patch_1d > THRESHOLD)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = THRESHOLD
        patch_1d[idx_background] = THRESHOLD
        val1 = int(patch_1d.mean())
        val2 = mask[y:y+patch_size, x:x+patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([coords, [[val1, x-patch_size, y-patch_size]]])
                else:
                    coords = np.concatenate([coords, [[val1, y-patch_size, x-patch_size]]])
            else:
                coords = np.concatenate([coords, [[val1, y, x]]])
               
    return coords

def compute_coords(image,
                   patch_size=PATCH_SIZE,
                   precompute=False,
                   min_patch_info=0.35,
                   min_axis_info=0.35,
                   min_consec_axis_info=0.35,
                   min_decimal_keep=0.7):

    if type(image) != np.ndarray:
        # if image is a Tensor
        image = image.numpy()
    
    # masked tissue will be used to compute the coordinates
    _, mask = cv2.threshold(np.array(image)[:,:,1], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # initialize coordinate accumulator
    coords = np.zeros([0, 3], dtype=int)

    # pad image and mask to make sure no tissue is potentially missed out
    image = pad_image(image, patch_size, 'maximum')
    mask = pad_image(mask, patch_size, 'minimum')
    
    y_sum = mask.sum(axis=1)
    x_sum = mask.sum(axis=0)

    # if on bits in x_sum is greater than in y_sum, the tissue is
    # likely aligned horizontally. The algorithm works better if
    # the image is aligned vertically, thus the image will be transposed
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        image = transpose_image(image)
        mask = transpose_image(mask)
        y_sum, _ = x_sum, y_sum
        transposed = True
    else:
        transposed = False
    
    # where y_sum is more than the minimum number of on-bits
    y_tissue = np.where(y_sum >= (patch_size*min_axis_info))[0]
    
    if len(y_tissue) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    y_tissue_parts_indices = get_tissue_parts_indices(
        y_tissue, patch_size*min_consec_axis_info)
    
    if len(y_tissue_parts_indices) < 1: 
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    # loop over the tissues in y-dimension
    for yidx in y_tissue_parts_indices:
        y_tissue_subparts_coords = get_tissue_subparts_coords(
            yidx, patch_size, min_decimal_keep)
        
        for y in y_tissue_subparts_coords:
            # in y_slice, where x_slice_sum is more than the minimum number of on-bits
            x_slice_sum = mask[y:y+patch_size, :].sum(axis=0)
            x_tissue = np.where(x_slice_sum >= (patch_size*min_axis_info))[0]
            
            x_tissue_parts_indices = get_tissue_parts_indices(x_tissue, patch_size*min_consec_axis_info)
            
            # loop over tissues in x-dimension (inside y_slice 'y:y+patch_size')
            for xidx in x_tissue_parts_indices:
                x_tissue_subparts_coords = get_tissue_subparts_coords(xidx, patch_size, min_decimal_keep)
                
                for x in x_tissue_subparts_coords:
                    coords = eval_and_append_xy_coords(coords, 
                                                       image, 
                                                       mask, 
                                                       patch_size, 
                                                       x, y, 
                                                       min_patch_info, 
                                                       transposed, 
                                                       precompute)     
    
    if len(coords) < 1:
        warnings.warn("Not enough tissue in image (x-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    if precompute: return coords
    else: return image, coords