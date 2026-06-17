import numpy as np
import cv2
from scipy import ndimage

def resize_img(img, portion=1):
    width = int(img.shape[1]*portion)
    height = int(img.shape[0]*portion)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # need modification
    
    return resized
    

def remove_small_objects(binary_image, min_size):
    
    # Label connected components in the binary image
    labeled_image, num_labels = ndimage.label(binary_image)

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())

    # Determine the indices of small components to be removed
    small_component_indices = np.where(component_sizes < min_size)[0]

    # Create a mask to remove the small components
    mask = np.isin(labeled_image, small_component_indices)

    # Set the pixels corresponding to the small components to 0 (background)
    binary_image[mask] = 0

    return binary_image


def binarize(wsi):
    tissue_dict = {}
    label_size = 256
    
    # grayscale & normalization histogram
    grayscale = cv2.cvtColor(wsi,cv2.COLOR_BGR2GRAY)
#     nhistogram = calc_nhist(grayscale)
    
    # otsu thresholding
    _, binary = cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    binary = cv2.dilate(binary,kernel,iterations=2)
    binary = cv2.erode(binary,kernel,iterations=2)
    binary = cv2.erode(binary,kernel,iterations=2)
    binary = cv2.dilate(binary,kernel,iterations=2)
    
    # contour
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    external_contours = np.zeros(binary.shape)
    internal_contours = np.zeros(binary.shape)
    
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(external_contours,contours,i,255,-1) # -1 --> fill
        else:
            cv2.drawContours(internal_contours,contours,i,255,-1)
    
    internal_contours = internal_contours.astype('uint8')
    external_contours = external_contours.astype('uint8')
    
    count,labels,stats,_ = cv2.connectedComponentsWithStats(internal_contours)
    internal = np.zeros(internal_contours.shape)
    for i in range(1,count):
        _,_,_,_,a = stats[i]
        if a>=label_size//4:
            internal[labels==i]=255
            
    count,labels,stats,_ = cv2.connectedComponentsWithStats(external_contours)
    external = np.zeros(external_contours.shape)
    for i in range(1,count):
        _,_,_,_,a = stats[i]
        if a>=label_size:
            external[labels==i]=255
    
    binary = (external - internal).astype('uint8')
    
    fill = ndimage.binary_fill_holes(binary).astype('uint8')
    remove_noise = remove_small_objects(fill, 3000)
    
    return remove_noise