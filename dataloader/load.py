import openslide
import numpy as np
import cv2
import tifffile as tff

def load_slide(slide_path, level=1):
    with openslide.open_slide(slide_path) as slide:
        width, height = (slide.level_dimensions[level])
        wsi = np.array(slide.read_region((0, 0), level, (width, height)))

    wsi = cv2.cvtColor(wsi, cv2.COLOR_RGBA2RGB)
    
    return wsi