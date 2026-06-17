import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from dataloader.load import load_slide
from dataloader.patch import PATCH_SIZE, compute_coords
from model.infer import model_predict, get_preprocessing_model
from model.load import load_model
from utils.preprocess import resize_img
from utils.save import save, SAVE_PATH
from utils.mask import Mask
import tifffile as tff
from utils.contour import Contour
import cv2

wsi_path = '/home/subrata/members/Data/Prostate/Test_Slides/8d023fab6d98cb21342681e0dda180d0.tiff'



def predict_and_json(model_file_names, wsi, coordinates, pattern):
    model_normal_cancer, model_low_high, model_4_5 = [load_model(file_name) for file_name in model_file_names]
    preprocessing_model = get_preprocessing_model()
    wsi_width = wsi.shape[1]
    contour = Contour(wsi_width)

    for (_, y, x) in coordinates:
        start_point = (x, y)
        patch = wsi[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        if patch.shape == (PATCH_SIZE, PATCH_SIZE, 3):
            normal_or_cancer = model_predict(patch, preprocessing_model, model_normal_cancer)
            if normal_or_cancer:  # normal
                pattern['amount']['normal'] += 1
            else:
                low_or_high = model_predict(patch, preprocessing_model, model_low_high)
                if low_or_high:
                    pattern['amount']['pattern3'] += 1
                    contour.stack(pattern_type='pattern3', img=patch, offset=start_point)
                else:
                    four_or_five = model_predict(patch, preprocessing_model, model_4_5)
                    if four_or_five == 0:
                        pattern['amount']['pattern4'] += 1
                        contour.stack(pattern_type='pattern4', img=patch, offset=start_point)
                    else:
                        pattern['amount']['pattern5'] += 1
                        contour.stack(pattern_type='pattern5', img=patch, offset=start_point)

    output = contour.to_json()
        
    return output

def predict_and_image(model_file_names, wsi, coordinates, pattern):
    
    model_normal_cancer, model_low_high, model_4_5 = [load_model(file_name) for file_name in model_file_names]
    preprocessing_model = get_preprocessing_model()
    
    mask = Mask(wsi.shape)
       
    for (_, y, x) in coordinates: # w? h?
        start_point = (x, y)
        end_point = (x+PATCH_SIZE, y+PATCH_SIZE)

        patch = wsi[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        
        # use recursive function?
        if patch.shape == (PATCH_SIZE, PATCH_SIZE, 3): # image channels
            normal_or_cancer = model_predict(patch, preprocessing_model, model_normal_cancer)
            if normal_or_cancer: # normal
                pattern['amount']['normal']+=1
                mask.draw_rect(start_point, end_point, pattern['color']['normal'])
                
            else: # cancer
                low_or_high = model_predict(patch, preprocessing_model, model_low_high)
                if low_or_high: # low
                    pattern['amount']['pattern3']+=1
                    mask.draw_rect(start_point, end_point, pattern['color']['pattern3'])
                    
                else: # high
                    four_or_five = model_predict(patch, preprocessing_model, model_4_5)
                    if four_or_five == 0: # four
                        pattern['amount']['pattern4']+=1
                        mask.draw_rect(start_point, end_point, pattern['color']['pattern4'])
                        
                    else: # five
                        pattern['amount']['pattern5']+=1
                        mask.draw_rect(start_point, end_point, pattern['color']['pattern5'])

    return mask.overlap(wsi)

#def predict(mode_file_names, wsi, coordinates, pattern, output_mode):
#     preprocessing_model = get_preprocessing_model()
#     if output_mode.lower() == 'json':
#         wsi_width = wsi.shape[1]
#         contour = Contour(wsi_width)
#         func = contour.stack
#     elif output_mode.lower() == 'image':
#         mask = Mask(wsi.shape)
#         func = mask.draw_rect

#     # customize
#     pattern_type = list(pattern['amount'].keys())
#     pattern_type[2], pattern_type[3] = pattern_type[3], pattern_type[2]
    
#     for (_, y, x) in coordinates:
#         start_point = (x,y)
#         end_point = (x+PATCH_SIZE, y+PATCH_SIZE)
#         patch = wsi[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

#         if patch.shape == (PATCH_SIZE, PATCH_SIZE, 3):
#             recursive_func()
        
#     return

# patch, preprocessing, model, pattern, func, kwargs, pattern_type

# def recursive_func(patch, preprocessing_model, pattern, pattern_type, func, repeat=0, *models):
#     model = models[repeat]
#     result = model_predict(patch, preprocessing_model, model)
#     if result:
#         pattern['type'][pattern_type[repeat]]+=1
#         func()
#     else:
#         repeat += 1
#         if repeat == len(models):
#             pattern['type'][pattern_type[repeat]]+=1
#             func()
#             return
#         recursive_func(patch, preprocessing_model, pattern, pattern_type, func, repeat, models)
#     return


def calculate(pattern_amount:dict, display=False):
    total_patches = sum(list(pattern_amount.values()))
    data = {'total':total_patches}
    data.update(pattern_amount)
    
    if display:
        print ('Predicted total patches: ', total_patches)
        for pattern, patches in pattern_amount.items():
            percent = round(((patches/total_patches)*100), 2)
            print(f'Number of {pattern} patches: {patches}/{total_patches} ({percent}%)')

    return data
        
def test_wsi(wsi_path:str, model_file_names:tuple):
    wsi_name = os.path.basename(wsi_path).split('.')[0]
    pattern = {'amount':{'normal':0, 'pattern3':0, 'pattern4':0, 'pattern5':0},
               'color':{'normal':'red', 'pattern3':'green', 'pattern4':'blue', 'pattern5':'yellow'}}
    
    load_wsi = load_slide(wsi_path)
    #wsi_image = cv2.cvtColor(load_wsi, cv2.COLOR_BGR2RGB)
    reshape_wsi = resize_img(load_wsi)
    coordinates = compute_coords(reshape_wsi, precompute=True)
    
    output = predict_and_json(model_file_names, 
                              reshape_wsi, 
                              coordinates, 
                              pattern) # core
                                  

    save(output, wsi_name, pattern['color'], func_mode='json')
    
    patch_count = calculate(pattern['amount'])
    save(patch_count, wsi_name, pattern['color'], func_mode='count')

    

if __name__ == "__main__":
    test_wsi(wsi_path, ('model_nc.tf', 'model_lh.tf', 'model_45.tf'))

