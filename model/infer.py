import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image

def model_predict(patch, preprocessing_model, model):
    input_ = image.img_to_array(patch)
    input_ = np.expand_dims(input_, axis=0)
    input_ = preprocessing_model.preprocess_input(input_)
    
    prediction = model.predict(input_, verbose=0)
    prediction_arg = np.argmax(prediction)
    
    return prediction_arg

def get_preprocessing_model(model_name='efficientnet'):
    if model_name == 'efficientnet':
        model = tensorflow.keras.applications.efficientnet
    

    return model