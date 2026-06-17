import os
from tensorflow.keras import models

MODEL_PATH = './model/weights'

def load_model(model_name:str):
    model_path = os.path.join(MODEL_PATH, model_name)
    model = models.load_model(model_path)
    
    return model

if __name__ == "__main__":
    print('TensorFlow Version ' + tensorflow.__version__)