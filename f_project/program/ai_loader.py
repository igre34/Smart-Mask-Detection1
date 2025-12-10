import cv2
import os
import sys 
from tensorflow.keras.models import load_model
import config


#function to load the ai models
def load_ai_mod():
    print("loding face detector...")
    face = cv2.dnn.readNet(config.PROTOTXT_PATH, config.WEIGHTS_PATH)

    print("loading mask detector...")
#if statement to check if the model path exists to avoid errors
    if os.path.exists(config.MODEL_PATH):
        
        mask = load_model(config.MODEL_PATH)

    else: 
        print(f"error: I cannot find model to detect masks at {config.MODEL_PATH}")
        sys.exit()
    return face, mask
