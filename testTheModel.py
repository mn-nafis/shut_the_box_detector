#Download the best.pt model put it in the same folder as your python code.
#In your favourite python environment write the following code
from ultralytics import YOLO
import cv2
"""
This script is to test our model

"""
video_source = 2
#Load a pretrained yolo model
model = YOLO('model/best.pt')
#Run inference on the source
results = model(source=video_source, show=True, conf=0.4, save=True)
