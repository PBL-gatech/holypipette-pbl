import os
from ultralytics import YOLO


model = YOLO('yolov8s.yaml')
model.train(data='/holypipette-pbl/holypipette/deepLearning/pipetteTipModel/training/data.yaml', epochs=50, imgsz=640)