import os
from ultralytics import YOLO

if __name__ == '__main__':
    path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteTipModel\training\data.yaml"
    model = YOLO('yolov8s.yaml')
    model.train(data=path, epochs=100, imgsz=640)
