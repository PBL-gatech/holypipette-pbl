import os
from ultralytics import YOLO

if __name__ == '__main__':
    path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteTipModel\training\focusdata.yaml"
    runspath = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteTipModel\results"
    model = YOLO('yolo11n.pt')
    model.train(data=path, epochs=100, imgsz=640, patience = 20, project=runspath,seed=0)
