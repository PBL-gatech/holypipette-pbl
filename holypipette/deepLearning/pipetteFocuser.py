#!/usr/bin/env python
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging

class PipetteFocuser:
    def __init__(self):
        # Set the scale factor as an instance attribute
   
        
        # Determine the model path
        cur_dir = Path(__file__).parent.absolute()
        model_path = os.path.join(cur_dir, 'pipetteModel', 'regression_model2.onnx')
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Input image size (as used in training transforms)
        self.imgSize = 224
        
        # Mean and std used in training (from Albumentations normalization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img):
        """
        Preprocess the image:
          - Resize to self.imgSize x self.imgSize.
          - Convert BGR (OpenCV) to RGB.
          - Scale pixel values to [0, 1].
          - Normalize with mean and std.
          - Rearrange dimensions from HWC to CHW and add a batch dimension.
        """
        img_resized = cv2.resize(img, (self.imgSize, self.imgSize))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def get_pipette_focus_value(self, img):
        """
        Runs the ONNX model on a preprocessed image and returns the defocus value in microns.
        """
        input_tensor = self.preprocess(img)
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        # print(f"Inference time: {inference_time:.4f} seconds")
        output_array = outputs[0]
        norm_pred = float(output_array.flatten()[0])
        # print(f"Predicted defocus value: {norm_pred:.2f}")
        # pred_microns = self.denormalize_z(norm_pred)
        pred_microns = norm_pred
        return pred_microns

if __name__ == '__main__':
    focuser = PipetteFocuser()
    
    # Adjust image path as needed
    # cur_dir = Path(__file__).parent.absolute()
    # image_path = os.path.join(cur_dir, "", "neg7_focus.png")
    image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\rig_recorder_data\2025_02_17-18_20\camera_frames\7524_1739834506.312564.webp"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        exit(1)
    
    focus_value = focuser.get_pipette_focus_value(img)
    # print(f"Predicted pipette focus value: {focus_value:.2f} microns")
    
    label = f"Focus: {focus_value:.2f} µm"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Pipette Focus Inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
