import onnxruntime as ort
import numpy as np
import cv2
import time
from pathlib import Path
from enum import Enum
import random
import os

class FocusLevels(Enum):
    IN_FOCUS = 0
    OUT_OF_FOCUS_UP = 1
    OUT_OF_FOCUS_DOWN = 2
    NO_PIPETTE = 3

class PipetteFocuser:
    def __init__(self):
        curFile = Path(__file__).parent.absolute()
        model_path = os.path.join(curFile, 'pipetteModel', 'pipetteFocusNet3.onnx')
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)

        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.imgSize = 640

        # Reduced classes
        self.focusedClasses = [0]  # classes where we can consider the cell "focused"
        self.outOfFocusUp = [1, 2, 3]  # pipette is below focal plane -- move up
        self.outOfFocusDown = [4, 5, 6]  # pipette is above focal plane -- move down
        self.noPipette = [7]

    def get_pipette_focus(self, img):
        '''Return a predicted focus level for the pipette in the image.'''
        # Resize image to imgSize
        img_resized = cv2.resize(img, (self.imgSize, self.imgSize))

        # Normalize image to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Convert image to CHW format (channels first)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW

        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)  # 1, C, H, W

        # Convert to contiguous array
        img_input = np.ascontiguousarray(img_batch)

        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: img_input})
        elapsed_time = time.time() - start_time
        print(f'Inference time: {elapsed_time:.4f} seconds')

        # Assuming outputs[0] contains class scores
        classes = outputs[0]
        print(f'Output shape: {classes.shape}')  # Debug statement

        # Handle different output shapes
        if classes.ndim == 2 and classes.shape[0] == 1:
            # Single prediction
            bestClass = np.argmax(classes, axis=1)[0]
        elif classes.ndim == 1:
            # Direct class scores
            bestClass = np.argmax(classes)
        else:
            # Multiple predictions; aggregate as needed
            bestClass = np.argmax(classes, axis=1)  # This will still be an array

        print(f'best class: {bestClass}')  # Debug statement

        # Ensure bestClass is a scalar integer
        if isinstance(bestClass, np.ndarray):
            if bestClass.size == 1:
                bestClass = int(bestClass[0])
            else:
                # Aggregate multiple predictions, e.g., majority vote
                unique, counts = np.unique(bestClass, return_counts=True)
                bestClass = unique[np.argmax(counts)]
                print(f'Aggregated best class: {bestClass}')

        if bestClass in self.focusedClasses:
            return FocusLevels.IN_FOCUS
        elif bestClass in self.outOfFocusUp:
            return FocusLevels.OUT_OF_FOCUS_UP
        elif bestClass in self.outOfFocusDown:
            return FocusLevels.OUT_OF_FOCUS_DOWN
        elif bestClass in self.noPipette:
            return FocusLevels.NO_PIPETTE
        else:
            print(f'ERROR: invalid class {bestClass}')
            return FocusLevels.NO_PIPETTE

if __name__ == '__main__':
    focuser = PipetteFocuser()
    dataPath = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\focus_dataset\focus_training_set\images"
    if not os.path.isdir(dataPath):
        print(f"The data path '{dataPath}' does not exist.")
        exit(1)

    files = os.listdir(dataPath)
    pngFiles = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    
    if not pngFiles:
        print("No image files found in the specified directory.")
        exit(1)

    # Grab a random image in the data folder
    randomPng = random.choice(pngFiles)
    print(f'random png: {randomPng}')
    
    # Correctly join the path
    imgPath = os.path.join(dataPath, randomPng)
    img = cv2.imread(imgPath)

    if img is None:
        print(f"Failed to read image: {imgPath}. Exiting.")
        exit(1)

    # Find pipette focus level
    focusLevel = focuser.get_pipette_focus(img)
    print(f'Focus level: {focusLevel}')

    # Optionally, annotate the image with focus level
    label = f'Focus: {focusLevel.name}'
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show img
    cv2.imshow("pipette finding test", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print("Exiting loop.")
    cv2.destroyAllWindows()
