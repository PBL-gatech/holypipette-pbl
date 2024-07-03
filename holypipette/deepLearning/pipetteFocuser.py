import cv2
import numpy as np
import time
from pathlib import Path
from enum import Enum

class FocusLevels(Enum):
	IN_FOCUS = 0
	OUT_OF_FOCUS_UP = 1
	OUT_OF_FOCUS_DOWN = 2
	NO_PIPETTE = 3

class PipetteFocuser():

	def __init__(self):
		curFile = str(Path(__file__).parent.absolute())
		self.yoloNet = cv2.dnn.readNetFromONNX(curFile + '/pipetteModel/pipetteFocusNet.onnx')

		layer_names = self.yoloNet.getLayerNames()
		self.output_layers = [layer_names[i-1] for i in self.yoloNet.getUnconnectedOutLayers()]

		self.imgSize = 512

		#orig net
		# self.focusedClasses = [0, 1, 7] #classes where we can consider the cell "focused"
		# self.outOfFocusUp = [2, 3, 4, 5, 6] #pipette is below focal plane -- move up
		# self.outOfFocusDown = [8, 9, 10, 11, 12] #pipette is above focal plane -- move down
		# self.noPipette = [13]

		#reduced classes
		self.focusedClasses = [0] #classes where we can consider the cell "focused"
		self.outOfFocusUp = [1, 2, 3] #pipette is below focal plane -- move up
		self.outOfFocusDown = [4, 5, 6] #pipette is above focal plane -- move down
		self.noPipette = [7]

	def get_pipette_focus(self, img):
		'''return a predicted focus level for the pipette in the image
		'''
		#resize image to imgSize
		# img = cv2.resize(img, (self.imgSize, self.imgSize))

		#normalize image
		img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
		cv2.imshow('normalized', img)
		cv2.waitKey(1)

		#make sure image has 3 channels
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

		#convert to blob
		blob = cv2.dnn.blobFromImage(img, 1, (self.imgSize, self.imgSize), (0, 0, 0), True, crop=False)
		print(f"max: {np.max(blob)}, min: {np.min(blob)}")
		
		#run blob through network
		self.yoloNet.setInput(blob)
		outputs = self.yoloNet.forward(self.output_layers)

		#find class with highest confidence
		classes = outputs[0]
		bestClass = np.argmax(classes, axis=1)
		print(f'best class: {bestClass}')
		
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
	import random
	import os
	focuser = PipetteFocuser()

	while True:

		dataPath = '/Users/nathanmalta/Downloads/pipetteFocusData/'
		files = os.listdir(dataPath)
		pngFiles = [f for f in files if f.endswith('.png')]
		#grab random image in data folder
		randomPng = random.choice(pngFiles)
		print(f'random png: {randomPng}')
		img = cv2.imread(dataPath + randomPng)

		#find pipette, draw location to img
		start = time.time()
		focusLevel = focuser.get_pipette_focus(img)
		print(f'framerate: {1 / (time.time() - start)}')

		print(f'focus level: {focusLevel}')

		#show img
		# cv2.imshow("pipette finding test", img)
		cv2.waitKey(0)
