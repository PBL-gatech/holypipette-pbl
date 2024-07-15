import time
import cv2
import numpy as np
from holypipette.devices.manipulator.microscope import Microscope
from holypipette.devices.manipulator import Manipulator
from holypipette.devices.camera import Camera
from threading import Thread
import math

class FocusHelper():
    '''A helper class to aid with microscope focusing
    '''
    
    FOCUSING_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 10000

    def __init__(self, microscope: Microscope, camera: Camera):
        self.microscope: Microscope = microscope
        self.camera : Camera = camera

    def autofocusContinuous(self, distance):
        '''tell the stage to go a certain (larger) distance at a low max speed.
           Take a bunch of pictures and determine focus score.  Finally,
           move the stage to the position with the best focus score 
        '''

        #move the microscope a certain distance forward
        commandedPos = self.microscope.position() + distance
        self.microscope.absolute_move(commandedPos)

        #start recording focus values and positions
        focusThread = FocusUpdater(self.microscope, self.camera)
        focusThread.start()

        #wait for the microscope to reach the pos
        posTimeArr = []
        currPos = self.microscope.position()
        while abs(currPos - commandedPos) > 0.3:
            currPos = self.microscope.position()
            posTimeArr.append((time.time(), currPos))
            time.sleep(0.1)

        posTimeArr = np.array(posTimeArr)

        #stop the focus recording thread
        focusThread.stop()

        #wait for focus thread to stop
        while not focusThread.didFinish:
            # print('waiting for thread finish...')
            time.sleep(0.01)

        #find index with best score
        bestIndex = np.argmax(focusThread.posFocusList[:, 1])

        #find results for that index
        bestScore = focusThread.posFocusList[bestIndex][1]
        bestPos = focusThread.posFocusList[bestIndex][0]
        
        #return best score, position
        return bestPos, bestScore

    def autofocus(self, dist=500):
        '''Attempts to auto focus the micrscope image by moving in the z-axis
        '''

        self.microscope.set_max_speed(self.FOCUSING_MAX_SPEED)
        initPos = self.microscope.position()
        bestForwardPos, bestForwardScore = self.autofocusContinuous(dist)
        
        self.microscope.set_max_speed(self.NORMAL_MAX_SPEED)
        self.microscope.absolute_move(initPos)
        self.microscope.wait_until_still()
        self.microscope.set_max_speed(self.FOCUSING_MAX_SPEED)

        bestBackwardPos, bestBackwardScore = self.autofocusContinuous(-dist)
        self.microscope.set_max_speed(self.NORMAL_MAX_SPEED)

        if bestBackwardScore < bestForwardScore:
            self.microscope.absolute_move(bestForwardPos)
        else:
            self.microscope.absolute_move(bestBackwardPos)

        self.microscope.wait_until_still()

class FocusUpdater(Thread):
    def __init__(self, microscope: Microscope, camera: Camera):
       Thread.__init__(self)

       self.isRunning : bool = True
       self.camera : Camera = camera
       self.microscope : Microscope = microscope
       self.lastFrame : int = 0
       self.posFocusList : list = []
       self.didFinish : bool = False

    def run(self):
        '''continuously read frames from camera, and record their focus and the microscope's z position.  Assumes constant velocity!
        '''
        while self.isRunning:
            while self.lastFrame == self.camera.get_frame_no() and self.isRunning:
                # print('waiting for new frame')
                time.sleep(0.01) #wait for a new frame to be read from the camera
            self.lastFrame = self.camera.get_frame_no()
            
            #get focus score from frame
            _, frametime, _, img = self.camera.raw_frame_queue[0]
            score = self._getFocusScore(img)

            #append to list
            self.posFocusList.append([self.microscope.position(), score])
        
        self.posFocusList = np.array(self.posFocusList)
        self.didFinish = True #create a flag when we creating the arr


    def _getFocusScore(self, image) -> float:
        '''Get a score stating how focused a given image is
        Higher Score == more focused image
        '''

        focusSize = 512
        x = image.shape[1]/2 - focusSize/2
        y = image.shape[0]/2 - focusSize/2
        crop_img = image[int(y):int(y+focusSize), int(x):int(x+focusSize)]

        xEdges = cv2.norm(cv2.Sobel(src=crop_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=7))
        yEdges = cv2.norm(cv2.Sobel(src=crop_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=7))
        score = xEdges ** 2 + yEdges ** 2
        # score = cv2.Laplacian(crop_img, cv2.CV_32F).var()
        
        return xEdges


    def stop(self):
        self.isRunning = False


class StageCalHelper():
    '''A helper class to aid with Stage Calibration
    '''
    
    CAL_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 10000

    def __init__(self, stage: Manipulator, camera: Camera, frameLag: int):
        self.stage : Manipulator = stage
        self.camera : Camera = camera
        self.lastFrameNo : int = None
        self.frameLag = frameLag

    def calibrateContinuous(self, distance, video=False):
        '''Tell the stage to go a certain distance at a low max speed.
           Take a bunch of pictures and run optical flow. Use optical flow information
           to create a linear transform from stage microns to image pixels.
           if set, video creates an mp4 of the optical flow running in the project directory.
        '''
        #move the microscope a certain distance forward and up
        currPos = self.stage.position()
        commandedPos = np.array([currPos[0] + distance, currPos[1] - distance])
        axes = np.array([0, 1], dtype=int)
        self.stage.absolute_move_group(commandedPos, axes)

        #wait for the microscope to reach the pos, recording frames
        framesAndPoses = []
        currPos = self.stage.position()
        startPos = currPos
        _, _, _, firstFrame = self.camera.raw_frame_queue[0]
        p0 = self.calcOpticalFlowP0(firstFrame)
        while abs(currPos[0] - commandedPos[0]) > 0.3 or abs(currPos[1] - commandedPos[1]) > 0.3:
            while self.lastFrameNo == self.camera.get_frame_no():
                time.sleep(0.05) #wait for a new frame to be read from the camera
            self.lastFrameNo = self.camera.get_frame_no()
            currPos = self.stage.position()

            #get latest img
            _, _, _, frame = self.camera.raw_frame_queue[0]

            framesAndPoses.append([frame.copy(), currPos[0] - startPos[0], currPos[1] - startPos[1]])

        #run optical flow on the recorded frames
        print('running optical flow...')
        imgPosStagePosList = []
        x_pix_total = 0
        y_pix_total = 0

        if video:
            out = cv2.VideoWriter('opticalFlow.mp4', -1, 10.0, (1024,1024))

        #calculate the average image
        avgImg = np.zeros_like(framesAndPoses[0][0], dtype=np.float64)
        for frame, _, _ in framesAndPoses:
            avgImg += frame
        avgImg = avgImg / len(framesAndPoses)
        avgImg = avgImg.astype(np.uint8)

        #subtract average from all frames
        for frame, _, _ in framesAndPoses:
            frame -= avgImg

        for i in range(len(framesAndPoses) - 1):
            currFrame, x_microns, y_microns = framesAndPoses[i + 1]
            lastFrame, last_x_microns, last_y_microns = framesAndPoses[i]

            p0 = self.calcOpticalFlowP0(lastFrame)

            x_pix, y_pix = self.calcOpticalFlow(lastFrame, currFrame, p0)
            x_pix_total += x_pix
            y_pix_total += y_pix

            if math.isnan(x_pix) or math.isnan(y_pix): #if no corners can be found with optical flow, nan could be returned.  Don't add this to the list
                continue

            if video:
                vidFrame = cv2.cvtColor(currFrame.copy(), cv2.COLOR_GRAY2BGR)
                cv2.line(vidFrame, (512,512), (512 + int(x_pix_total), 512 + int(y_pix_total)), (255,0,0), 3)
                cv2.line(vidFrame, (600,100), (600 + int(x_pix_total), 100 + int(y_pix_total)), (255,0,0), 3)
                cv2.line(vidFrame, (900,400), (900 + int(x_pix_total), 400 + int(y_pix_total)), (255,0,0), 3)
                out.write(vidFrame)


            imgPosStagePosList.append([x_pix_total, y_pix_total, x_microns, y_microns])
        imgPosStagePosList = np.array(imgPosStagePosList)
        

        if video:
            out.release()
        
        #for some reason, estimateAffinePartial2D only works with int64
        #we can multiply by 100, to preserve 2 decimal places without affecting rotation / scaling portion of affline transform
        imgPosStagePosList = (imgPosStagePosList).astype(np.int64)
        print(imgPosStagePosList)

        #compute affine transformation matrix
        mat, inVsOut = cv2.estimateAffinePartial2D(imgPosStagePosList[:,2:4], imgPosStagePosList[:,0:2])

        #fix intercept - set image center --> stage center
        mat[0,2] = 0
        mat[1,2] = 0

        print('completed optical flow. matrix:')
        print(mat)

        #return transformation matrix
        return mat

    def calcOpticalFlowP0(self, firstFrame):
        #params for corner detector
        feature_params = dict(maxCorners = 100,
                                qualityLevel = 0.1,
                                minDistance = 10,
                                blockSize = 10)
        p0 = cv2.goodFeaturesToTrack(firstFrame, 70, 0.05, 25)
        # p0 = cv2.goodFeaturesToTrack(firstFrame, mask = None, **feature_params)

        return p0
    
    def calcMotionTranslation(self, lastFrame, currFrame):
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

        #compute transformation for image translation
        (cc, warp_matrix) = cv2.findTransformECC (lastFrame, currFrame,warp_matrix, warp_mode, criteria)

        #get x, y translation (pixels)
        print(warp_matrix)
        x_pix = warp_matrix[0,2]
        y_pix = warp_matrix[1,2]

        print(x_pix, y_pix)

        return x_pix, y_pix

    def calcOpticalFlow(self, lastFrame, currFrame, p0):
        #params for optical flow
        lk_params = dict(winSize  = (20, 20),
                    maxLevel = 20)

        # calculate optical flow from first frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(lastFrame, currFrame, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
         #find median movement vector
        dMovement = good_new - good_old
        medianVect = np.median(dMovement, axis=0)
        
        return medianVect[0], medianVect[1]


    def calibrate(self, dist=500):
        '''Calibrates the microscope stage using optical flow and stage encoders to create a um -> pixels transformation matrix
        '''

        self.stage.set_max_speed(self.CAL_MAX_SPEED)
        # self.stage.set_max_accel(10)

        initPos = self.stage.position()
        print('starting optical flow')
        mat = self.calibrateContinuous(dist)
        # commandedPos = np.array([initPos[0] + 200, initPos[1] - 200])
        # axes = np.array([0, 1], dtype=int)
        # self.stage.absolute_move_group(commandedPos, axes)
        self.stage.wait_until_still()
        currPos = self.stage.position()
        
        self.stage.set_max_speed(self.NORMAL_MAX_SPEED)
        self.stage.absolute_move(initPos)
        self.stage.wait_until_still()

        return mat