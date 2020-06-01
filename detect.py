import cv2
import os
import numpy as np
import dlib
from datetime import datetime
import time

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

class detector:
    """Class that runs detections and objects tracking for a frame"""
    trackers = {} 
    writer   = None
    camName  = None
    fourcc   = None   
    classesOfInterest = [3, 6, 7, 8, 10, 12, 13, 14, 15, 17]
    objColor = {
        3: (0, 255, 0),
        6: (200, 200,0),
        7: (200, 255, 100),
        8: (0, 255, 200),
        10: (0, 0, 200),
        12: (100, 0, 255),
        13: (200, 0, 255),
        14: (255, 100, 150),
        15: (150, 150, 100),
        17: (0, 150, 255)}
    frameNum    = 0
    fps         = 6 
    timestamp   = 0
    mysql       = None

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    def __init__(self, camName, protocol, model):
        self.fourcc     = cv2.VideoWriter_fourcc(*'H264')
        self.net        = cv2.dnn.readNetFromCaffe(protocol, model)
        self.camName    = camName                

    def updateTrackers(self, frame):
        for idx, tracker in self.trackers.items():
            frame = self.updateTracker(frame, idx)
        return frame

    def updateTracker(self, frame, idx):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.trackers[idx].update(rgb)
        pos     = self.trackers[idx].get_position()
        startX  = int(pos.left())
        startY  = int(pos.top())
        endX    = int(pos.right())
        endY    = int(pos.bottom())
        cv2.rectangle(frame, (startX, startY), (endX, endY), self.objColor[idx], 2)
        cv2.putText(frame, self.label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.objColor[idx], 2)
        return frame

    def startTracker(self, frame, startX, startY, endX, endY, idx):
        self.trackers[idx] = dlib.correlation_tracker()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(startX, startY, endX, endY)
        self.trackers[idx].start_track(rgb, rect)

    def detect(self, frame, fps):
    
        if frame is None:
            return
        #Count frames for FPS calculation
        self.frameNum = self.frameNum + 1    
        if self.timestamp != 0:
            self.fps = 1/(time.time() - self.timestamp)
        self.timestamp = time.time()
        
        #If there were no dlib correlation trackers or we just got 20 frames passed
        if not self.trackers  or self.frameNum == 20:
            if self.frameNum < 5:
                return frame
            self.frameNum = 0 
            
            detections = self.detectObjects(frame)
            if not detections:
                self.trackers = {}
                return frame
            for idx, box in detections.items():
                self.label = self.CLASSES[idx]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.objColor[idx], 2)
                cv2.putText(frame, self.label, (box[0], box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.objColor[idx], 2)
                self.startTracker(frame, box[0], box[1], box[2], box[3], idx)
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.writeFrame(frame)
            return frame
        frame = self.updateTrackers(frame)
        self.writeFrame(frame, fps)
        return frame


    def writeFrame(self, frame, fps):
        """Write a frame to a file. If there is no file handler, creates new one.
          File will be located in video/{camName}/{date}/{time}.avi
        """
        if self.writer is None:
            if not os.path.exists('video/' + self.camName + '/' + datetime.now().strftime("%Y-%m-%d")):
                os.makedirs('video/' + self.camName + '/' + datetime.now().strftime("%Y-%m-%d"))
            (h, w) = frame.shape[:2]
            self.writer = cv2.VideoWriter('video/' + self.camName + '/' + datetime.now().strftime("%Y-%m-%d") + '/' + datetime.now().strftime("%H-%M-%S")+ '.avi',self.fourcc, fps, (w, h), True)
        self.writer.write(frame)

    def detectObjects(self, frame):
        """Run objects detection on the frame. In case of detecting objects return list of arrays
           (boxes coordinates), object id and object type
        """
        (h, w) = frame.shape[:2]
        resizedFrame = cv2.resize(frame, (300, 300))
        blob   = cv2.dnn.blobFromImage(resizedFrame,  0.007843, (300, 300), (127, 127, 127), False)
        self.net.setInput(blob)
        detections = self.net.forward()
        detected   = dict() 
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.95 and int(detections[0, 0, i, 1]) in self.classesOfInterest:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                detected[idx] = box.astype("int")
        return detected
