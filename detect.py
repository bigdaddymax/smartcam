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
    tracker = None
    writer  = None
    camName = None
    fourcc  = None   
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

    def updateTracker(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tracker.update(rgb)
        pos     = self.tracker.get_position()
        startX  = int(pos.left())
        startY  = int(pos.top())
        endX    = int(pos.right())
        endY    = int(pos.bottom())
        cv2.rectangle(frame, (startX, startY), (endX, endY), self.objColor[idx], 2)
        cv2.putText(frame, self.label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.objColor[idx], 2)
        return frame

    def startTracker(self, frame, startX, startY, endX, endY):
        self.tracker = dlib.correlation_tracker()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(startX, startY, endX, endY)
        self.tracker.start_track(rgb, rect)

    def detect(self, frame):
    
        if frame is None:
            return
        (h, w) = frame.shape[:2]
        resizedFrame = cv2.resize(frame, (300, 300))

        self.frameNum = self.frameNum + 1
    
        if self.timestamp != 0:
            self.fps = 1/(time.time() - self.timestamp)
        self.timestamp = time.time()
        if self.tracker is None or self.frameNum == 20:
            if self.frameNum < 5:
                return frame
            self.frameNum = 0 
            
            detections = self.detect(frame)
            for idx, box in detections:
                self.label = self.CLASSES[idx]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), self.objColor[idx], 2)
                cv2.putText(frame, self.label, (box[0], box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.objColor[idx], 2)
                self.startTracker(frame, box[0], box[1], box[2], box[3])
                self.writeFrame(frame)
                return frame
            self.tracker = None
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            return frame
        frame = self.updateTracker(frame)
        self.writer.write(frame)
        return frame


    def writeFrame(frame):
        """Write a frame to a file. If there is no file handler, creates new one.
          File will be located in video/{camName}/{date}/{time}.avi
        """
    
        if self.writer is None:
            if not os.path.exists('video/' + self.camName + '/' + datetime.now().strftime("%Y-%m-%d")):
                os.makedirs('video/' + self.camName + '/' + datetime.now().strftime("%Y-%m-%d"))
            self.writer = cv2.VideoWriter('video/' + self.camName + '/' + datetime.now().strftime("%Y-%m-%d") + '/' + datetime.now().strftime("%H-%M-%S")+ '.avi',self.fourcc, self.fps, (w, h), True)
        self.writer.write(frame)

    def detectObjects(frame):
        """Run objects detection on the frame. In case of detecting objects return list of arrays
           (boxes coordinates), object id and object type
        """
        (h, w) = frame.shape[:2]
        resizedFrame = cv2.resize(frame, (300, 300))
        blob   = cv2.dnn.blobFromImage(resizedFrame,  0.007843, (300, 300), (127, 127, 127), False)
        self.net.setInput(blob)
        detections = self.net.forward()
        detected   = {}
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7 and int(detections[0, 0, i, 1]) in self.classesOfInterest:
                idx = int(detections[0, 0, i, 1])
                if detected[idx] is null:
                    detected[idx] = []
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                detected[idx].append(box.astype("int"))
        return detected
