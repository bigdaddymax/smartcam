import cv2
import os
import numpy as np
import dlib
from datetime import datetime
import time

ssdProtocol = '../MobileNetSSD_deploy.prototxt'
ssdModel    = '../MobileNetSSD_deploy.caffemodel'

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

classesOfInterest = [3, 6, 7, 8, 10, 12, 13, 14, 15, 17]
tracker = None
frameNum = 0
filename = ''
writer = None
label  = ''
timestamp = 0.0

fourcc = cv2.VideoWriter_fourcc(*'H264')


def loadNet(protocol, model):
    return cv2.dnn.readNetFromCaffe(protocol, model)

def updateTracker(frame, label):
    global tracker
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracker.update(rgb)
    pos = tracker.get_position()
    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return frame

def startTracking(rect, frame):
    global tracker
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracker.start_track(rgb, rect)

def detect(net, frame, directory):
    global tracker, frameNum, writer, fourcc, label, timestamp
    
    if frame is None:
        return
    (h, w) = frame.shape[:2]
    resizedFrame = cv2.resize(frame, (300, 300))

    frameNum = frameNum + 1
    
    frameRate = 6
    if timestamp != 0:
    #    duration = time.clock() - timestampsif timestamp != 0.0:
        frameRate = 1/(time.time() - timestamp)
    #print(directory + ': ' + str(time.time() - timestamp))
    #print(directory + ' ' + str(frameRate) + ' ' + str(frameNum))
    timestamp = time.time()
    if tracker is None:
        if frameNum < 5:
            return frame
        frameNum = 0
        blob   = cv2.dnn.blobFromImage(resizedFrame,  0.007843, (300, 300), (127, 127, 127), False)
        
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6 and int(detections[0, 0, i, 1]) in classesOfInterest:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = CLASSES[int(detections[0, 0, i, 1])]
                print(directory + ' ' + label)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                startTracking(rect, frame)
                if writer is None:
                    if not os.path.exists('video/' + directory + '/' + datetime.now().strftime("%Y-%m-%d")):
                        os.makedirs('video/' + directory + '/' + datetime.now().strftime("%Y-%m-%d"))
                    writer = cv2.VideoWriter('video/' + directory + '/' + datetime.now().strftime("%Y-%m-%d") + '/' + datetime.now().strftime("%H-%M-%S")+ '.avi',fourcc, frameRate, (w, h), True)
                writer.write(frame)
                return frame
        tracker = None
        if writer is not None:
            writer.release()
            writer = None
        return frame
    frame = updateTracker(frame, label)
    writer.write(frame)
    #print('Write one of next frames', frame.shape[:2])
    return frame
