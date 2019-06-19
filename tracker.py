import cv
import dlib

def track(tracker, frame, rect):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
