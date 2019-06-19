import webcam
from werkzeug.wrappers import Request, Response
from pprint import pprint
import detect
import cv2
import numpy as np
import os
import psutil
from multiprocessing import Process, Manager
import time

ssdProtocol = '../MobileNetSSD_deploy.prototxt'
ssdModel    = '../MobileNetSSD_deploy.caffemodel'

cams = { 
    'cam1': 'http://192.168.0.109:8081',
    'cam2': 'http://192.168.0.108/videostream.cgi?user=max&pwd=deep%20purple',
    'cam3': 'rtsp://cam3:ux4pOi6GSf@192.168.0.104:88/videoMain'
}

cam = webcam.webcam('http://192.168.0.109:8081')
#cam = webcam.webcam('http://192.168.0.108/videostream.cgi?user=max&pwd=deep%20purple')
#cam = webcam.webcam('rtsp://cam3:ux4pOi6GSf@192.168.0.104:88/videoMain')

def runDetectionsOnCam(url, directory):
    net = detect.loadNet(ssdProtocol, ssdModel)
    cam = webcam.threadCamReader(url, 50)
    cam.start()
    
    while True:
        frame    = cam.q.get()
        newFrame = detect.detect(net, frame, directory)
        process  = psutil.Process(os.getpid())

if __name__ == '__main__':
    for directory, url in cams.items():
        p = Process(target=runDetectionsOnCam, args=(url,directory))
        p.start()
