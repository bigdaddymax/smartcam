import webcam
from werkzeug.wrappers import Request, Response
from pprint import pprint
import detect
import cv2
import numpy as np
import os
import psutil

ssdProtocol = '../MobileNetSSD_deploy.prototxt'
ssdModel    = '../MobileNetSSD_deploy.caffemodel'
cam = webcam.webcam('http://192.168.0.109:8081')
#cam = webcam.webcam('http://192.168.0.108/videostream.cgi?user=cam2&pwd=bla%20bla%20cam')
net = detect.loadNet(ssdProtocol, ssdModel) 
#cam = webcam.webcam('rtsp://cam3:ux4pOi6GSf@192.168.0.104:88/videoMain')

def runCam(cam):
    net = detect.loadNet(ssdProtocol, ssdModel)
    while true:
        frames = cam.getFrame()
        for frame in frames:
             newFrame = detect.detect(net, frame) 


def writeStream():
    frames = cam.getFrame()
    for frame in frames:
        newFrame = detect.detect(net, frame)
        yield(cam.prepMultipart(newFrame))

@Request.application
def application(request):
    return Response(writeStream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
  from werkzeug.serving import run_simple
  run_simple('192.168.0.114', 4000, application)    
