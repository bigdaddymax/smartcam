import webcam
from pprint import pprint
import detect
import detect_motion
import os
import psutil
from multiprocessing import Process, Manager
import imagezmq
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import numpy as np
import cv2
import time
import threading
import sys
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

camNames = config.get('cameras', 'cameras').split()
cams     = {}
for camName in camNames:
   cams[camName] = config.get(camName, 'url')

#cams = { 
#    'cam1': 'http://192.168.0.109:8081',
#    #'cam2': 'http://192.168.0.108/videostream.cgi?user=cam2&pwd=bla%20bla%20cam',
#    'cam3': 'rtsp://cam3:ux4pOi6GSf@192.168.0.104:88/videoSub'
#}

imagezmqPort = 5555
ports = []
output = {}

def prepMultipart(frame):
   jpg = cv2.imencode('.jpg', frame)[1]
   return b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'

def runDetectionsOnCam(url, camName):
   global config

   newFrame = None
   detector  = detect.detector(camName, config.get('ssd', 'protocol'), config.get('ssd', 'model'))
   ##detector = detect_motion.detect_motion()
   cam = webcam.threadCamReader(url)
   cam.start()
   sender = imagezmq.ImageSender(connect_to='tcp://*:555' + camName[-1:] ,block = False)    
   readFrameID = None
   while True:
      time.sleep(1/10000)
      frame, frameID    = cam.read()
      #Just skip if we read the same frame or done have a frame at all
      if frame is None or frameID is None or readFrameID == frameID:
         #print(camName + ' ' + str(frameID))        
         continue
      
      readFrameID = frameID
      newFrame    = detector.detect(frame, cam.fps)
                
      if newFrame is not None:
         sender.send_image(camName, newFrame)

def readToWeb():
    while True:
        outFrame = np.concatenate(list(output.values()))
        yield prepMultipart(outFrame)

def montage():
    global output
    receiver = imagezmq.ImageHub(open_port='tcp://localhost:5551', block = False)
    for name, url in cams.items():
        receiver.connect(open_port = 'tcp://127.0.0.1:555' + name[-1:])
    while True:
         time.sleep(1/10000)   
         camName, frame = receiver.recv_image()
         (h, w) = frame.shape[:2]
         frame = cv2.copyMakeBorder(frame, round((w - h) / 2), round((w - h) / 2), 0, 0, (0,0,0,0))
         output[camName] = cv2.resize(frame, (640, 480))

@Request.application
def application(request):
    return Response(readToWeb(), mimetype='multipart/x-mixed-replace; boundary=frame')

def startWeb():
    thread = threading.Thread(target=montage)
    thread.daemon = True
    thread.start()    
    run_simple('192.168.0.114', 4000, application)

if __name__ == '__main__':
    for camName, url in cams.items():
        p = Process(target=runDetectionsOnCam, args=(url,camName))
        p.start()
    p1 = Process(target=startWeb, args=())
    p1.start()
    p1.join()


