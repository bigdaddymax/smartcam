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

ssdProtocol = '../MobileNetSSD_deploy.prototxt'
ssdModel    = '../MobileNetSSD_deploy.caffemodel'

cams = { 
    'cam1': 'http://192.168.0.109:8081',
    'cam2': 'http://192.168.0.108/videostream.cgi?user=cam2&pwd=bla%20bla%20cam',
    'cam3': 'rtsp://cam3:ux4pOi6GSf@192.168.0.104:88/videoSub'
}

imagezmqPort = 5555
ports = []
output = {}

def prepMultipart(frame):
   jpg = cv2.imencode('.jpg', frame)[1]
   return b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'

def runDetectionsOnCam(url, camName):
    newFrame = None
    detector  = detect.detector(camName, ssdProtocol, ssdModel)
    ##detector = detect_motion.detect_motion()
    cam = webcam.threadCamReader(url)
    cam.start()
    sender = imagezmq.ImageSender(connect_to='tcp://*:555' + camName[-1:] ,block = False)    
    i = 0
    t = time.time()
    s = 0
    readFrameID = None
    while True:
        i = i + 1
        frame, frameID    = cam.read()
   
        if frame is None or frameID is None or readFrameID == frameID:
            #print(camName + ' ' + str(frameID))        
            continue

        readFrameID = frameID

        newFrame = detector.detect(frame)
        
        
        if newFrame is not None:
            s = s + sys.getsizeof(newFrame)
            if time.time() > t:
                #print(camName + ' '  + str(round(s / (time.time() - t))/ (1024 * 1024)) + ' Mb/s')       
                s = 0 
                t = time.time()
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
        camName, frame = receiver.recv_image()
    #    print(camName)
#        receiver.send_reply(b'OK')
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


