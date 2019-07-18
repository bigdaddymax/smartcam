import webcam
from pprint import pprint
import detect
import os
import psutil
from multiprocessing import Process, Manager
import imagezmq
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import numpy as np
import cv2
ssdProtocol = '../MobileNetSSD_deploy.prototxt'
ssdModel    = '../MobileNetSSD_deploy.caffemodel'

cams = { 
    'cam1': 'http://192.168.0.109:8081',
    'cam2': 'http://192.168.0.108/videostream.cgi?user=cam2&pwd=bla%20bla%20cam',
    'cam3': 'rtsp://cam3:ux4pOi6GSf@192.168.0.104:88/videoSub'
}

imagezmqPort = 5555
ports = []

def prepMultipart(frame):
   jpg = cv2.imencode('.jpg', frame)[1]
   return b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'

def runDetectionsOnCam(url, camName):
    detector  = detect.detector(camName, ssdProtocol, ssdModel)
    cam = webcam.threadCamReader(url, 1)
    cam.start()
    sender = imagezmq.ImageSender(connect_to='tcp://*:555' + camName[-1:] ,block = False)    
    while True:
        frame    = cam.q.get()
        newFrame = detector.detect(frame)
        if newFrame is not None:
            sender.send_image(camName, newFrame)

def montage():
    output = {}
    receiver = imagezmq.ImageHub(open_port='tcp://localhost:5551', block = False)
    for name, url in cams.items():
        receiver.connect(open_port = 'tcp://127.0.0.1:555' + name[-1:])
    while True:
        camName, frame = receiver.recv_image()
#        print(camName)
#        receiver.send_reply(b'OK')
        output[camName] = cv2.resize(frame, (640, 480))
        outFrame = np.concatenate(list(output.values())) 
        yield prepMultipart(outFrame) 

@Request.application
def application(request):
    return Response(montage(), mimetype='multipart/x-mixed-replace; boundary=frame')

def startWeb():
    run_simple('192.168.0.114', 4000, application)

if __name__ == '__main__':
    for camName, url in cams.items():
        p = Process(target=runDetectionsOnCam, args=(url,camName))
        p.start()
    p1 = Process(target=startWeb, args=())
    p1.start()


