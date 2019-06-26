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
    'cam3': 'rtsp://cam3:ux4pOi6GSf@192.168.0.104:88/videoMain'
}

receiver = imagezmq.ImageHub()

def prepMultipart(frame):
   jpg = cv2.imencode('.jpg', frame)[1]
   return b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'

def runDetectionsOnCam(url, camName):
    net = detect.loadNet(ssdProtocol, ssdModel)
    cam = webcam.threadCamReader(url, 50)
    cam.start()
    sender = imagezmq.ImageSender()    
    while True:
        frame    = cam.q.get()
        newFrame = detect.detect(net, frame, camName)
        sender.send_image(camName, newFrame)
        process  = psutil.Process(os.getpid())

def montage():
    output = {}
    while True:
        camName, frame = receiver.recv_image()
        receiver.send_reply(b'OK')
        output[camName] = cv2.resize(frame, (640, 480))
        outFrame = np.concatenate(list(output.values())) 
        yield prepMultipart(outFrame) 

@Request.application
def application(request):
    return Response(montage(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    for camName, url in cams.items():
        p = Process(target=runDetectionsOnCam, args=(url,camName))
        p.start()
    run_simple('192.168.0.114', 4000, application)


