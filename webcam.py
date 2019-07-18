import urllib
import urllib.request    
import numpy as np
import cv2
from pprint import pprint
import threading
from queue import Queue

class threadCamReader(threading.Thread):
    url         = ''
    q           = None

    def __init__(self, url, queueSize = 10):
        threading.Thread.__init__(self)
        self.url = url
        self.q   = Queue(queueSize)

    def run(self):    
        """Runs a video stream reader in a separate thread.
           Depending on URL runs either RTSP or Multipart reader.
        """
        if self.url.find('rtsp') != -1:
            self.rtspReader()
        self.frameReader()


    def rtspReader(self):
        """Runs an RTSP reader:
           - opens a stream;
           - reads a fream;
           - pushes a frame to Queue
        """
        stream = cv2.VideoCapture(self.url)
        while True:
            if not self.q.full():
                try:
                    ret, frame = stream.read()
                    #If a frame is None need to re-init it: 
                    # - close a stream;
                    # - reopen it;
                    # - read frame again
                    if frame is None:
                         stream.release()
                         stream = cv2.VideoCapture(self.url)
                         ret, frame = stream.read()
                    self.q.put(frame)
                except:
                    stream = cv2.VideoCapture(self.url) 

    def frameReader(self):
        """Runs a multipart readers from URL
           and pushes to Queue
        """
        stream = urllib.request.urlopen(self.url)
        bts = b''
        while True:
            if not self.q.full():
                try:
                    bts += stream.read(1024)
                except:
                    stream = urllib.request.urlopen(self.url)
                a = bts.find(b'\xff\xd8')
                b = bts.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bts[a:b+2]
                    bts = bts[b+2:]
                    self.q.put(cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR ))        
class webcam:
    url = '';
    def __init__(self, url):
        self.url = url

    def getFrame(self):
       if self.url.find('rtsp') != -1:
            yield from self.rtspReader()
       yield from  self.frameReader()

    def rtspReader(self):
        stream = cv2.VideoCapture(self.url)
        print(stream.get(cv2.CAP_PROP_FPS))
        while True:#(stream.isOpened()):
            ret, frame = stream.read()
            yield(frame)

    def frameReader(self):
        stream = urllib.request.urlopen(self.url)
        bts = b''
        while True:
            bts += stream.read(1024)
            a = bts.find(b'\xff\xd8')
            b = bts.find(b'\xff\xd9')
            if a != -1 and b != -1:
                 jpg = bts[a:b+2]
                 bts = bts[b+2:]
                 yield(cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR ))

    def prepMultipart(self, frame):
        jpg = cv2.imencode('.jpg', frame)[1]
        return b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'


