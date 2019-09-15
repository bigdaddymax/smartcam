import cv2
from pprint import pprint
from threading import Thread
import time
import uuid

class threadCamReader:
    url         = ''
    frame       = None
    frameID     = None
    stream      = None
    fps         = 0    

    def __init__(self, url):
        self.url    = url
        self.stream = cv2.VideoCapture(self.url)
#        self.stream.set(cv2.CAP_PROP_FPS, 10)

    def start(self):
        Thread(target=self.run, args=()).start()
        return self

    def run(self):    
        """Runs a video stream reader in a separate thread.
           Depending on URL runs either RTSP or Multipart reader.
        """
        i = 0
        t = time.time()
        while True:
             i = i + 1
             ret, frame = self.stream.read()
             if (i == 20):
                 self.fps = 20/(time.time() - t)
                 t = time.time()
                 i = 0
             #If a frame is None need to re-init it: 
             # - close a stream;
             # - reopen it;
             # - read frame again
             if frame is None:
                 self.stream.release()
                 self.stream = cv2.VideoCapture(self.url)
                 ret, frame = self.stream.read()
             self.frame = frame
             self.frameID = uuid.uuid4()

    def read(self):
          text = time.strftime('%Y-%m-%d %H:%M:%S')
          if (self.fps > 0):
              text = text + ' FPS: ' + str(round(self.fps))
          newFrame = cv2.putText(self.frame, text, (10, int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
          return newFrame, self.frameID
