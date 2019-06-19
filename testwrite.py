import cv2
import webcam
import time

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#writer = cv2.VideoWriter('output.avi',fourcc, 20.0, (640, 480), True)
cam = webcam.webcam('http://192.168.0.109:8081')
frames = cam.getFrame()
num = 120
start = time.time()
for frame in frames:
     num = num - 1
     if num == 0:
         end = time.time()
         seconds = end - start
         fps  = 120 / seconds;
         print("Estimated frames per second : ", fps);

