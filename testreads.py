import webcam

cam = webcam.threadCamReader('http://192.168.0.108/videostream.cgi?user=cam2&pwd=bla%20bla%20cam')
cam.start()
i = 0
while True:
    frame = cam.read()
