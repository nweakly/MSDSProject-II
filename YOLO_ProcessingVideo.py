## YOLO Processing Video
## using pretrained YOLOv2 model
## adapted from https://github.com/markjay4k/YOLO-series/blob/master/part3-processing_video.py
## by Mark Jay


### Imports
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

### Model

#model options
option = {
    'model': 'cfg/yolov2.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 1.0
  }

tfnet= TFNet(option)

capture = cv2.VideoCapture('test_video/edited_6653902835505346918.mp4') # video file example
colors=[tuple(255*np.random.rand(3)) for i in range (5)] # random colors for drawing boxes

while (capture.isOpened()):
    stime=time.time() # start time
    ret, frame=capture.read()
    results=tfnet.return_predict(frame)
    if ret: #if video is playing draw prediction boxes
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence=result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame=cv2.rectangle(frame, tl, br, color, 2)
            frame=cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break




