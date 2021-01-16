import numpy as np
import imutils
import time
import cv2

prototext = 'MobileNetSSD_deploy.prototxt.txt'
model= 'mobilenet_iter_73000.caffemodel'
confthresh = 0.2

CLASSES = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dinningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmoniter']
COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))

print("loading model...")
net = cv2.dnn.readNetFromCaffe(prototext, model)
print("model loaded")
print("Starting camera feed")

cam = cv2.VideoCapture(0) # put link if you are using Ip webcam app else put(0)
time.sleep(2)

while True:
    _,frame= cam.read()
    frame = imutils.resize(frame, width=500)
    h,w = frame.shape[:2]
    
    imresize = cv2.resize(frame,(300,300))
    blob = cv2.dnn.blobFromImage(imresize,0.007843,(300,300),127.5)

    net.setInput(blob)
    detections = net.forward()
    det_shape = detections.shape[2]
    
    for i in np.arange(0,det_shape):
        confidence =detections[0,0,i,2]
        if confidence > confthresh:
            idx = int(detections[0,0,i,1])
            print('class id :',detections[0,0,i,1])
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy)=box.astype('int')
            label ="{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startx,starty),(endx,endy),COLORS[idx],2)
            if startx-15 > 15:
                y = starty-15
            else:
                y = starty+15
            cv2.putText(frame,label,(startx,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
    cv2.imshow("camera feed",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()            