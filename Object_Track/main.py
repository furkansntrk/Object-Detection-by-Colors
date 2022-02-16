import cv2
import time
import numpy as np
from collections import deque

# storing center of the object
buffer_size = 50
pts = deque(maxlen= buffer_size)

# blue color range
colorLower = (90, 50, 70)
colorUpper = (128, 255, 255)

# capture
cap = cv2.VideoCapture("balon.mp4")



if cap.isOpened() == False:
    print('Error')
    
videoWidth = int(cap.get(3))
videoHeight = int(cap.get(4))
size = (videoWidth, videoHeight)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         60, size)

while True:
    ret, frame = cap.read()
    
    if ret == True:
        time.sleep(0.01)
        cv2.imshow("Video",frame)
        
        
        #blurred
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        
        #hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)
        
        # mask for blue
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        #cv2.imshow("Mask Image", mask)
        
        # delete noise that around mask
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        cv2.imshow('Mask + Genisleme',mask)
        
        #contours
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        if len(contours) > 0:
            
            c = max(contours, key = cv2.contourArea)
            
            rect = cv2.minAreaRect(c)
            ((x,y),(width,height), rotation) = rect
            
            s = 'x: {}, y: {}, width: {}, height: {}, rotation: {}'.format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            
            print(s)
            
            # box
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            #moment
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            #draw contour
            cv2.drawContours(frame, [box], 0, (0,255,255), 2)
            
            #draw point at the center
            cv2.circle(frame, center, 5, (255,0,255), -1)
            
            #print the information on the screen
            cv2.putText(frame, s, (25,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0,0,0), 2)
            #cv2.imshow('"Son',frame)
            
        #deque
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
            
            cv2.line(frame, pts[i-1], pts[i], (0,255,0), 3)
        
        cv2.imshow('Tespit', frame)
        result.write(frame)
            
    else: break    

    if cv2.waitKey(1) & 0xFF == ord('q'): break



cap.release()
cv2.destroyAllWindows()


