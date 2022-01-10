import cv2
import numpy as np
import pyautogui

def max_cnt(contours):
    cnt=[]
    Max=0
    for i in range(len(contours)):
        area=cv2.contourArea(contours[i])
        if(area>Max):
            cnt=i
            Max=area
    return cnt

vid = cv2.VideoCapture(0) # starting a video capture value-0 means the integrated webcam.
prev_pos='neutral'

while(1):
    _, frame = vid.read()#it returns the tuple value of which second val is important.

    frame = cv2.flip(frame,1)# discarding the mirror image effect.(as it appears right in cam if we point towards left)  parameter value 1 signifies that..

    frame = frame[:300, 300:600]# crop-out the required part
    frame = cv2.GaussianBlur(frame,(5,5),0)# blurring the image

    lower_ = np.array([13,16,27]) # lower boundary for thresholding
    higher_ = np.array([87,105,125]) # upper boundary for thresholding

    mask = cv2.inRange(frame,lower_,higher_) # use of inRange func it sets the bit accordingly if the bit value is b/w upper and lower boundary.


    _, thres = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue

    max_contour=max(contours,key=cv2.contourArea)# we find the contour that has max area ,, , to find the area we pass the key as cv2.contourArea() func.
    cnt=max_cnt(contours)

    epsilon = 0.01 * cv2.arcLength(max_contour, True)
    max_contour = cv2.approxPolyDP(max_contour, epsilon, True)

    m=cv2.moments(max_contour)
    try:
        x=int( m['m10'] / m['m00'])
        y=int(m['m01'] /m['m00'])
    except ZeroDivisionError:
        continue

    frame=cv2.circle(frame,(x,y),10,(255,0,0),2)
    frame=cv2.drawContours(frame,[max_contour],-1,(255,0,0),3)

    frame=cv2.line(frame,(75,0),(75,299),(255,255,255),2)
    frame = cv2.line(frame, (225, 0), (225, 299), (255, 255, 255), 2)
    frame = cv2.line(frame, (75, 150), (225, 150), (255, 255, 255), 2)
    frame = cv2.line(frame, (75, 200), (225, 200), (255, 255, 255), 2)
    cv2.imshow('image', frame)

    if x<75:
        curr_pos='left'
    elif x>225:
        curr_pos='right'
    elif x > 75 and x < 225 and y<200:
        curr_pos='up'
    elif x > 75 and x < 225 and y>250:
        curr_pos='down'
    else:
        curr_pos='neutral'

    if(curr_pos!=prev_pos):
        if(curr_pos!='neutral'):
            pyautogui.press(curr_pos)
        prev_pos=curr_pos

    if cv2.waitKey(1) == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()