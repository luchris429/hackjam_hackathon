import numpy as np
import cv2
import pyautogui
import os
import pygame
from pymouse import PyMouse

cap = cv2.VideoCapture(0)
mouse = PyMouse()

H_MIN = 0
H_MAX = 256
S_MIN = 0
S_MAX = 256
V_MIN = 0
V_MAX = 256
FRAME_WIDTH = 640.0
FRAME_HEIGHT = 480.0
SCREEN_WIDTH = 2000.0
SCREEN_HEIGHT = 1200.0
WIDTH_OFFSET = 200
HEIGHT_OFFSET = 150
HEIGHT_RATIO = SCREEN_HEIGHT / FRAME_HEIGHT
WIDTH_RATIO = SCREEN_WIDTH / FRAME_WIDTH
MAX_NUM_OBJECTS = 10
MIN_OBJECT_AREA = 20*20
MAX_OBJECT_AREA = int(FRAME_HEIGHT * FRAME_WIDTH / 1.5)
CLICK_THRESHOLD = 4000
SCROLL_THRESHOLD = 1000


def nothing(x):
    pass

cv2.namedWindow('image')
cv2.namedWindow('image2')
cv2.createTrackbar('H_MIN','image',37,255,nothing)
cv2.createTrackbar('S_MIN','image',43,255,nothing)
cv2.createTrackbar('V_MIN','image',77,255,nothing)
cv2.createTrackbar('H_MAX','image',87,255,nothing)
cv2.createTrackbar('S_MAX','image',196,255,nothing)
cv2.createTrackbar('V_MAX','image',255,255,nothing)

switch = '0: OFF \n1 : ON'
cv2.createTrackbar(switch,'image',0,1,nothing)

while(True):
        # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
    h_min = cv2.getTrackbarPos('H_MIN', 'image')
    h_max = cv2.getTrackbarPos('H_MAX', 'image')
    s_min = cv2.getTrackbarPos('S_MIN', 'image')
    s_max = cv2.getTrackbarPos('S_MAX', 'image')
    v_min = cv2.getTrackbarPos('V_MIN', 'image')
    v_max = cv2.getTrackbarPos('V_MAX', 'image')


    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    on = cv2.getTrackbarPos(switch,'image')

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    if cv2.waitKey(1) & 0xFF == ord('q') or on:
        break

    cv2.imshow('image2', mask)
    # When everything done, release the capture
cv2.destroyAllWindows()
clicked = False
pygame.init()
pygame.mixer.music.load("click_one.wav")

while(True):
        # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        screen_x = int((FRAME_WIDTH - center[0]) * WIDTH_RATIO) - WIDTH_OFFSET
        screen_y = int(center[1] * HEIGHT_RATIO) - HEIGHT_OFFSET
        screen_x = 0 if screen_x < 0 else screen_x
        screen_x = SCREEN_WIDTH if screen_x > SCREEN_WIDTH else screen_x
        screen_y = 0 if screen_y < 0 else screen_y
        screen_y = SCREEN_HEIGHT if screen_y > SCREEN_HEIGHT else screen_y
        screen_center = [screen_x, screen_y]
        #pyautogui.moveTo(screen_center[0],screen_center[1])
        mouse.move(screen_x, screen_y)
        area = cv2.contourArea(c)
        if area > CLICK_THRESHOLD and not clicked:
            clicked = True
            mouse.click(screen_x, screen_y)
            pygame.mixer.music.play()
        elif area < SCROLL_THRESHOLD and screen_center[1] < SCREEN_HEIGHT / 2 and not clicked:
            mouse.scroll(vertical=2)
        elif area < SCROLL_THRESHOLD and screen_center[1] > SCREEN_HEIGHT / 2 and not clicked:
            mouse.scroll(vertical=-2)
        elif area < CLICK_THRESHOLD and clicked:
            clicked = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
