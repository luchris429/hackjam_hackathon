import numpy as np
import cv2
import pyautogui
import os
import pygame

cap = cv2.VideoCapture(0)

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


def nothing(x):
    pass

cv2.namedWindow('image')
cv2.namedWindow('image2')
cv2.createTrackbar('H_MIN','image',28,255,nothing)
cv2.createTrackbar('S_MIN','image',62,255,nothing)
cv2.createTrackbar('V_MIN','image',11,255,nothing)
cv2.createTrackbar('H_MAX','image',55,255,nothing)
cv2.createTrackbar('S_MAX','image',255,255,nothing)
cv2.createTrackbar('V_MAX','image',174,255,nothing)

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
pygame.mixer.music.load("beep-02.wav")

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
        screen_center = [int((FRAME_WIDTH - center[0]) * WIDTH_RATIO), int(center[1] * HEIGHT_RATIO)]
        pyautogui.moveTo(screen_center[0] - WIDTH_OFFSET,screen_center[1] - HEIGHT_OFFSET)
        area = cv2.contourArea(c)
        if area > 3500 and not clicked:
            clicked = True
            pyautogui.click()
            pygame.mixer.music.play()
        elif area < 1000 and screen_center[1] < SCREEN_HEIGHT / 2 and not clicked:
            pyautogui.scroll(2)
        elif area < 1000 and screen_center[1] > SCREEN_HEIGHT / 2 and not clicked:
            pyautogui.scroll(-2)
        elif area < 3500 and clicked:
            clicked = False
    cv2.imshow('image2', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
