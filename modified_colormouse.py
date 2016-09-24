import numpy as np
import cv2
import pyautogui

cap = cv2.VideoCapture(0)

H_MIN = 0
H_MAX = 256
S_MIN = 0
S_MAX = 256
V_MIN = 0
V_MAX = 256
FRAME_WIDTH = 640.0
FRAME_HEIGHT = 480.0
SCREEN_WIDTH = 1600.0
SCREEN_HEIGHT = 900.0
HEIGHT_RATIO = SCREEN_HEIGHT / FRAME_HEIGHT
WIDTH_RATIO = SCREEN_WIDTH / FRAME_WIDTH
MAX_NUM_OBJECTS = 10
MIN_OBJECT_AREA = 20*20
MAX_OBJECT_AREA = int(FRAME_HEIGHT * FRAME_WIDTH / 1.5)


def nothing(x):
    pass

cv2.namedWindow('image')
cv2.namedWindow('image2')
cv2.createTrackbar('H_MIN','image',77,255,nothing)
cv2.createTrackbar('S_MIN','image',69,255,nothing)
cv2.createTrackbar('V_MIN','image',0,255,nothing)
cv2.createTrackbar('H_MAX','image',91,255,nothing)
cv2.createTrackbar('S_MAX','image',255,255,nothing)
cv2.createTrackbar('V_MAX','image',174,255,nothing)
cv2.createTrackbar('H_MIN2','image',16,255,nothing)
cv2.createTrackbar('S_MIN2','image',79,255,nothing)
cv2.createTrackbar('V_MIN2','image',116,255,nothing)
cv2.createTrackbar('H_MAX2','image',44,255,nothing)
cv2.createTrackbar('S_MAX2','image',157,255,nothing)
cv2.createTrackbar('V_MAX2','image',225,255,nothing)

switch = '0: OFF \n1 : ON'
cv2.createTrackbar(switch,'image',0,1,nothing)

switch2 = '0: Movement \n1: Clicking'
cv2.createTrackbar(switch2,'image',0,1,nothing)
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

    h_min2 = cv2.getTrackbarPos('H_MIN2', 'image')
    h_max2 = cv2.getTrackbarPos('H_MAX2', 'image')
    s_min2 = cv2.getTrackbarPos('S_MIN2', 'image')
    s_max2 = cv2.getTrackbarPos('S_MAX2', 'image')
    v_min2 = cv2.getTrackbarPos('V_MIN2', 'image')
    v_max2 = cv2.getTrackbarPos('V_MAX2', 'image')


    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    lower2 = np.array([h_min2, s_min2, v_min2])
    upper2 = np.array([h_max2, s_max2, v_max2])

    on = cv2.getTrackbarPos(switch,'image')

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    if cv2.waitKey(1) & 0xFF == ord('q') or on:
        break

    if not cv2.getTrackbarPos(switch2,'image'):
        cv2.imshow('image2', mask)
    else:
        cv2.imshow('image2',mask2)
    # When everything done, release the capture
cv2.destroyAllWindows()

while(True):
        # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    contours2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        screen_center = [int((FRAME_WIDTH - center[0]) * WIDTH_RATIO), int(center[1] * HEIGHT_RATIO)]
    if on:
        pyautogui.moveTo(screen_center[0],screen_center[1])

    if len(contours2) > 0:
        c2 = max(contours2, key=cv2.contourArea)
        if cv2.contourArea(c2) > MIN_OBJECT_AREA:
            pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
