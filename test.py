import numpy as np
import cv2

cap = cv2.VideoCapture(0)

H_MIN = 0
H_MAX = 256
S_MIN = 0
S_MAX = 256
V_MIN = 0
V_MAX = 256
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_NUM_OBJECTS = 10
MIN_OBJECT_AREA = 20*20
MAX_OBJECT_AREA = int(FRAME_HEIGHT * FRAME_WIDTH / 1.5)

windowName = "Original Image";
windowName1 = "HSV Image";
windowName2 = "Thresholded Image";
windowName3 = "After Morphological Operations";
trackbarWindowName = "Trackbars";


while(True):
        # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything done, release the capture
    #cap.release()
    #cv2.destroyAllWindows()
