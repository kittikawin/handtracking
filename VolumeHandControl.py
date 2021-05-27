import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript

# control volume library

###################
wCam, hCam = 640.0, 480.0
###################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # width camera
cap.set(4, hCam)  # height camera
# c1 = cap.get(0), cap.get(1), cap.get(2), cap.get(3), cap.get(4)
# print(c1)

detector = htm.handDetector(detectionCon=0.7)

minVol = 0
maxVal = 100
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if success:
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList):
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # calculate middle between fingers (thumb, index)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)  # draw circle to the index finger
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)  # draw circle to the thumb finger
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # create line between that fingers
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)  # calculate length between fingers
            # Hand range 15 - 150
            vol = np.interp(length, [15, 150], [minVol, maxVal])
            volBar = np.interp(length, [15, 150], [400, 150])
            volPer = np.interp(length, [15, 150], [0, 100])
            osascript.osascript(f"set volume output volume {vol}")

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        cv2.waitKey(1)  # waiting 1 millisecond
