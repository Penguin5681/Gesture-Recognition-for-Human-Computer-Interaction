import cv2
from HandTrackingModule import HandDetector
import mediapipe as mp

cam = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    _, img = cam.read()
    hands, img = detector.findHands(img)    # Edges will be drawn
    # hands = detector.findHands(img, draw=False)     # Edges won't be drawn
    # print(len(hands))
    if hands:
        # Hand 1
        hand_1 = hands[0]
    cv2.imshow("Imag", img)
    cv2.waitKey(1)