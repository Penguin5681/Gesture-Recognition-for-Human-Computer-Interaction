import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

rightClickCount = 0
leftClickCount = 0

smooth_factor = 5  # larger value = more smoothing
prev_x = 0
prev_y = 0


def find_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5


# Main loop
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_x, thumb_y = int(
                thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(
                index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            smooth_x = (prev_x * (smooth_factor - 1) + index_x) / smooth_factor
            smooth_y = (prev_y * (smooth_factor - 1) + index_y) / smooth_factor
            prev_x = smooth_x
            prev_y = smooth_y

            pinch_distance = find_distance(
                (thumb_x, thumb_y), (index_x, index_y))
            if pinch_distance < 40:
                if results.multi_handedness[0].classification[0].label == "Right":
                    if rightClickCount == 0:
                        pyautogui.rightClick()
                        rightClickCount = 1
                    cv2.putText(frame, "Right Click Triggered", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 36, 102), 2)
                    print(f"Right Click Count: {rightClickCount}")
                elif results.multi_handedness[0].classification[0].label == "Left":
                    if leftClickCount == 0:
                        pyautogui.click()
                        leftClickCount = 1
                    cv2.putText(frame, "Left Click Triggered", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 36, 102), 2)
                    print(f"Left Click Count: {leftClickCount}")
            else:
                rightClickCount = 0
                leftClickCount = 0
                # adjust the scaling factor here, < 1 -> less mouse sens : > 1 -> more mouse sens
                scaling_factor = 3
                pyautogui.moveTo(int(smooth_x * scaling_factor), int(smooth_y * scaling_factor))
                # pyautogui.moveTo(int(smooth_x), int(smooth_y))

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand-Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()