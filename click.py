import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

rightClickCount = 0
leftClickCount = 0

prev_x = 0
prev_y = 0
alpha = 0.3  # Smoothing factor

pinch_detected = False

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

            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            scaling_factor = 6  # Vary the sens here, < 1 => Less sensitive : > 1 => More sensitive

            # Smooth out the mouse movement
            curr_x = int(index_x * scaling_factor)
            curr_y = int(index_y * scaling_factor)
            smooth_x = int((1 - alpha) * prev_x + alpha * curr_x)
            smooth_y = int((1 - alpha) * prev_y + alpha * curr_y)
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x = smooth_x
            prev_y = smooth_y

            pinch_distance = find_distance((thumb_x, thumb_y), (index_x, index_y))
            if pinch_distance < 40:
                if not pinch_detected:
                    pinch_detected = True
                    if results.multi_handedness[0].classification[0].label == "Right":
                        pyautogui.rightClick()
                        rightClickCount += 1
                        cv2.putText(frame, "Right Click Triggered", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 36, 102), 2)
                        print(f"Right Click Count: {rightClickCount}")
                    elif results.multi_handedness[0].classification[0].label == "Left":
                        pyautogui.click()
                        leftClickCount += 1
                        cv2.putText(frame, "Left Click Triggered", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 36, 102), 2)
                        print(f"Left Click Count: {leftClickCount}")
            else:
                pinch_detected = False

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand-Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
