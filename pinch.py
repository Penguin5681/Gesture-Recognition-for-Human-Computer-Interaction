import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

pinch_threshold = 0.03


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    pinch_status = "Not Pinching"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            distance = thumb_tip.y - index_tip.y

            pinch = distance < pinch_threshold

            pinch_status = "Pinching" if pinch else "Not Pinching"

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Pinch Status: {pinch_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 36, 102), 2)

    cv2.imshow("Hand Tracking", frame)

    print(f"Pinch Status: {pinch_status}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
