import math
import cv2
import mediapipe as mp
import time
import numpy as np

balls = []
gravity = 0
brush_color = (0, 255, 0)  # Default color is green
eraser_mode = False

def on_key_pressed(key):
    global eraser_mode
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == ord('e'):
        eraser_mode = not eraser_mode
    elif key == ord('f'):
        toggle_fullscreen()

def on_color_change(_):
    global brush_color
    brush_color = (cv2.getTrackbarPos('Blue', 'Color Selector'),
                   cv2.getTrackbarPos('Green', 'Color Selector'),
                   cv2.getTrackbarPos('Red', 'Color Selector'))

def toggle_fullscreen():
    current_state = cv2.getWindowProperty("Drawing App", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Drawing App", cv2.WND_PROP_FULLSCREEN, not current_state)

# Set up video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Drawing App", cv2.WINDOW_NORMAL)  # Make the window resizable
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Create trackbars for color selection
cv2.namedWindow('Color Selector')
cv2.createTrackbar('Blue', 'Color Selector', 0, 255, on_color_change)
cv2.createTrackbar('Green', 'Color Selector', 0, 255, on_color_change)
cv2.createTrackbar('Red', 'Color Selector', 0, 255, on_color_change)


# Initialize the timestamp
pTime = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]  # Assuming only one hand for simplicity
        landmarks = np.array([[lm.x, lm.y] for lm in handLms.landmark])

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        index_cx, index_cy = int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0])
        thumb_cx, thumb_cy = int(thumb_tip[0] * frame.shape[1]), int(thumb_tip[1] * frame.shape[0])

        is_hand_closed = landmarks[5, 1] < landmarks[8, 1]  # Check if the hand is closed

        if is_hand_closed:
            eraser_mode = True
        else:
            eraser_mode = False

        if is_hand_closed:
            balls = [ball for ball in balls if
                     np.linalg.norm(np.array(ball['position']) - np.array([index_cx, index_cy])) >= 90]
        else:
            balls.append({
                'position': [index_cx, index_cy],
                'velocity': [0, 0],
                'color': (255, 255, 255) if eraser_mode else brush_color
            })

        for ball in balls:
            ball['position'][0] += ball['velocity'][0]
            ball['position'][1] += ball['velocity'][1]
            ball['velocity'][1] += gravity

            cv2.circle(frame, (int(ball['position'][0]), int(ball['position'][1])), 10, ball['color'], cv2.FILLED)

        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    cv2.putText(frame, f"Eraser Mode: {eraser_mode}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
    cv2.imshow("Drawing App", frame)

    key = cv2.waitKey(1)
    on_key_pressed(key)

# Release resources
cap.release()
cv2.destroyAllWindows()
