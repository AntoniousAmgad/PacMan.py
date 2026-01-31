import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key

# -------------------------
# Configuration: Pac-Man "Twitch" Settings
# -------------------------
CAMERA_INDEX = 0

# TIGHTER GRIP:
# We brought these numbers closer to 0.5 (the center).
# You now only need to move 10% from the center to trigger a key.
LEFT_BOUND = 0.40   # Was 0.3
RIGHT_BOUND = 0.60  # Was 0.7
UP_BOUND = 0.40     # Was 0.3
DOWN_BOUND = 0.60   # Was 0.7

# -------------------------
# Setup
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
keyboard = Controller()

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

current_key = None

def trigger_key(key):
    global current_key
    if current_key == key:
        return
    if current_key is not None:
        keyboard.release(current_key)
    if key is not None:
        keyboard.press(key)
    current_key = key

# -------------------------
# Main Loop
# -------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

print("Pac-Man Mode Started")
print("High Sensitivity: Small movements will trigger keys.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw the "Tight" Grid
        cv2.line(frame, (int(w * LEFT_BOUND), 0), (int(w * LEFT_BOUND), h), (255, 255, 255), 1)
        cv2.line(frame, (int(w * RIGHT_BOUND), 0), (int(w * RIGHT_BOUND), h), (255, 255, 255), 1)
        cv2.line(frame, (0, int(h * UP_BOUND)), (w, int(h * UP_BOUND)), (255, 255, 255), 1)
        cv2.line(frame, (0, int(h * DOWN_BOUND)), (w, int(h * DOWN_BOUND)), (255, 255, 255), 1)

        detected_key = None
        status_text = "NEUTRAL"
        status_color = (200, 200, 200)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Center of the palm (Middle Finger MCP)
            cx = hand_landmarks.landmark[9].x
            cy = hand_landmarks.landmark[9].y

            # Calculate distance from center (0.5, 0.5)
            x_dist = abs(cx - 0.5)
            y_dist = abs(cy - 0.5)

            # -------------------------
            # LOGIC: Axis Dominance
            # -------------------------
            # We check which axis (X or Y) has the greater movement.
            # This prevents accidental "Left" turns when you are trying to go "Up".
            
            if x_dist > y_dist: 
                # Horizontal movement is dominant
                if cx < LEFT_BOUND:
                    detected_key = Key.left
                    status_text = "LEFT"
                    status_color = (0, 255, 255)
                elif cx > RIGHT_BOUND:
                    detected_key = Key.right
                    status_text = "RIGHT"
                    status_color = (0, 255, 255)
            else:
                # Vertical movement is dominant
                if cy < UP_BOUND:
                    detected_key = Key.up
                    status_text = "UP"
                    status_color = (0, 255, 0)
                elif cy > DOWN_BOUND:
                    detected_key = Key.down
                    status_text = "DOWN"
                    status_color = (0, 0, 255)

            # Visual Feedback for Hand Position
            cv2.circle(frame, (int(cx * w), int(cy * h)), 10, status_color, -1)

        trigger_key(detected_key)

        cv2.putText(frame, f"CMD: {status_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

        cv2.imshow("Pac-Man Controller", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    if current_key:
        keyboard.release(current_key)
    cap.release()
    cv2.destroyAllWindows()