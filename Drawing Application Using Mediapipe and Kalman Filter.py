import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import time

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Kalman Filter Setup
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.array([[1e-4, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 1e-4, 0], [0, 0, 0, 1e-4]], np.float32)
kalman.measurementNoiseCov = np.array([[1e-1, 0], [0, 1e-1]], np.float32)

# Global Variables
pen_color = (0, 255, 0)  # Default green
pen_thickness = 4
canvas = None
frame_interval = 1 / 30  # Target 30 FPS
window_width, window_height = 1280, 720  # Default window size
message = ""
message_time = 0

# Function to apply Kalman filter
def apply_kalman_filter(prev_coords, current_coords):
    if prev_coords is None:
        return current_coords
    kalman.correct(np.array([current_coords[0], current_coords[1]], np.float32))
    prediction = kalman.predict()
    return int(prediction[0][0]), int(prediction[1][0])

# Function to process hand drawing
def process_hand(frame, results, is_drawing, prev_coords):
    global canvas, pen_color, pen_thickness

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            # Apply Kalman filter to smooth the coordinates
            smoothed_coords = apply_kalman_filter(prev_coords, (x, y))

            if is_drawing:
                if prev_coords is not None:
                    cv2.line(canvas, prev_coords, smoothed_coords, pen_color, pen_thickness)
                prev_coords = smoothed_coords
            else:
                prev_coords = None

            cv2.circle(frame, smoothed_coords, 8, (0, 0, 255), -1)

    return prev_coords

# Function to display temporary messages
def display_message(frame, message, message_time):
    if time.time() - message_time < 1.5:  # Display message for 1.5 seconds
        cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# Function to handle user input
def handle_user_input(key, is_drawing):
    global pen_color, pen_thickness, canvas, message, message_time
    if key == ord('d'):
        is_drawing = not is_drawing
        message = f"Drawing {'ON' if is_drawing else 'OFF'}"
        message_time = time.time()
    elif key == ord('c'):
        canvas.fill(0)
        message = "Canvas Cleared"
        message_time = time.time()
    elif key == ord('r'):
        pen_color = (0, 0, 255)  # Red
        message = "Pen Color: Red"
        message_time = time.time()
    elif key == ord('g'):
        pen_color = (0, 255, 0)  # Green
        message = "Pen Color: Green"
        message_time = time.time()
    elif key == ord('b'):
        pen_color = (255, 0, 0)  # Blue
        message = "Pen Color: Blue"
        message_time = time.time()
    elif key == ord('+') and pen_thickness < 10:
        pen_thickness += 1
        message = f"Pen Thickness: {pen_thickness}"
        message_time = time.time()
    elif key == ord('-') and pen_thickness > 1:
        pen_thickness -= 1
        message = f"Pen Thickness: {pen_thickness}"
        message_time = time.time()
    elif key == 27:  # ESC to exit
        return 'exit', is_drawing
    return None, is_drawing

# Function to process the video frame and perform drawing
def video_loop(cap, hands):
    global canvas, message, message_time
    is_drawing = False
    prev_coords = None
    last_time = time.time()

    while cap.isOpened():
        now = time.time()
        if now - last_time < frame_interval:
            continue
        last_time = now

        success, frame = cap.read()
        if not success:
            print("Cannot read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Resize frame to match the window size
        frame = cv2.resize(frame, (window_width, window_height))
        height, width, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Convert to RGB and process hand landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        prev_coords = process_hand(frame, results, is_drawing, prev_coords)

        # Blend the canvas with the frame
        blended = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display temporary messages
        display_message(blended, message, message_time)

        # Display the blended frame
        cv2.imshow("Drawing App", blended)

        # Handle user input
        key = cv2.waitKey(1) & 0xFF
        action, is_drawing = handle_user_input(key, is_drawing)
        if action == 'exit':
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    global canvas, window_width, window_height

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    # Dynamically get the webcam resolution and adjust window size
    window_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    window_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow("Drawing App", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drawing App", window_width, window_height)

    # Set OpenCV threading for better performance
    cv2.setNumThreads(4)

    with mp_hands.Hands(
        static_image_mode=False,  # Faster video processing
        max_num_hands=1,          # Detect only one hand
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        video_loop(cap, hands)

if __name__ == "__main__":
    main()
