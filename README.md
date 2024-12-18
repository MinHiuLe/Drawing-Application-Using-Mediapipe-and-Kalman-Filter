# Drawing App with Hand Gesture Recognition

This project implements a real-time drawing application that allows users to draw on a virtual canvas using their finger gestures tracked via webcam. The application leverages OpenCV for computer vision tasks, MediaPipe for hand tracking, and a Kalman Filter to smooth finger movements.

---

## Features
- **Real-time Hand Tracking**: Uses MediaPipe to detect and track hand landmarks.
- **Finger Drawing**: Tracks the index fingertip to draw on the canvas.
- **Kalman Filter Integration**: Smoothens the drawing path to reduce noise.
- **Customizable Pen**:
  - Toggle drawing mode on/off.
  - Change pen color (Red, Green, Blue).
  - Adjust pen thickness.
- **Canvas Controls**:
  - Clear the canvas.
- **User Feedback**:
  - On-screen messages for mode changes.
- **Easy-to-Use Interface**: Controlled through keyboard shortcuts.

---

## Installation
1. **Install Dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

2. **Run the Application:**
   ```bash
   python draw.py
   ```

---

## Usage

### **Keyboard Controls**:
- **`d`**: Toggle drawing mode ON/OFF.
- **`c`**: Clear the canvas.
- **`r`**: Change pen color to red.
- **`g`**: Change pen color to green.
- **`b`**: Change pen color to blue.
- **`+`**: Increase pen thickness.
- **`-`**: Decrease pen thickness.
- **`ESC`**: Exit the application.

---

## Implementation Details

### **Core Components**:
1. **Hand Tracking**:
   - MediaPipe's `Hands` module detects hand landmarks in real-time.
   - Tracks the position of the index finger tip (`INDEX_FINGER_TIP`).

2. **Kalman Filter**:
   - Filters the fingertip's position to reduce jitter and noise during drawing.

3. **Drawing Logic**:
   - Captures fingertip positions to draw lines or points on a virtual canvas.
   - Blends the canvas with the webcam feed for visualization.

4. **Keyboard Interaction**:
   - Handles user inputs for controlling the pen and canvas.

---

## Dependencies
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

---

## Future Improvements
- Add multi-hand support.
- Introduce gesture-based commands (e.g., "pinch" to erase).
- Save drawings as image files.
- Enhance UI with on-screen buttons and settings panel.

---

## Acknowledgments
- [MediaPipe](https://google.github.io/mediapipe/) for the robust hand tracking solution.
- [OpenCV](https://opencv.org/) for providing excellent tools for computer vision.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

