import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

BRUSH_COLOR = (0, 255, 0)   # Green
BRUSH_THICKNESS = 6
ERASER_THICKNESS = 50       # Large eraser

INDEX_FINGER_TIP = 8

# ==========================================
# CHECK AI MODEL
# ==========================================

if not MODEL_PATH.is_file():
    print("ERROR: hand_landmarker.task was not found!")
    print("Expected location:", MODEL_PATH)
    raise SystemExit(1)

# ==========================================
# MEDIAPIPE HAND TRACKER
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

canvas = None
previous_point = None
timestamp_ms = 0
erase_mode = False

print("AI Hand Drawing started!")
print("D = Draw | E = Erase | C = Clear | Q = Quit")

# ==========================================
# MAIN LOOP
# ==========================================

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            if canvas is None:
                # Black pixels mean empty canvas
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Convert webcam image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # ==================================
            # TRACK INDEX FINGERTIP
            # ==================================

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                fingertip = hand[INDEX_FINGER_TIP]

                x = int(fingertip.x * width)
                y = int(fingertip.y * height)

                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                current_point = (x, y)

                # Show fingertip cursor
                if erase_mode:
                    cv2.circle(
                        frame,
                        current_point,
                        ERASER_THICKNESS // 2,
                        (0, 0, 255),
                        2
                    )
                else:
                    cv2.circle(
                        frame,
                        current_point,
                        8,
                        (0, 0, 255),
                        -1
                    )

                # Draw or erase between consecutive positions
                if previous_point is not None:

                    if erase_mode:
                        # TRUE ERASER:
                        # Remove pixels from the canvas.
                        cv2.line(
                            canvas,
                            previous_point,
                            current_point,
                            (0, 0, 0),
                            ERASER_THICKNESS,
                            cv2.LINE_8
                        )

                        cv2.circle(
                            canvas,
                            current_point,
                            ERASER_THICKNESS // 2,
                            (0, 0, 0),
                            -1
                        )

                    else:
                        # Draw green strokes
                        cv2.line(
                            canvas,
                            previous_point,
                            current_point,
                            BRUSH_COLOR,
                            BRUSH_THICKNESS,
                            cv2.LINE_AA
                        )

                previous_point = current_point

                cv2.putText(
                    frame,
                    f"Finger: ({x}, {y})",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Display all detected hand landmarks
                for landmark in hand:
                    lx = int(landmark.x * width)
                    ly = int(landmark.y * height)

                    lx = max(0, min(width - 1, lx))
                    ly = max(0, min(height - 1, ly))

                    cv2.circle(
                        frame,
                        (lx, ly),
                        3,
                        (255, 0, 0),
                        -1
                    )

            else:
                # Prevent a line joining two separate movements
                previous_point = None

            # ==================================
            # COMBINE CAMERA AND DRAWING
            # ==================================

            output = cv2.add(frame, canvas)

            mode_text = (
                "ERASER MODE (E)"
                if erase_mode
                else "DRAW MODE (D)"
            )

            cv2.putText(
                output,
                "AI HAND DRAWING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                "Move your index finger",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                mode_text,
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                output,
                "D: Draw | E: Erase | C: Clear | Q: Quit",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("AI Hand Drawing", output)

            # ==================================
            # KEYBOARD CONTROLS
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                canvas[:] = 0
                previous_point = None
                print("Canvas cleared.")

            elif key == ord("e"):
                erase_mode = True
                previous_point = None
                print("Eraser mode enabled.")

            elif key == ord("d"):
                erase_mode = False
                previous_point = None
                print("Drawing mode enabled.")

finally:
    cap.release()
    cv2.destroyAllWindows()

print("AI Hand Drawing stopped.")