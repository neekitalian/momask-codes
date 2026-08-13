import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
CAMERA_INDICES = [1, 2]      # The two webcams. Change if needed (see notes below).
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30                     # Target FPS for the output files.
OUTPUT_DIR = "recordings"
BACKGROUND_COLOR = (0, 255, 0)  # Green background for the "body only" video (BGR).
SHOW_PREVIEW = True          # Set False to record without preview windows.

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class CameraRecorder(threading.Thread):
    """Handles one webcam: capture + MediaPipe pose + writing the 3 output files."""

    def __init__(self, cam_index, stop_event, session_dir):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.stop_event = stop_event
        self.session_dir = session_dir
        self.ok = False  # Whether the camera opened successfully.

    def run(self):
        # CAP_DSHOW is the most reliable backend for webcams on Windows.
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[Camera {self.cam_index}] Could not open. Skipping.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        # Read the size the camera actually gave us (may differ from requested).
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ok = True
        print(f"[Camera {self.cam_index}] Opened at {width}x{height}.")

        # Set up the three video writers (MP4).
        fourcc = cv2.VideoWriter_fourcc(*"mp4d") if False else cv2.VideoWriter_fourcc(*"mp4v")
        prefix = os.path.join(self.session_dir, f"cam{self.cam_index}")
        writer_raw = cv2.VideoWriter(f"{prefix}_raw.mp4", fourcc, FPS, (width, height))
        writer_pose = cv2.VideoWriter(f"{prefix}_pose.mp4", fourcc, FPS, (width, height))
        writer_body = cv2.VideoWriter(f"{prefix}_body.mp4", fourcc, FPS, (width, height))

        window_name = f"Camera {self.cam_index} (pose)"

        with mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=True,      # Needed to extract the body silhouette.
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while not self.stop_event.is_set():
                grabbed, frame = cap.read()
                if not grabbed:
                    print(f"[Camera {self.cam_index}] Lost frame.")
                    break

                # MediaPipe expects RGB.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)

                # --- Output 1: raw frame ---
                writer_raw.write(frame)

                # --- Output 2: pose skeleton overlay ---
                pose_frame = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        pose_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                writer_pose.write(pose_frame)

                # --- Output 3: body only (background removed) ---
                body_frame = frame.copy()
                if results.segmentation_mask is not None:
                    # mask values are 0..1; keep pixels above 0.5 as "person".
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
                    bg = np.zeros(frame.shape, dtype=np.uint8)
                    bg[:] = BACKGROUND_COLOR
                    body_frame = np.where(condition, frame, bg)
                writer_body.write(body_frame)

                if SHOW_PREVIEW:
                    cv2.imshow(window_name, pose_frame)
                    # Any 'q' press stops every camera.
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()
                        break

        cap.release()
        writer_raw.release()
        writer_pose.release()
        writer_body.release()
        if SHOW_PREVIEW:
            cv2.destroyWindow(window_name)
        print(f"[Camera {self.cam_index}] Stopped. Files saved with prefix '{prefix}_*.mp4'.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    print(f"Saving recordings to: {os.path.abspath(session_dir)}")

    stop_event = threading.Event()
    recorders = [CameraRecorder(idx, stop_event, session_dir) for idx in CAMERA_INDICES]

    for r in recorders:
        r.start()
        time.sleep(1.0)  # Stagger startup; helps when both cams share a USB hub.

    print("Recording... press 'q' in any window to stop (or Ctrl+C in this terminal).")
    try:
        while any(r.is_alive() for r in recorders):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    for r in recorders:
        r.join()

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()