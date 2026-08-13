"""record_and_sweep.py — Interactive dual-camera capture → verb_sweep pipeline.

Asks the user to:
  1. See all detected cameras with live previews
  2. Pick TWO cameras to record with simultaneously
  3. Pick WHICH ONE of the two to use for the verb_sweep (MediaPipe extraction)
  4. Enter the motion verb (one of the 8 spectrum verbs)
  5. Enter an experiment ID

Records both cameras simultaneously until the user presses 'q', saves videos as:
  input_videos/<verb>_<experiment_id>_cam<A>.mp4   ← sweep camera
  recordings/<verb>_<experiment_id>_cam<B>.mp4     ← second camera (archived)

Then immediately runs verb_sweep.py on the sweep camera's video.

Usage:
    python record_and_sweep.py
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2

VERBS           = ['walk', 'dance', 'run', 'jump', 'spin', 'kick', 'wave', 'stand']
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
FPS             = 30
RECORD_SECONDS  = 10
INPUT_DIR       = Path('input_videos')
ARCHIVE_DIR     = Path('recordings')
SECONDARY_VERB  = 'dance'


# ---------------------------------------------------------------------------
# Camera detection
# ---------------------------------------------------------------------------

def detect_cameras(max_index: int = 10) -> list[dict]:
    """Return info dicts for all cameras that open successfully."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            found.append({'index': i, 'w': w, 'h': h})
    return found


def print_camera_table(cameras: list[dict]) -> None:
    print("\n┌─────┬──────────────┬────────────┐")
    print("│  #  │  Camera idx  │ Resolution │")
    print("├─────┼──────────────┼────────────┤")
    for n, cam in enumerate(cameras):
        print(f"│  {n}  │  index {cam['index']:>4}  │ {cam['w']}x{cam['h']} │")
    print("└─────┴──────────────┴────────────┘")


def pick_two_cameras(cameras: list[dict]) -> tuple[dict, dict]:
    """Ask user to pick two camera slots and which one is for sweep."""
    n = len(cameras)
    while True:
        raw = input(f"\nEnter two camera numbers to record with (e.g. 0 1): ").strip().split()
        if len(raw) == 2 and all(r.isdigit() for r in raw):
            a, b = int(raw[0]), int(raw[1])
            if 0 <= a < n and 0 <= b < n and a != b:
                break
        print(f"  Enter two different numbers between 0 and {n-1}.")

    cam_a, cam_b = cameras[a], cameras[b]
    print(f"\n  Camera {a} (index {cam_a['index']})  ←→  Camera {b} (index {cam_b['index']})")

    while True:
        raw = input(f"  Which camera to use for verb_sweep? [{a}/{b}]: ").strip()
        if raw == str(a):
            return cam_a, cam_b   # sweep cam, archive cam
        if raw == str(b):
            return cam_b, cam_a
        print(f"  Enter {a} or {b}.")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def pick_verb() -> str:
    print(f"\nVerbs: {', '.join(VERBS)}")
    while True:
        v = input("Motion verb: ").strip().lower()
        if v in VERBS:
            return v
        print(f"  Not a valid verb. Choose from: {VERBS}")


def pick_experiment_id() -> str:
    while True:
        eid = input("Experiment ID (letters/numbers/underscores): ").strip()
        if eid and all(c.isalnum() or c == '_' for c in eid):
            return eid
        print("  Use only letters, numbers, and underscores.")


# ---------------------------------------------------------------------------
# Per-camera recording thread
# ---------------------------------------------------------------------------

class CameraRecorder(threading.Thread):
    def __init__(self, cam_index: int, output_path: Path, stop_event: threading.Event,
                 label: str = ''):
        super().__init__(daemon=True)
        self.cam_index   = cam_index
        self.output_path = output_path
        self.stop_event  = stop_event
        self.label       = label
        self.frame_count = 0
        self.error: str | None = None

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error = f"Could not open camera {self.cam_index}."
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Use the FPS the camera actually reports after configuration
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = FPS
        print(f"[{self.label}] Opened at {w}x{h} @ {actual_fps:.1f}fps")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            actual_fps, (w, h)
        )

        # Warm up: read and discard a few frames so the camera auto-exposure settles
        # and any residual terminal keypresses are flushed before we start listening
        for _ in range(10):
            cap.read()

        t_start = time.time()
        win = f"{self.label} (q to stop)"

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print(f"[{self.label}] Lost frame.")
                break

            writer.write(frame)
            self.frame_count += 1

            elapsed = time.time() - t_start
            overlay = frame.copy()
            cv2.putText(overlay,
                        f"REC {self.label}  {elapsed:.1f}s  {self.frame_count}f  (q=stop)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(win, overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        cap.release()
        writer.release()
        cv2.destroyWindow(win)
        duration = time.time() - t_start
        print(f"[{self.label}] Stopped — {self.frame_count} frames in {duration:.1f}s → {self.output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Scanning for cameras ...")
    cameras = detect_cameras()
    if len(cameras) < 2:
        print(f"ERROR: Found {len(cameras)} camera(s) — need at least 2.")
        sys.exit(1)

    print_camera_table(cameras)
    sweep_cam, archive_cam = pick_two_cameras(cameras)
    verb  = pick_verb()
    exp_id = pick_experiment_id()

    stem         = f"{verb}_{exp_id}"
    sweep_path   = INPUT_DIR  / f"{stem}_cam{sweep_cam['index']}.mp4"
    archive_path = ARCHIVE_DIR / f"{stem}_cam{archive_cam['index']}.mp4"

    for p in (sweep_path, archive_path):
        if p.exists():
            ans = input(f"\n{p} already exists. Overwrite? [y/N]: ").strip().lower()
            if ans != 'y':
                print("Aborted.")
                sys.exit(0)

    print(f"\n  Sweep  camera → {sweep_path}")
    print(f"  Archive camera → {archive_path}")
    print("\nBoth cameras recording. Press 'q' in either preview window to stop.\n")

    stop_event = threading.Event()
    recorders = [
        CameraRecorder(sweep_cam['index'],   sweep_path,   stop_event, label=f"cam{sweep_cam['index']} [SWEEP]"),
        CameraRecorder(archive_cam['index'], archive_path, stop_event, label=f"cam{archive_cam['index']} [archive]"),
    ]

    for r in recorders:
        r.start()
        time.sleep(0.5)   # stagger startup to avoid USB contention

    try:
        while any(r.is_alive() for r in recorders):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping ...")
        stop_event.set()

    for r in recorders:
        r.join()
    cv2.destroyAllWindows()

    for r in recorders:
        if r.error:
            print(f"ERROR: {r.error}")
            sys.exit(1)

    if not sweep_path.exists() or sweep_path.stat().st_size == 0:
        print("ERROR: Sweep camera recording failed or produced an empty file.")
        sys.exit(1)

    # --- Launch verb_sweep (skip if source and secondary verb are the same) ---
    if verb == SECONDARY_VERB:
        print(f"\nSource verb and secondary verb are both '{verb}' — skipping sweep.")
        print(f"Recording saved to: {sweep_path}")
    else:
        print(f"\n{'='*60}")
        print(f"Launching verb_sweep: {verb} → {SECONDARY_VERB}")
        print(f"Video: {sweep_path}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, '-u', 'scripts/verb_sweep.py',
            '--videos',                   str(sweep_path),
            '--secondary_verb',           SECONDARY_VERB,
            '--num_quantiles',            '10',
            '--quantile_prompt_strategy', '1',
            '--source_verb_mode',         'fixed',
            '--videos_per_section',       '1',
            '--gpu_id',                   '0',
        ]

        print('$', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
