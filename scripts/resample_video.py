"""resample_video.py — Resample a video to 20fps using ffmpeg.

Usage:
    python scripts/resample_video.py input.mp4
    python scripts/resample_video.py input.mp4 --fps 20 --out output.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Resample a video to a target fps.')
    parser.add_argument('video', type=Path, help='Input video file.')
    parser.add_argument('--fps', type=int, default=20, help='Target fps. Default: 20.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output path. Default: <stem>_20fps.mp4 next to input.')
    args = parser.parse_args()

    if not args.video.exists():
        print(f'ERROR: {args.video} not found.')
        sys.exit(1)

    out = args.out or args.video.with_name(f'{args.video.stem}_{args.fps}fps.mp4')

    cmd = [
        'ffmpeg', '-y', '-i', str(args.video),
        '-vf', f'fps={args.fps}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-an',   # drop audio
        str(out),
    ]
    print('$', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
