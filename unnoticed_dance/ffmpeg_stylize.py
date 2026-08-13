"""
FFmpeg Stylization Pipeline
============================

Apply GLSL shaders and artistic effects to motion videos using FFmpeg.
Supports: painterly (Van Gogh), wave distortion, grid overlay, etc.

Author: Neekita Lian
Lab: Embodied Media Lab, KMD Keio
"""

import subprocess
import sys
import argparse
from pathlib import Path
import json

class FFmpegStyler:
    """Wrapper for FFmpeg video stylization."""
    
    def __init__(self, ffmpeg_path='ffmpeg'):
        self.ffmpeg = ffmpeg_path
        self.verify_ffmpeg()
    
    def verify_ffmpeg(self):
        """Check FFmpeg installation and GLSL support."""
        try:
            result = subprocess.run(
                [self.ffmpeg, '-filters'],
                capture_output=True,
                text=True
            )
            if 'glsl' not in result.stdout:
                print("⚠️  WARNING: FFmpeg GLSL filter not available")
                print("   Install: apt-get install ffmpeg (with libglsl)")
                self.has_glsl = False
            else:
                self.has_glsl = True
        except FileNotFoundError:
            print(f"❌ ERROR: FFmpeg not found at '{self.ffmpeg}'")
            sys.exit(1)
    
    def apply_painterly(self, input_video, output_video, strength=1.0, duration=15):
        """Apply Van Gogh style painterly effect."""
        print(f"[INFO] Applying painterly effect (strength={strength})...")
        
        # Use FFmpeg's edge and color quantization filters
        # (GLSL version requires custom shader)
        
        filters = [
            f"scale=1280:720",  # Scale to standard resolution
            f"fps=30",  # Standard 30fps
            f"edgedetect=sobel",  # Edge detection
            f"curves=master=0,0 1,1",  # Adjust contrast
            f"hue=s={strength}",  # Saturation adjustment
        ]
        
        cmd = [
            self.ffmpeg,
            '-i', input_video,
            '-vf', ','.join(filters),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_video,
            '-y'
        ]
        
        return self.run_ffmpeg(cmd)
    
    def apply_wave(self, input_video, output_video, amplitude=0.05, frequency=3, duration=15):
        """Apply wave distortion effect."""
        print(f"[INFO] Applying wave effect (amp={amplitude}, freq={frequency})...")
        
        filters = [
            f"scale=1280:720",
            f"fps=30",
            f"fftdim=9:16",  # Apply some wave via frequency domain
        ]
        
        cmd = [
            self.ffmpeg,
            '-i', input_video,
            '-vf', ','.join(filters),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_video,
            '-y'
        ]
        
        return self.run_ffmpeg(cmd)
    
    def apply_grid(self, input_video, output_video, grid_size=20, line_width=2, duration=15):
        """Apply grid overlay effect."""
        print(f"[INFO] Applying grid overlay (size={grid_size}px)...")
        
        filters = [
            f"scale=1280:720",
            f"fps=30",
            f"drawgrid=w={grid_size}:h={grid_size}:thickness={line_width}:color=black@0.3",
        ]
        
        cmd = [
            self.ffmpeg,
            '-i', input_video,
            '-vf', ','.join(filters),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_video,
            '-y'
        ]
        
        return self.run_ffmpeg(cmd)
    
    def apply_glsl_shader(self, input_video, output_video, shader_file, duration=15):
        """Apply custom GLSL shader (requires FFmpeg GLSL support)."""
        if not self.has_glsl:
            print("❌ ERROR: GLSL filter not available")
            return False
        
        print(f"[INFO] Applying GLSL shader: {shader_file}")
        
        # Read shader file
        try:
            with open(shader_file, 'r') as f:
                shader_code = f.read()
        except FileNotFoundError:
            print(f"❌ ERROR: Shader file not found: {shader_file}")
            return False
        
        filters = [
            f"scale=1280:720",
            f"fps=30",
            f"glsl='{shader_file}'",
        ]
        
        cmd = [
            self.ffmpeg,
            '-i', input_video,
            '-vf', ','.join(filters),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_video,
            '-y'
        ]
        
        return self.run_ffmpeg(cmd)
    
    def apply_color_quantize(self, input_video, output_video, levels=16, duration=15):
        """Apply color quantization for posterized effect."""
        print(f"[INFO] Applying color quantization ({levels} levels)...")
        
        filters = [
            f"scale=1280:720",
            f"fps=30",
            f"eq=contrast=1.2",  # Increase contrast
            f"quantize={levels}",  # Quantize colors
        ]
        
        cmd = [
            self.ffmpeg,
            '-i', input_video,
            '-vf', ','.join(filters),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_video,
            '-y'
        ]
        
        return self.run_ffmpeg(cmd)
    
    def apply_dither(self, input_video, output_video, pattern='bayer', duration=15):
        """Apply dithering for retro halftone effect."""
        print(f"[INFO] Applying dithering ({pattern} pattern)...")
        
        # Floyd-Steinberg dithering simulation
        filters = [
            f"scale=1280:720",
            f"fps=30",
            f"format=pix_fmts=rgb24",
            f"geq=r='if(abs(r(x\\,y)-g(x\\,y))>50|abs(g(x\\,y)-b(x\\,y))>50, 255, 0)':g='r':b='r'",
        ]
        
        cmd = [
            self.ffmpeg,
            '-i', input_video,
            '-vf', ','.join(filters),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_video,
            '-y'
        ]
        
        return self.run_ffmpeg(cmd)
    
    def apply_composite(self, video_a, video_b, output_video, mode='side_by_side', duration=15):
        """Composite two videos (side-by-side or overlay)."""
        print(f"[INFO] Compositing videos ({mode})...")
        
        if mode == 'side_by_side':
            filter_complex = "[0:v]scale=640:360[v0];[1:v]scale=640:360[v1];[v0][v1]hstack[out]"
        elif mode == 'overlay':
            filter_complex = "[0:v][1:v]overlay=0:0[out]"
        else:
            print(f"❌ ERROR: Unknown mode: {mode}")
            return False
        
        cmd = [
            self.ffmpeg,
            '-i', video_a,
            '-i', video_b,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_video,
            '-y'
        ]
        
        return self.run_ffmpeg(cmd)
    
    def run_ffmpeg(self, cmd):
        """Execute FFmpeg command."""
        try:
            print(f"[CMD] {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode == 0:
                print("[INFO] FFmpeg operation successful")
                return True
            else:
                print(f"[ERROR] FFmpeg failed with code {result.returncode}")
                return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

def main():
    """Main stylization pipeline."""
    parser = argparse.ArgumentParser(description='Stylize motion videos with FFmpeg')
    parser.add_argument('--input', type=str, required=True, help='Input video file')
    parser.add_argument('--output', type=str, required=True, help='Output video file')
    parser.add_argument('--style', type=str, default='painterly',
                       choices=['painterly', 'wave', 'grid', 'quantize', 'dither', 'custom'])
    parser.add_argument('--shader', type=str, help='GLSL shader file (for custom style)')
    parser.add_argument('--duration', type=float, default=15, help='Output duration (seconds)')
    parser.add_argument('--ffmpeg-path', type=str, default='ffmpeg', help='Path to FFmpeg binary')
    parser.add_argument('--strength', type=float, default=1.0, help='Effect strength')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Unnoticed Dance - FFmpeg Stylization")
    print("=" * 70)
    
    styler = FFmpegStyler(ffmpeg_path=args.ffmpeg_path)
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Apply effect
    success = False
    if args.style == 'painterly':
        success = styler.apply_painterly(args.input, args.output, args.strength, args.duration)
    elif args.style == 'wave':
        success = styler.apply_wave(args.input, args.output, duration=args.duration)
    elif args.style == 'grid':
        success = styler.apply_grid(args.input, args.output, duration=args.duration)
    elif args.style == 'quantize':
        success = styler.apply_color_quantize(args.input, args.output, duration=args.duration)
    elif args.style == 'dither':
        success = styler.apply_dither(args.input, args.output, duration=args.duration)
    elif args.style == 'custom':
        if not args.shader:
            print("❌ ERROR: --shader required for custom style")
            sys.exit(1)
        success = styler.apply_glsl_shader(args.input, args.output, args.shader, args.duration)
    
    if success:
        print("=" * 70)
        print("✓ Stylization complete!")
        print(f"  Output: {args.output}")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("❌ Stylization failed")
        print("=" * 70)
        sys.exit(1)

if __name__ == '__main__':
    main()
