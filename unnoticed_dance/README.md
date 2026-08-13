# Unnoticed Dance

Motion regeneration through human & machine perception. Extract 3D poses from everyday movement videos, blend with AIST++ dance choreography, and render stylized motion output.

---

## Overview

This project implements an end-to-end pipeline for **motion-to-motion generation**, transforming everyday movement into choreographed dance through the lens of both human and machine perception.

### Pipeline Flow

```
Daily Movement Video
    ↓ [CoMotion]
SMPL 3D Poses [T, 144]
    ↓ [Feature Extraction]
Motion Features [velocity, rhythm, expressivity]
    ↓ [Motion-to-Motion Transformer]
Generated Dance Poses [T, 144]
    ↓ [Blender + SMPL]
3D Animation [GLB/MP4]
    ↓ [GLSL Shaders + FFmpeg]
Stylized Motion Video [15s MP4]
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/neekita/unnoticed-dance.git
cd unnoticed-dance
```

### 2. Install Dependencies
```bash
# Create conda environment
conda create -n comotion python=3.10
conda activate comotion

# Install dependencies
pip install -r requirements.txt

# Download CoMotion
bash setup/download_comotion.sh

# Download SMPL (manual step - register at https://smpl.is.tue.mpg.de/)
# Place SMPL_NEUTRAL.pkl at: data/smpl/SMPL_NEUTRAL.pkl

# Download AIST++ (optional)
bash setup/download_aist.sh
```

### 3. Quick Test
```bash
# Extract poses from a test video
python src/extract_poses.py \
  --input-video examples/sample_walk.mp4 \
  --output-dir output/test \
  --num-frames 300

# View results
python -c "import torch; data = torch.load('output/test/poses.pt'); print(data['poses'].shape)"
```

### 4. Run Web UI
```bash
# Static HTML
python -m http.server 8000
# Open http://localhost:8000/ui/index.html

# Or with Node.js backend
cd ui
npm install
npm start
# Open http://localhost:3000
```

---

## 📋 Project Structure

```
unnoticed-dance/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
│
├── setup/                              # Setup scripts
│   ├── download_comotion.sh            # Download CoMotion models
│   ├── download_aist.sh                # Download AIST++ dataset
│   └── install_blender.sh              # Install Blender Python API
│
├── src/                                # Core Python pipeline
│   ├── extract_poses.py                # CoMotion pose extraction
│   ├── blend_poses.py                  # SMPL pose blending
│   ├── features.py                     # Motion feature extraction
│   ├── model.py                        # Motion-to-motion transformer
│   ├── render.py                       # Blender rendering
│   └── stylize.py                      # GLSL shader + FFmpeg
│
├── blender/                            # Blender Python scripts
│   ├── render_smpl.py                  # SMPL animation renderer
│   ├── anim_utils.py                   # Animation utilities
│   └── materials.blend                 # Blender materials file
│
├── shaders/                            # GLSL shader files
│   ├── painterly.glsl                  # Van Gogh style
│   ├── wave.glsl                       # Wave distortion
│   └── grid.glsl                       # Grid overlay
│
├── ui/                                 # Web interface
│   ├── index.html                      # Retro UI (daveos.fun style)
│   ├── api.js                          # Node.js backend
│   ├── package.json                    # NPM dependencies
│   └── vercel.json                     # Vercel deployment config
│
├── examples/                           # Example files
│   ├── sample_walk.mp4                 # Test video
│   └── sample_config.yaml              # Configuration template
│
├── data/                               # Data directory (not in git)
│   ├── smpl/
│   │   └── SMPL_NEUTRAL.pkl            # (Download manually)
│   └── aistplusplus/                   # (Download with script)
│
├── output/                             # Results directory
│   ├── poses/                          # Extracted SMPL poses
│   ├── blended/                        # Blended poses
│   └── renders/                        # Rendered videos
│
├── notebooks/                          # Jupyter notebooks
│   ├── 01_pose_extraction.ipynb        # Tutorial: extract poses
│   ├── 02_feature_analysis.ipynb       # Analyze motion features
│   └── 03_training.ipynb               # Train motion-to-motion model
│
├── tests/                              # Unit tests
│   ├── test_extract.py
│   ├── test_blend.py
│   └── test_render.py
│
├── docs/                               # Documentation
│   ├── INSTALLATION.md                 # Detailed setup
│   ├── USAGE.md                        # How to use each component
│   ├── ARCHITECTURE.md                 # Technical design
│   ├── THESIS.md                       # Thesis framework
│   └── API.md                          # API documentation
│
└── config/                             # Configuration files
    ├── default.yaml                    # Default settings
    ├── gpu.yaml                        # GPU optimization
    └── demo.yaml                       # Demo mode (fast)
```

---

## Core Components

### 1. Pose Extraction (CoMotion)
Extract 3D SMPL poses from video:
```python
from src.extract_poses import extract_poses_from_video

poses = extract_poses_from_video(
    video_path='path/to/video.mp4',
    num_frames=300,
    output_dir='output/poses'
)
print(poses.shape)  # [300, 144]
```

### 2. Feature Extraction
Compute motion features from poses:
```python
from src.features import extract_motion_features

features = extract_motion_features(
    poses=poses,
    fps=30
)
# Returns: velocity, rhythm, expressivity, body_spread
```

### 3. Motion-to-Motion Model
Train or use pretrained transformer:
```python
from src.model import MotionTransformer

model = MotionTransformer.from_pretrained('checkpoints/motion_model.pt')
generated = model(daily_motion_features)  # [300, 144]
```

### 4. Blender Rendering
Render SMPL animation to video:
```bash
blender --background --python blender/render_smpl.py -- \
  --poses output/poses/poses.pt \
  --output output/renders/animation.mp4
```

### 5. Stylization with FFmpeg
Apply GLSL shaders and effects:
```bash
ffmpeg -i animation.mp4 \
  -vf "glsl=shaders/painterly.glsl" \
  -t 15 output/final_video.mp4
```

---

## Web UI

### Option 1: Static HTML (Vercel)
```bash
cp ui/index.html ui/index.html
vercel deploy --prod
```

### Option 2: Full-Stack (Node.js + Python)
```bash
cd ui
npm install
npm start
# Open http://localhost:3000
```

### Features
- Video input (YouTube or local)
- Pipeline configuration
- Real-time progress tracking
- Results visualization
- Terminal output logging
- Retro daveos.fun aesthetic

---

## Data & Models

### SMPL Model
- **Download:** https://smpl.is.tue.mpg.de/ (requires registration)
- **Format:** `SMPL_NEUTRAL.pkl` (v1.1.0)
- **Size:** ~25 MB
- **Place at:** `data/smpl/SMPL_NEUTRAL.pkl`

### AIST++ Dataset
- **Download:** https://google.github.io/aistplusplus_dataset
- **Motions:** 10 genres, 1000+ sequences
- **Format:** SMPL `.pkl` files
- **Size:** ~10 GB
```bash
bash setup/download_aist.sh
```

### Pretrained Models
- **Motion-to-Motion Transformer:** [Download link]
- **Video Diffusion (fallback):** Replicate API

---

## Motion-to-Motion Model (Option B)

Core thesis contribution: **Condition dance generation on everyday movement instead of music.**

### Architecture
```
Input: Daily motion features [T, 6]
  ↓
Transformer Encoder (6 layers, 256 dims)
  ├─ Self-attention across motion sequence
  ├─ Learned embeddings for motion types
  └─ Positional encoding
  ↓
Transformer Decoder (6 layers)
  ├─ Cross-attention with AIST++ database
  └─ Generates pose parameters
  ↓
Output: Generated dance [T, 144]
```

### Training
```bash
python src/model.py \
  --mode train \
  --data-dir data/aistplusplus \
  --epochs 100 \
  --batch-size 32 \
  --output-dir checkpoints/
```

### Inference
```python
from src.model import MotionTransformer

model = MotionTransformer.from_pretrained('checkpoints/motion_model.pt')
generated = model(daily_features)
torch.save({'poses': generated}, 'output/generated.pt')
```

---

## Documentation

- **[INSTALLATION.md](docs/INSTALLATION.md)** — Detailed setup for all platforms
- **[USAGE.md](docs/USAGE.md)** — Complete usage guide
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Technical design & algorithms
- **[THESIS.md](docs/THESIS.md)** — Thesis framework & research questions
- **[API.md](docs/API.md)** — Python API reference

---

## Notebooks

Interactive tutorials:
```bash
jupyter notebook notebooks/
```

1. **01_pose_extraction.ipynb** — Extract SMPL poses from video
2. **02_feature_analysis.ipynb** — Analyze motion features
3. **03_training.ipynb** — Train motion-to-motion model
4. **04_rendering.ipynb** — Render animations in Blender

---

## Example Workflow

### Full Pipeline Example
```bash
# 1. Download YouTube video
yt-dlp -f "best[ext=mp4]" "https://youtube.com/watch?v=..." -o examples/my_video.mp4

# 2. Extract poses
python src/extract_poses.py \
  --input-video examples/my_video.mp4 \
  --output-dir output/mytest \
  --num-frames 300

# 3. Extract features
python src/features.py \
  --poses output/mytest/poses.pt \
  --output output/mytest/features.pt

# 4. Generate dance (using pretrained model)
python src/model.py \
  --mode inference \
  --checkpoint checkpoints/motion_model.pt \
  --input output/mytest/features.pt \
  --output output/mytest/generated.pt

# 5. Blend with AIST++
python src/blend_poses.py \
  --daily output/mytest/generated.pt \
  --aist data/aistplusplus/motions/gHO_sBM_cAll_d27_mWA5_ch05.pkl \
  --ratio 0.3 \
  --output output/mytest/blended.pt

# 6. Render in Blender
blender --background --python blender/render_smpl.py -- \
  --poses output/mytest/blended.pt \
  --output output/mytest/animation.mp4

# 7. Stylize with FFmpeg
ffmpeg -i output/mytest/animation.mp4 \
  -vf "glsl=shaders/painterly.glsl" \
  -t 15 output/mytest/final.mp4

# Done! Final video at: output/mytest/final.mp4
```

---

## Performance

### Extraction (CoMotion)
- **CPU:** ~1 fps
- **GPU:** ~10-30 fps (depends on model)
- **Time for 300 frames:** 10-30 sec (GPU), 5 min (CPU)

### Feature Extraction
- **Time:** <1 sec for 300 frames

### Model Inference
- **Single frame:** ~10-50 ms
- **300 frames:** ~3-15 sec

### Rendering
- **Blender (Eevee):** ~1-2 min for 10s video
- **Blender (Cycles):** ~5-10 min for 10s video

### Stylization
- **FFmpeg GLSL:** ~30 sec for 10s video

**Total time:** 15-45 minutes for full pipeline (GPU recommended)

---

## Testing

Run tests:
```bash
pytest tests/
```

Individual test suites:
```bash
pytest tests/test_extract.py -v       # Test pose extraction
pytest tests/test_blend.py -v         # Test pose blending
pytest tests/test_render.py -v        # Test Blender rendering
```

---

## Contributing

Contributions welcome! Areas of interest:
- [ ] Improve pose extraction accuracy
- [ ] Optimize Blender rendering
- [ ] Add new GLSL shaders
- [ ] Expand documentation
- [ ] Bug fixes and optimizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License — Free for research and educational use.

---


