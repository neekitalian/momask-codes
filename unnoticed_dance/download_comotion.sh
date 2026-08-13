#!/bin/bash

# Download CoMotion Models
# ========================
# Downloads pretrained CoMotion models from Apple's ML repository

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  Unnoticed Dance - CoMotion Setup"
echo "=========================================="

# Create data directories
echo "[INFO] Creating directories..."
mkdir -p "$PROJECT_ROOT/data/comotion"
mkdir -p "$PROJECT_ROOT/data/smpl"

# Download CoMotion repo
echo "[INFO] Cloning CoMotion repository..."
cd "$PROJECT_ROOT"

if [ -d "ml-comotion" ]; then
    echo "[INFO] ml-comotion already exists, pulling latest..."
    cd ml-comotion
    git pull origin main
    cd ..
else
    echo "[INFO] Cloning ml-comotion..."
    git clone https://github.com/apple/ml-comotion.git
fi

# Install CoMotion
echo "[INFO] Installing CoMotion..."
cd ml-comotion
pip install -e '.[all]'
bash get_pretrained_models.sh
cd ..

echo "[INFO] CoMotion setup complete!"
echo ""
echo "[NEXT STEP] Download SMPL model:"
echo "  1. Register at https://smpl.is.tue.mpg.de/"
echo "  2. Download SMPL_NEUTRAL.pkl (v1.1.0)"
echo "  3. Place at: $PROJECT_ROOT/data/smpl/SMPL_NEUTRAL.pkl"
echo ""
echo "=========================================="
