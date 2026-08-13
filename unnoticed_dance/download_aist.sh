#!/bin/bash

# Download AIST++ Dataset
# =======================
# Downloads the AIST++ motion dataset (~10GB)
# Source: https://google.github.io/aistplusplus_dataset

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

echo "=========================================="
echo "  Unnoticed Dance - AIST++ Download"
echo "=========================================="
echo ""
echo "[IMPORTANT] This will download ~10GB of data"
echo "[IMPORTANT] Ensure you have ~20GB free disk space"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 1
fi

# Create directories
echo "[INFO] Creating data directory..."
mkdir -p "$DATA_DIR/aistplusplus"
cd "$DATA_DIR/aistplusplus"

# Clone AIST++ repo
if [ -d "aistplusplus_dataset" ]; then
    echo "[INFO] aistplusplus_dataset already exists"
    cd aistplusplus_dataset
else
    echo "[INFO] Cloning AIST++ dataset..."
    git clone https://github.com/google/aistplusplus_dataset.git
    cd aistplusplus_dataset
fi

# Download motions
echo "[INFO] Downloading motion data (this will take a while)..."
python downloader.py --download_folder=data/motions

echo ""
echo "=========================================="
echo "✓ AIST++ download complete!"
echo "  Location: $DATA_DIR/aistplusplus/aistplusplus_dataset/data/motions"
echo ""
echo "[INFO] Dataset includes:"
echo "  - 1000+ motion sequences"
echo "  - 10 dance genres (gBR, gPO, gLO, gMH, gWA, gKR, gJS, gJB, gMW, gHO)"
echo "  - SMPL parametrization"
echo "=========================================="
