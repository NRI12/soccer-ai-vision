#!/bin/bash
set -e

echo "=== OSNet ReID - Install ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON_VERSION"

# Check CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch with CUDA already installed, skipping."
else
    echo "Installing PyTorch with CUDA 12.4..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install numpy Cython
pip3 install yacs gdown scipy Pillow opencv-python imageio soccernet h5py timm

# Build Cython extension
echo "Building Cython extension..."
RANK_CYLIB="$(dirname "$0")/osnet_reid/metrics/rank_cylib"
cd "$RANK_CYLIB" && python3 setup.py build_ext --inplace && rm -rf build && cd -

# Register package path
echo "Registering package..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$(dirname "$(realpath "$0")")" > "$SITE_PACKAGES/osnet_reid_dev.pth"
echo "Package registered at $SITE_PACKAGES/osnet_reid_dev.pth"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Download dataset:"
echo "       python3 tools/download_data.py --password <your_password>"
echo ""
echo "  2. Train:"
echo "       bash train.sh [config]        # default: configs/rtx3090_config.yaml"
echo ""
echo "  3. Test:"
echo "       bash test.sh <checkpoint.pth>"
