#!/bin/bash
# Quick setup with pre-compiled packages (2-3 minutes)

#!/bin/bash
# Quick setup with pre-compiled packages (2-3 minutes)

set -e  # Exit on error

echo "Quick Setup - Using Pre-Compiled Packages"
echo "=========================================="
echo ""

# Install pre-compiled llama-cpp-python
echo "[1/3] Installing llama-cpp-python (pre-compiled)..."
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
  --quiet

if [ $? -eq 0 ]; then
    echo "✓ llama-cpp-python installed"
else
    echo "✗ llama-cpp-python installation failed"
    exit 1
fi

echo ""

# Install vLLM (if not already there)
echo "[2/3] Installing vLLM..."
pip install vllm --quiet 2>&1 | grep -v "already satisfied" || echo "✓ vLLM already installed"

echo ""

# Install analysis tools
echo "[3/3] Installing analysis tools..."
pip install pandas numpy scipy scikit-learn matplotlib seaborn huggingface-hub --quiet 2>&1 | grep -v "already satisfied" || echo "✓ Analysis tools already installed"


echo ""
echo "✅ Setup Complete!"
echo "Ready to run inference!"

# Test installations
echo "Testing packages..."
python -c "from llama_cpp import Llama; print('  ✓ llama-cpp-python works')"
python -c "import vllm; print('  ✓ vLLM works')"
python -c "import pandas, numpy, matplotlib; print('  ✓ Analysis tools work')"

echo ""
echo "All systems ready! 🚀"
echo ""
