#!/bin/bash
# RunPod Setup Script for Mortgage Approval Inference
# Run this on your RunPod instance after launching

set -e

echo "======================================================================"
echo "RunPod Environment Setup for Mortgage Approval Inference"
echo "======================================================================"
echo ""

# Prompt for HuggingFace token
echo "HuggingFace Token Setup"
echo "-----------------------"
echo "You need a HuggingFace token to download gated models (Gemma, LLaMA)."
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN

if [ ! -z "$HF_TOKEN" ]; then
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc
    export HF_TOKEN="$HF_TOKEN"
    echo "✓ HuggingFace token saved to ~/.bashrc"
    echo ""
else
    echo "⚠ Skipping token setup. Set manually with: export HF_TOKEN='your_token'"
    echo ""
fi

# Update and install system dependencies
echo "[1/6] Updating system packages..."
apt-get update -qq
apt-get install -y git wget curl

# Install Python dependencies
echo "[2/6] Installing Python packages..."
pip install --upgrade pip

# Install vLLM (for fast inference)
echo "[3/6] Installing vLLM..."
pip install vllm==0.6.5 --no-cache-dir

# Install llama.cpp Python bindings for grammar-constrained outputs (RECOMMENDED)
echo "[4/6] Installing llama-cpp-python with CUDA support..."
# Following TAKES A LONG TIME.
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
# Instead
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
  --quiet

# Install other dependencies
echo "[5/6] Installing additional packages..."
pip install pandas numpy scipy scikit-learn matplotlib seaborn huggingface-hub

echo "[6/6] Setup complete!"

echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "What's installed:"
echo "  ✓ vLLM (fast inference, ~95-98% valid outputs)"
echo "  ✓ llama-cpp-python (grammar constraints, 100% valid outputs)"
echo "  ✓ Analysis tools (pandas, matplotlib, etc.)"
echo ""
echo "Next steps:"
echo ""
echo "1. Download GGUF models (for llama.cpp with grammar):"
echo "   See GGUF_CONVERSION_GUIDE.md for details"
echo ""
echo "2. Trial run (200 samples) with vLLM:"
echo "   python runpod_inference.py --csv mortgage_bias_dataset.csv --mode ab --samples 200"
echo ""
echo "3. Trial run (200 samples) with llama.cpp + grammar (RECOMMENDED):"
echo "   python runpod_inference_llama.py --csv mortgage_bias_dataset.csv --model /path/to/model.gguf --mode ab --samples 200"
echo ""
echo "4. Full dataset (3000 samples):"
echo "   python runpod_inference_llama.py --csv mortgage_bias_dataset.csv --model /path/to/model.gguf --mode ab --samples 3000"
echo ""
echo "5. Analyze results:"
echo "   python analyze_results.py --results results/*.csv --mode ab"
echo ""
echo "See README.md for complete instructions!"
echo ""
