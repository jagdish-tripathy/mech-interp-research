#!/bin/bash
# Quick setup with pre-compiled packages (2-3 minutes)

#!/bin/bash
# Quick setup with pre-compiled packages (2-3 minutes)

set -e  # Exit on error

echo "Quick Setup - Using Pre-Compiled Packages"
echo "=========================================="
echo ""

# ============================================================================
# [1/3] Install llama-cpp-python
# ============================================================================

echo "[1/3] Installing llama-cpp-python (pre-compiled)..."

# Check if already installed
if python -c "from llama_cpp import Llama" 2>/dev/null; then
    echo "✓ llama-cpp-python already installed"
else
    echo "   Downloading and installing (this may take 2-5 minutes)..."
    pip install llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
      --no-cache-dir \
      --quiet
    
    # Verify installation
    if python -c "from llama_cpp import Llama" 2>/dev/null; then
        echo "✓ llama-cpp-python installed successfully"
    else
        echo "✗ llama-cpp-python installation FAILED"
        echo "   Continuing anyway (you can try manual installation)"
    fi
fi

echo ""

# ============================================================================
# [2/3] Install vLLM (optional)
# ============================================================================

echo "[2/3] Installing vLLM (optional)..."

# Check if already installed
if python -c "import vllm" 2>/dev/null; then
    echo "✓ vLLM already installed"
else
    echo "   Downloading and installing (this may take 3-5 minutes)..."
    
    # Clear cache first to avoid disk space issues
    pip cache purge 2>/dev/null || true
    
    # Install without cache
    pip install vllm --no-cache-dir --quiet 2>&1 | grep -E "Successfully installed|ERROR" || true
    
    # Verify installation
    if python -c "import vllm" 2>/dev/null; then
        echo "✓ vLLM installed successfully"
    else
        echo "⚠️  vLLM installation failed (not critical - you can use llama-cpp-python)"
        echo "   This is often due to disk space or CUDA version issues"
        echo "   You can continue without vLLM"
    fi
fi

echo ""

# ============================================================================
# [3/3] Install analysis tools
# ============================================================================

echo "[3/3] Installing analysis and interpretability tools..."

# Check if core packages are installed
if python -c "import pandas, numpy, matplotlib, scipy, sae_lens" 2>/dev/null; then
    echo "✓ All tools already installed"
else
    echo "   Installing required packages..."
    
    # Install analysis tools
    echo "   - pandas, numpy, scipy, scikit-learn, matplotlib, seaborn..."
    pip install pandas numpy scipy scikit-learn matplotlib seaborn \
      --no-cache-dir \
      --quiet
    
    # Install HuggingFace tools
    echo "   - huggingface-hub, transformers, accelerate..."
    pip install huggingface-hub transformers accelerate \
      --no-cache-dir \
      --quiet
    
    # Install SAE lens for interpretability
    echo "   - sae-lens (SAE interpretability)..."
    pip install sae-lens \
      --no-cache-dir \
      --quiet
    
    # Verify installation
    echo ""
    echo "Verifying installation..."
    
    if python -c "import pandas, numpy, matplotlib, scipy" 2>/dev/null; then
        echo "✓ Analysis tools installed successfully"
    else
        echo "✗ Analysis tools installation FAILED"
    fi
    
    if python -c "import transformers, accelerate" 2>/dev/null; then
        echo "✓ Transformers installed successfully"
    else
        echo "✗ Transformers installation FAILED"
    fi
    
    if python -c "import sae_lens" 2>/dev/null; then
        echo "✓ SAE Lens installed successfully"
    else
        echo "✗ SAE Lens installation FAILED"
        echo "   Try: pip install sae-lens"
    fi
fi

# And just in case, since this helped previously: SAE layers were not loading otherwise
pip install --upgrade sae-lens

echo ""
echo "======================================================================"
echo "Setup Summary"
echo "======================================================================"
echo ""

# Test all packages and show status
echo "Package Status:"
echo "---------------"

# Test llama-cpp-python
if python -c "from llama_cpp import Llama" 2>/dev/null; then
    echo "✅ llama-cpp-python - Ready"
else
    echo "❌ llama-cpp-python - NOT installed"
fi

# Test vLLM
if python -c "import vllm" 2>/dev/null; then
    echo "✅ vLLM - Ready"
else
    echo "⚠️  vLLM - Not available (optional)"
fi

# Test analysis tools
if python -c "import pandas, numpy, matplotlib, scipy" 2>/dev/null; then
    echo "✅ Analysis tools - Ready"
else
    echo "❌ Analysis tools - Incomplete"
fi

echo ""
echo "======================================================================"

