#!/bin/bash
# Download instruction-tuned AND base models for all Gemma versions

cd /workspace/models

echo "======================================================================"
echo "DOWNLOADING GEMMA MODELS - INSTRUCTION-TUNED AND BASE VERSIONS"
echo "======================================================================"
echo ""
echo "Model naming:"
echo "  - IT (Instruction-Tuned): Follows instructions, understands tasks"
echo "  - BASE/PT (Pre-Trained): No instruction tuning, completion-style"
echo ""

# ============================================================================
# GEMMA-2-9B - INSTRUCTION-TUNED (IT) VERSION
# ============================================================================

echo "[1/4] Gemma-2-9B-IT (Instruction-Tuned) - Q4_K_M - 5 GB"
echo "----------------------------------------------------------------------"

FILE1="gemma-2-9b-it-Q4_K_M.gguf"
REPO1="bartowski/gemma-2-9b-it-GGUF"

if [ -f "$FILE1" ]; then
    echo "✓ $FILE1 already exists (skipping)"
    ls -lh "$FILE1"
else
    echo "Downloading $FILE1 from bartowski..."
    huggingface-cli download \
      "$REPO1" \
      "$FILE1" \
      --local-dir . \
      --token $HF_TOKEN
    
    if [ -f "$FILE1" ]; then
        echo "✓ Download successful!"
        ls -lh "$FILE1"
    else
        echo "❌ Download failed"
        exit 1
    fi
fi

echo ""

# ============================================================================
# GEMMA-2-9B - BASE VERSION (Non-Instruction-Tuned)
# ============================================================================

echo "[2/4] Gemma-2-9B-BASE (Pre-Trained / Non-Instruction) - Q4_K_M - 5 GB"
echo "----------------------------------------------------------------------"
echo "Note: Using nalf3in's quantization (bartowski doesn't provide base)"

FILE2="gemma-2-9b-q4_k_m.gguf"
REPO2="nalf3in/gemma-2-9b-Q4_K_M-GGUF"

if [ -f "$FILE2" ]; then
    echo "✓ $FILE2 already exists (skipping)"
    ls -lh "$FILE2"
else
    echo "Downloading $FILE2 from nalf3in..."
    huggingface-cli download \
      "$REPO2" \
      "$FILE2" \
      --local-dir . \
      --token $HF_TOKEN
    
    if [ -f "$FILE2" ]; then
        echo "✓ Download successful!"
        ls -lh "$FILE2"
    else
        echo "❌ Download failed"
        exit 1
    fi
fi

echo ""

# ============================================================================
# GEMMA-3-12B - INSTRUCTION-TUNED (IT) VERSION WITH QAT
# ============================================================================

echo "[3/4] Gemma-3-12B-IT (Instruction-Tuned) - QAT Q4_0 - 7 GB"
echo "----------------------------------------------------------------------"
echo "Note: Official Google release with QAT (Quantization-Aware Training)"

FILE3="gemma-3-12b-it-qat-q4_0.gguf"
REPO3="google/gemma-3-12b-it-qat-q4_0-gguf"

if [ -f "$FILE3" ]; then
    echo "✓ $FILE3 already exists (skipping)"
    ls -lh "$FILE3"
else
    echo "Downloading $FILE3 from Google..."
    huggingface-cli download \
      "$REPO3" \
      "$FILE3" \
      --local-dir . \
      --token $HF_TOKEN
    
    if [ -f "$FILE3" ]; then
        echo "✓ Download successful!"
        ls -lh "$FILE3"
    else
        echo "❌ Download failed"
        exit 1
    fi
fi

echo ""

# ============================================================================
# GEMMA-3-12B - BASE/PT VERSION (Pre-Trained) WITH QAT
# ============================================================================

echo "[4/4] Gemma-3-12B-PT (Pre-Trained / Base) - QAT Q4_0 - 8 GB"
echo "----------------------------------------------------------------------"
echo "Note: 'PT' = Pre-Trained = Base (Google's naming convention)"

FILE4="gemma-3-12b-pt-q4_0.gguf"
REPO4="google/gemma-3-12b-pt-qat-q4_0-gguf"

if [ -f "$FILE4" ]; then
    echo "✓ $FILE4 already exists (skipping)"
    ls -lh "$FILE4"
else
    echo "Downloading $FILE4 from Google..."
    huggingface-cli download \
      "$REPO4" \
      "$FILE4" \
      --local-dir . \
      --token $HF_TOKEN
    
    if [ -f "$FILE4" ]; then
        echo "✓ Download successful!"
        ls -lh "$FILE4"
    else
        echo "❌ Download failed"
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "DOWNLOAD SUMMARY"
echo "======================================================================"
echo ""

# Count files
IT_COUNT=$(ls -1 *-it-*.gguf 2>/dev/null | wc -l)
BASE_COUNT=$(ls -1 gemma-2-9b-Q4_K_M.gguf gemma-3-12b-pt-*.gguf 2>/dev/null | wc -l)

echo "Instruction-Tuned (IT) models: $IT_COUNT"
ls -lh *-it-*.gguf 2>/dev/null || echo "  None found"

echo ""
echo "Base (PT/Non-IT) models: $BASE_COUNT"
ls -lh gemma-2-9b-Q4_K_M.gguf gemma-3-12b-pt-*.gguf 2>/dev/null || echo "  None found"

echo ""
echo "All GGUF files:"
ls -lh *.gguf 2>/dev/null || echo "  None found"

echo ""
echo "Total storage used:"
du -sh . 2>/dev/null

echo ""
echo "======================================================================"
echo "MODEL COMPARISON GUIDE"
echo "======================================================================"
echo ""
echo "Gemma-2-9B:"
echo "  IT:   gemma-2-9b-it-Q4_K_M.gguf      (5 GB, bartowski)"
echo "  BASE: gemma-2-9b-Q4_K_M.gguf         (5 GB, nalf3in)"
echo ""
echo "Gemma-3-12B:"
echo "  IT:   gemma-3-12b-it-qat-q4_0.gguf   (7 GB, Google official)"
echo "  BASE: gemma-3-12b-pt-qat-q4_0.gguf   (7 GB, Google official)"
echo ""
echo "Quantization methods:"
echo "  - Q4_K_M: Standard 4-bit (bartowski/nalf3in)"
echo "  - QAT Q4_0: Quantization-Aware Training (Google)"
echo ""
echo "======================================================================"