#!/bin/bash

# ============================================================================
# GEMMA-3-12B-IT + GEMMA-3-12B-PT +  SAE DOWNLOAD SCRIPT
# Directories: /workspace/models/ and /workspace/models/saes/
# ============================================================================

# Create base directories
mkdir -p /workspace/models
mkdir -p /workspace/models/saes

# ============================================================================
# GEMMA-3-12B - INSTRUCTION-TUNED (FP16 for SAE compatibility)
# ============================================================================
echo "[1/2] Gemma-3-12B-IT (Instruction-Tuned) - FP16 - ~24 GB"
echo "----------------------------------------------------------------------"

REPO_GEMMA3="google/gemma-3-12b-it"
MODEL_DIR="/workspace/models/gemma-3-12b-it"

if [ -d "$MODEL_DIR" ]; then
    echo "✓ $MODEL_DIR already exists (skipping)"
    du -sh "$MODEL_DIR"
else
    echo "Downloading Gemma-3-12B-IT (FP16) from Google..."
    
    # Download the full model (FP16 safetensors format)
    huggingface-cli download \
      "$REPO_GEMMA3" \
      --local-dir "$MODEL_DIR" \
      --token $HF_TOKEN \
      --exclude "*.gguf" "*.bin" "*pytorch_model*"
    
    if [ -d "$MODEL_DIR" ]; then
        echo "✓ Download successful!"
        du -sh "$MODEL_DIR"
        echo "Files:"
        ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null || ls -lh "$MODEL_DIR"
    else
        echo "❌ Download failed"
        exit 1
    fi
fi
echo ""

# ============================================================================
# GEMMA-3-12B - PRE-TRAINED (FP16 for SAE compatibility)
# ============================================================================
echo "[1/2] Gemma-3-12B-PT (Instruction-Tuned) - FP16 - ~24 GB"
echo "----------------------------------------------------------------------"

REPO_GEMMA3="google/gemma-3-12b-pt"
MODEL_DIR="/workspace/models/gemma-3-12b-pt"

if [ -d "$MODEL_DIR" ]; then
    echo "✓ $MODEL_DIR already exists (skipping)"
    du -sh "$MODEL_DIR"
else
    echo "Downloading Gemma-3-12B-PT (FP16) from Google..."
    
    # Download the full model (FP16 safetensors format)
    huggingface-cli download \
      "$REPO_GEMMA3" \
      --local-dir "$MODEL_DIR" \
      --token $HF_TOKEN \
      --exclude "*.gguf" "*.bin" "*pytorch_model*"
    
    if [ -d "$MODEL_DIR" ]; then
        echo "✓ Download successful!"
        du -sh "$MODEL_DIR"
        echo "Files:"
        ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null || ls -lh "$MODEL_DIR"
    else
        echo "❌ Download failed"
        exit 1
    fi
fi
echo ""

# ============================================================================
# GEMMA SCOPE 2 - SAE WEIGHTS (4 layers, 16k width)
# ============================================================================
echo "[2/2] Gemma Scope 2 SAEs - Layers 6, 12, 18, 24 (16k width) - ~800 MB"
echo "----------------------------------------------------------------------"

REPO_SAE="google/gemma-scope-2-12b-it"
SAE_DIR="/workspace/models/saes"

# Layers to download (early, mid, late, final)
LAYERS=(6 12 18 24)
WIDTH="16k"

echo "Downloading SAEs from $REPO_SAE to $SAE_DIR..."

for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "Downloading Layer $LAYER SAE (16k width)..."
    
    # Create layer-specific directory
    SAE_LAYER_DIR="$SAE_DIR/gemma-3-12b-it_layer_${LAYER}_width_${WIDTH}"
    
    if [ -d "$SAE_LAYER_DIR" ] && [ "$(ls -A $SAE_LAYER_DIR)" ]; then
        echo "✓ Layer $LAYER already exists (skipping)"
        du -sh "$SAE_LAYER_DIR"
    else
        mkdir -p "$SAE_LAYER_DIR"
        
        # Download SAE files for this layer
        huggingface-cli download \
          "$REPO_SAE" \
          --include "layer_${LAYER}/width_${WIDTH}/*" \
          --local-dir "$SAE_LAYER_DIR" \
          --token $HF_TOKEN
        
        if [ -d "$SAE_LAYER_DIR" ] && [ "$(ls -A $SAE_LAYER_DIR)" ]; then
            echo "✓ Layer $LAYER downloaded successfully!"
            du -sh "$SAE_LAYER_DIR"
        else
            echo "⚠ Layer $LAYER download incomplete - will auto-download when needed"
        fi
    fi
done

echo ""
echo "======================================================================"
echo "DOWNLOAD COMPLETE"
echo "======================================================================"
echo "Downloaded to:"
echo "  Model:  $MODEL_DIR"
echo "  SAEs:   $SAE_DIR"
echo ""
echo "Directory structure:"
tree -L 2 /workspace/models 2>/dev/null || ls -lh /workspace/models
echo ""
echo "Disk usage:"
du -sh "$MODEL_DIR" 2>/dev/null
du -sh "$SAE_DIR" 2>/dev/null
echo ""
echo "Note: If SAE download didn't work, don't worry!"
echo "SAELens will auto-download them when you run:"
echo "  sae = SAE.from_pretrained('gemma-scope-2-12b-it', 'layer_12/width_16k')"
echo ""
echo "Next steps:"
echo "  1. Test setup:          python test_sae_load.py"
echo "  2. Complete Week 1 reading (SAE papers)"
echo "  3. Begin experiments (Week 2)"
echo "======================================================================"
