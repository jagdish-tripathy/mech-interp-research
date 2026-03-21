# check_available_saes.py

from huggingface_hub import list_repo_files
import re
from collections import defaultdict

print("="*70)
print("CHECKING AVAILABLE SAES IN GEMMA SCOPE 2")
print("="*70)

repo_id = "google/gemma-scope-2-12b-it"

# Get all files in the repo
print(f"\nFetching file list from {repo_id}...")
all_files = list_repo_files(repo_id)

print(f"Total files in repo: {len(all_files)}")

# Organize by SAE type
sae_types = defaultdict(lambda: defaultdict(list))

# Parse file structure
for file_path in all_files:
    # Look for pattern: {type}/layer_{layer}_width_{width}_l0_{l0}/...
    match = re.match(r'([^/]+)/layer_(\d+)_width_(\w+)_l0_(\w+)/', file_path)
    if match:
        sae_type, layer, width, l0 = match.groups()
        key = f"layer_{layer}_width_{width}_l0_{l0}"
        sae_types[sae_type][int(layer)].append((width, l0, key))

# Display results
print("\n" + "="*70)
print("AVAILABLE SAE TYPES AND LAYERS")
print("="*70)

for sae_type in sorted(sae_types.keys()):
    print(f"\n📁 {sae_type}/")
    print(f"   Layers available: {sorted(sae_types[sae_type].keys())}")
    
    # Show details for first layer as example
    first_layer = min(sae_types[sae_type].keys())
    print(f"   Example (layer {first_layer}):")
    
    for width, l0, key in sorted(set(sae_types[sae_type][first_layer])):
        print(f"      - width={width}, l0={l0}")

# Recommended configuration
print("\n" + "="*70)
print("RECOMMENDED FOR YOUR RESEARCH")
print("="*70)

if 'resid_post' in sae_types:
    resid_layers = sorted(sae_types['resid_post'].keys())
    print(f"\n✓ Use: resid_post")
    print(f"  Available layers: {resid_layers}")
    
    # Select strategic layers
    if len(resid_layers) >= 4:
        # Select: first, 25%, 50%, 75%, last
        strategic = [
            resid_layers[0],                           # First
            resid_layers[len(resid_layers)//4],        # 25%
            resid_layers[len(resid_layers)//2],        # 50%
            resid_layers[3*len(resid_layers)//4],      # 75%
            resid_layers[-1]                           # Last
        ]
        print(f"  Recommended strategic layers: {strategic}")
    else:
        print(f"  Use all layers: {resid_layers}")
    
    # Check widths and L0s
    example_layer = resid_layers[len(resid_layers)//2]  # Middle layer
    available_configs = sae_types['resid_post'][example_layer]
    
    print(f"\n  Available configurations (layer {example_layer} example):")
    widths = set()
    l0s = set()
    for width, l0, _ in available_configs:
        widths.add(width)
        l0s.add(l0)
    
    print(f"    Widths: {sorted(widths)}")
    print(f"    L0 values: {sorted(l0s)}")
    print(f"\n  💡 Recommendation: width=16k, l0=medium")
    print(f"     (Good balance of interpretability and coverage)")

else:
    print("❌ resid_post not found - check repo structure")

# Generate example loading code
print("\n" + "="*70)
print("EXAMPLE CODE TO LOAD SAES")
print("="*70)

if 'resid_post' in sae_types:
    resid_layers = sorted(sae_types['resid_post'].keys())
    example_layer = resid_layers[len(resid_layers)//2]
    
    print(f"""
from sae_lens import SAE

# Load SAE for layer {example_layer}
sae = SAE.from_pretrained(
    release="gemma-scope-2-12b-it-resid_post",
    sae_id="layer_{example_layer}_width_16k_l0_medium",
    device="cuda"
)

# Or for multiple layers:
layers_to_use = {strategic if len(resid_layers) >= 4 else resid_layers}
saes = {{}}

for layer in layers_to_use:
    saes[layer] = SAE.from_pretrained(
        release="gemma-scope-2-12b-it-resid_post",
        sae_id=f"layer_{{layer}}_width_16k_l0_medium",
        device="cuda"
    )
""")

print("="*70)