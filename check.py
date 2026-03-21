import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "/workspace/outputs/steering_analysis/gemma-3-12b-it/raw_output"

# Load trajectories
between = np.load(f"{OUTPUT_DIR}/info_magnitude_between_group.npy")
within_w = np.load(f"{OUTPUT_DIR}/info_magnitude_within_white.npy")
within_b = np.load(f"{OUTPUT_DIR}/info_magnitude_within_black.npy")

# Compute ratios
ratio_w = between / np.maximum(within_w, 1e-6)  # Avoid division by zero
ratio_b = between / np.maximum(within_b, 1e-6)

# Find peaks
peak_between = np.argmax(between)
peak_ratio_w = np.argmax(ratio_w)

print("="*60)
print("FULL LAYER ANALYSIS")
print("="*60)

print(f"\nPeak between-group magnitude:")
print(f"  Layer {peak_between}: {between[peak_between]:.2f}")
print(f"  Within-White: {within_w[peak_between]:.2f}")
print(f"  Ratio: {ratio_w[peak_between]:.2f}x")

print(f"\nPeak ratio (Between/Within-White):")
print(f"  Layer {peak_ratio_w}: ratio = {ratio_w[peak_ratio_w]:.2f}x")
print(f"  Between: {between[peak_ratio_w]:.2f}")
print(f"  Within-White: {within_w[peak_ratio_w]:.2f}")

print("\n" + "="*60)
print("KEY LAYERS:")
print("="*60)

for layer in [0, 12, 24, 36, 46, 47, 48]:
    idx = layer if layer == 0 else layer + 1
    if idx < len(between):
        print(f"\nLayer {layer}:")
        print(f"  Between:      {between[idx]:8.2f}")
        print(f"  Within-White: {within_w[idx]:8.2f}")
        print(f"  Within-Black: {within_b[idx]:8.2f}")
        print(f"  Ratio (W):    {ratio_w[idx]:8.2f}x")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel 1: Magnitudes
axes[0].plot(between, label='Between-group (Black-White)', linewidth=2)
axes[0].plot(within_w, label='Within-White', linewidth=2, linestyle='--')
axes[0].plot(within_b, label='Within-Black', linewidth=2, linestyle='--')
axes[0].set_ylabel('Magnitude')
axes[0].set_xlabel('Layer')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Representational Distance Across Layers')

# Panel 2: Ratios
axes[1].plot(ratio_w, label='Between / Within-White', linewidth=2)
axes[1].axhline(y=2.0, color='orange', linestyle='--', label='Threshold (2x)')
axes[1].axhline(y=5.0, color='red', linestyle='--', label='Strong signal (5x)')
axes[1].set_ylabel('Ratio')
axes[1].set_xlabel('Layer')
axes[1].set_ylim([0, max(10, ratio_w.max() * 1.1)])
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Signal-to-Baseline Ratio')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/baseline_comparison_all_layers.png", dpi=150)
print(f"\nSaved plot: {OUTPUT_DIR}/baseline_comparison_all_layers.png")