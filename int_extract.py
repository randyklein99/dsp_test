# %% Cell 3: Visualize feature extraction for combined subregion and heatmap
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_80211ag_preamble
from feature_extractor import extract_features, compute_statistics

# Define debugging level (0 = no debug, 1 = basic, 2 = detailed)
debug_level = 0  # Set to basic debugging for this run

# Define parameters
fs = 20e6  # Sampling frequency in Hz
nlevels = 5  # Number of DTCWT decomposition levels
t_start = 0  # Start time of the subregion in signal (0 µs)
t_end = 16e-6  # End time of the subregion in signal (16 µs)
add_rf_fingerprint = True  # Option to enable RF fingerprint simulation
seed = 42  # Fixed seed for reproducibility of RF fingerprint

# Generate the signal with optional RF fingerprint
t, input_signal = generate_80211ag_preamble(
    fs=fs, add_rf_fingerprint=add_rf_fingerprint, seed=seed
)

# Extract the combined subregion: 0 to 16 µs (16 µs duration)
idx_combined = (t >= t_start) & (t < t_end)
subregion = input_signal[idx_combined]
t_sub = t[idx_combined]
if debug_level >= 1:
    print(f"Subregion length: {len(subregion)}, Expected samples: {int((t_end - t_start) * fs)}")
    print(f"Subregion t_sub range: min={t_sub[0]*1e6:.2f}µs, max={t_sub[-1]*1e6:.2f}µs")
    print(f"Subregion amplitude range: min={np.min(np.abs(subregion)):.4f}, max={np.max(np.abs(subregion)):.4f}")

# Extract features and get centered characteristics
feature_vector, centered_chars = extract_features(
    input_signal, t, fs, preamble_start=0, debug_level=debug_level
)

# Use the centered characteristics for the combined subregion (sub3)
centered = centered_chars['sub3']

# Plot time series of centered characteristics for each level
for level in range(nlevels):
    char_level = centered[level]
    # Adjust t_level to match original time steps, accounting for downsampling
    downsample_factor = 2 ** level
    t_original = np.linspace(0, 16, len(subregion))
    t_adjusted = t_original[::downsample_factor][:len(char_level["amplitude"])]
    t_phase = t_adjusted if len(char_level["phase"]) > 0 else np.array([])
    t_frequency = t_adjusted if len(char_level["frequency"]) > 0 else np.array([])

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(
        t_adjusted,
        char_level["amplitude"],
        label=f"Centered Amplitude (Level {level})",
    )
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.title(f"Centered Characteristics (Level {level}, Combined Subregion)")

    plt.subplot(3, 1, 2)
    if len(t_phase) > 0:
        plt.plot(t_phase, char_level["phase"], label="Centered Phase")
    else:
        plt.text(
            0.5,
            0.5,
            "No phase data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
    plt.xlabel("Time (µs)")
    plt.ylabel("Phase (radians)")

    plt.subplot(3, 1, 3)
    if len(t_frequency) > 0:
        plt.plot(t_frequency, char_level["frequency"], label="Centered Frequency")
    else:
        plt.text(
            0.5,
            0.5,
            "No frequency data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
    plt.xlabel("Time (µs)")
    plt.ylabel("Relative Frequency (unitless)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Compute and print statistics for level 0 and level 3
for level in [0, 3]:
    stats = compute_statistics([centered[level]], debug_level=debug_level)[0]
    if debug_level >= 1:
        print(f"Statistics for Level {level}, Combined Subregion:")
        print(f"Amplitude: var={stats[0]:.4f}, skew={stats[1]:.4f}, kurt={stats[2]:.4f}")
        print(f"Phase: var={stats[3]:.4f}, skew={stats[4]:.4f}, kurt={stats[5]:.4f}")
        print(f"Frequency: var={stats[6]:.4f}, skew={stats[7]:.4f}, kurt={stats[8]:.4f}")

# Print the complete 135-element feature vector
if debug_level >= 1:
    print("\nComplete 135-Element Feature Vector:")
    for i, value in enumerate(feature_vector):
        print(f"Feature {i+1:3d}: {value:.4f}")

# Add heatmap plotting with separate normalization
# For demonstration, use a single burst (extend later for multiple bursts/devices)
feature_matrix = feature_vector.reshape(15, 9)  # 15 segments x 9 stats
amp_stats = feature_matrix[:, :3]  # Variance, skew, kurtosis for amplitude (3 stats)
phase_stats = feature_matrix[:, 3:6]  # Variance, skew, kurtosis for phase (3 stats)
freq_stats = feature_matrix[:, 6:9]  # Variance, skew, kurtosis for frequency (3 stats)

# Apply separate min-max normalization for each feature type
amp_normalized = (amp_stats - np.min(amp_stats, axis=0)) / (np.max(amp_stats, axis=0) - np.min(amp_stats, axis=0) + 1e-10)  # Per-stat normalization
phase_normalized = (phase_stats - np.min(phase_stats, axis=0)) / (np.max(phase_stats, axis=0) - np.min(phase_stats, axis=0) + 1e-10)
freq_normalized = (freq_stats - np.min(freq_stats, axis=0)) / (np.max(freq_stats, axis=0) - np.min(freq_stats, axis=0) + 1e-10)

# Combine normalized stats into a single matrix for plotting
normalized_matrix = np.hstack((amp_normalized, phase_normalized, freq_normalized))

# Plot heatmap
plt.figure(figsize=(12, 6))
plt.imshow(normalized_matrix, cmap='jet', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Normalized Value')
plt.title('WD Fingerprint (Single Burst, Separate Normalization)')
plt.xlabel('Statistic (1-9: var, skew, kurt for amp, phase, freq)')
plt.ylabel('Segment (1-15)')
plt.xticks(np.arange(9), ['Var Amp', 'Skew Amp', 'Kurt Amp', 'Var Phase', 'Skew Phase', 'Kurt Phase', 'Var Freq', 'Skew Freq', 'Kurt Freq'])
plt.yticks(np.arange(15), np.arange(1, 16))
plt.show()

# %%