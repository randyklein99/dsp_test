# %% Cell 3: Visualize feature extraction for combined subregion
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_80211ag_preamble
from dtcwt import Transform1d
from feature_extractor import (
    compute_characteristics,
    center_characteristics,
    compute_statistics,
    extract_features,
)

# Define parameters
fs = 20e6  # Sampling frequency in Hz
nlevels = 5  # Number of DTCWT decomposition levels
t_start = 0  # Start time of the subregion in signal (0 µs, since no noise)
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

# Perform DTCWT decomposition on the subregion directly
target_length = 2 ** int(np.ceil(np.log2(len(subregion))))
subregion_padded = np.pad(subregion, (0, target_length - len(subregion)), mode="constant")
transform = Transform1d()
coeffs = transform.forward(subregion_padded, nlevels=nlevels)
coeffs = [np.array(c) for c in coeffs.highpasses]  # Extract highpass coefficients

# Compute characteristics for multiple levels
chars = [compute_characteristics([coeffs[i : i + 1]], fs, 1)[0] for i in range(nlevels)]
centered = [center_characteristics([c])[0] for c in chars]

# Plot time series of centered characteristics for each level
for level in range(nlevels):
    char_level = centered[level]
    t_amplitude = np.linspace(0, 16, len(char_level["amplitude"]))
    t_phase = (
        np.linspace(0, 16, len(char_level["phase"]))
        if len(char_level["phase"]) > 0
        else np.array([])
    )
    t_frequency = (
        np.linspace(0, 16, len(char_level["frequency"]))
        if len(char_level["frequency"]) > 0
        else np.array([])
    )

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(
        t_amplitude,
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
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()

# Compute and print statistics for level 0 and level 3
for level in [0, 3]:
    stats = compute_statistics([centered[level]])[0]
    print(f"Statistics for Level {level}, Combined Subregion:")
    print(f"Amplitude: var={stats[0]:.4f}, skew={stats[1]:.4f}, kurt={stats[2]:.4f}")
    print(f"Phase: var={stats[3]:.4f}, skew={stats[4]:.4f}, kurt={stats[5]:.4f}")
    print(f"Frequency: var={stats[6]:.4f}, skew={stats[7]:.4f}, kurt={stats[8]:.4f}")

# Compute and print the complete 135-element feature vector
feature_vector = extract_features(
    input_signal, t, fs, preamble_start=0
)
print("\nComplete 135-Element Feature Vector:")
for i, value in enumerate(feature_vector):
    print(f"Feature {i+1:3d}: {value:.4f}")

# %%