# %% Cell 1: Generate signal and apply detectors
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_80211ag_preamble
from detectors import variance_trajectory_detector, matched_filter_detector

# Define parameters
fs = 20e6  # Sampling frequency in Hz (can be changed)
add_rf_fingerprint = True  # Option to enable RF fingerprint simulation
seed = 42  # Fixed seed for reproducibility of RF fingerprint (optional)

# Generate signal using the updated 802.11 a/g preamble generator
t, input_signal = generate_80211ag_preamble(
    fs=fs, add_rf_fingerprint=add_rf_fingerprint, seed=seed
)

# The preamble starts at t=0 in this signal, extract STF (first 8 µs) for matched filter
stf = input_signal[: int(8e-6 * fs)]  # First 8 µs, scales with fs

# Generate a signal with noise before the preamble for context
noise_duration = 10e-6  # 10 µs of noise, scales with fs
noise_samples = int(noise_duration * fs)
noise = (np.random.randn(noise_samples) + 1j * np.random.randn(noise_samples)) * 0.1
t_noise = np.arange(noise_samples) / fs
input_signal = np.concatenate([noise, input_signal])
t = np.concatenate([t_noise, t_noise[-1] + (1 / fs) + t])

plt.plot(t * 1e6, input_signal, label="Signal with Noise")
plt.plot(
    t[int(noise_samples) : int(noise_samples + 16e-6 * fs)],
    input_signal[int(noise_samples) : int(noise_samples + 16e-6 * fs)],
    label="True Preamble (16 µs)",
    linestyle="--",
)
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("802.11 a/g Signal with 10 µs Noise and Full Preamble")
plt.show()
print("Main env NumPy version:", np.__version__)
print("Signal length:", len(input_signal))

# Apply detectors
detected_var, variance_traj, threshold_var, denoised = variance_trajectory_detector(
    input_signal, t
)
detected_mf, mf_output, threshold_mf = matched_filter_detector(input_signal, stf, fs=fs)

# %% Cell 2: Visualize detection results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[: len(variance_traj)] * 1e6, variance_traj, label="Variance Trajectory")
plt.plot(
    t * 1e6,
    detected_var * np.max(variance_traj),
    label="Detected (Variance)",
    linestyle="--",
)
plt.axhline(threshold_var, color="r", linestyle=":", label="Threshold")
plt.xlabel("Time (µs)")
plt.ylabel("Variance")
plt.title("Variance Trajectory Detector with Full DTCWT Denoising")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t * 1e6, mf_output, label="Matched Filter Output")
plt.plot(
    t * 1e6, detected_mf * np.max(mf_output), label="Detected (MF)", linestyle="--"
)
plt.axhline(threshold_mf, color="r", linestyle=":", label="Threshold")
plt.xlabel("Time (µs)")
plt.ylabel("Normalized Output")
plt.title("Matched Filter Detector (Full STF)")
plt.legend()
plt.tight_layout()
plt.show()

for name, detected, time_array in [
    ("Variance Trajectory", detected_var, t),
    ("Matched Filter", detected_mf, t),
]:
    indices = np.where(detected)[0]
    if len(indices) > 0:
        start, end = time_array[indices[0]] * 1e6, time_array[indices[-1]] * 1e6
        print(f"{name}: Detected burst from {start:.2f}µs to {end:.2f}µs")
    else:
        print(f"{name}: No burst detected")

# %% Cell 3: Visualize feature extraction for combined subregion
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_80211ag_preamble
from dtcwt_utils import run_dtcwt
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
venv_path = (
    "/home/randy/code/test/.venv_dtcwt"  # Explicitly specify virtual environment path
)

# Generate the signal with optional RF fingerprint
t, input_signal = generate_80211ag_preamble(
    fs=fs, add_rf_fingerprint=add_rf_fingerprint, seed=seed
)

# Extract the combined subregion: 0 to 16 µs (16 µs duration)
idx_combined = (t >= t_start) & (t < t_end)
subregion = input_signal[idx_combined]
t_sub = t[idx_combined]

# Perform DTCWT decomposition on the subregion
coeffs = run_dtcwt(subregion, t_sub, nlevels=nlevels, venv_path=venv_path)

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
    input_signal, t, fs, preamble_start=0, venv_path=venv_path
)
print("\nComplete 135-Element Feature Vector:")
for i, value in enumerate(feature_vector):
    print(f"Feature {i+1:3d}: {value:.4f}")

# %%
