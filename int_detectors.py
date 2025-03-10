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

# %%