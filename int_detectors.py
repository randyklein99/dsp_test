# %%
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_80211ag_preamble
from detectors import variance_trajectory_detector, matched_filter_detector
import mplcursors  # For interactive cursor

# Enable interactive plotting (use in Jupyter Notebook)
# %matplotlib notebook  # Uncomment this line in Jupyter Notebook

# Define parameters
fs = 20e6  # Sampling frequency in Hz
add_rf_fingerprint = True
seed = 42

# Generate signal
t, input_signal = generate_80211ag_preamble(fs=fs, add_rf_fingerprint=add_rf_fingerprint, seed=seed)

# Extract STF for matched filter
stf = input_signal[: int(8e-6 * fs)]

# Add extended noise preamble (20 µs)
noise_duration = 20e-6
noise_samples = int(noise_duration * fs)
noise = (np.random.randn(noise_samples) + 1j * np.random.randn(noise_samples)) * 0.05
t_noise = np.arange(noise_samples) / fs
input_signal = np.concatenate([noise, input_signal])
t = np.concatenate([t_noise, t_noise[-1] + (1 / fs) + t])

# Plot the input signal
plt.figure()
plt.plot(t * 1e6, input_signal.real, label="Signal with Noise (Real)")
plt.plot(t[int(noise_samples):int(noise_samples + 16e-6 * fs)], 
         input_signal[int(noise_samples):int(noise_samples + 16e-6 * fs)].real, 
         label="True Preamble (16 µs)", linestyle="--")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("802.11 a/g Signal with 20 µs Noise and Full Preamble")
plt.show()
print("Signal length:", len(input_signal))

# Apply detectors
variance_traj_denoised, threshold_var, denoised_mag = variance_trajectory_detector(
    input_signal, t, threshold_multiplier=1.5
)
detected_mf, mf_output, threshold_mf = matched_filter_detector(input_signal, stf, fs=fs)

# Compute VT path (variance trajectory on raw signal magnitude)
window_size = 320  # Matches dissertation (16 µs at 20 MHz)
# Center the signal magnitude using the mean of the noise region
noise_mag = np.abs(input_signal[:noise_samples])
noise_mean = np.mean(noise_mag)
raw_mag = np.abs(input_signal) - noise_mean
raw_mag = np.maximum(raw_mag, 0)  # Ensure non-negative
variance_traj_raw = np.zeros(len(input_signal) - window_size + 1)
for i in range(len(variance_traj_raw)):
    window = raw_mag[i : i + window_size]
    variance_traj_raw[i] = np.var(window)

# Time axis for variance trajectory (shifted to end of window)
t_var = t[window_size - 1:len(variance_traj_raw) + window_size - 1]

# Normalize variance trajectory
max_variance = np.max(variance_traj_raw)  # Use full range
variance_traj_raw = variance_traj_raw / max_variance if max_variance > 0 else variance_traj_raw

# Debug: Print VT values and ensure correct noise region
noise_mask = (t_var * 1e6 >= 16.05) & (t_var * 1e6 < 19.95)  # Use t_var for accurate window end times
noise_var_raw = variance_traj_raw[noise_mask]
print(f"Raw VT Values in Noise Region (16.05-19.95 µs): {noise_var_raw}")
print(f"Number of Noise Region Samples: {len(noise_var_raw)}")
noise_median_raw = np.median(noise_var_raw) if noise_var_raw.size > 0 else 0.0
print(f"Raw Noise Region Median VT (16.05-19.95 µs): {noise_median_raw}")
# Use max VT in noise region for threshold to avoid early triggering
threshold_var_raw = 1.2 * np.max(noise_var_raw) if noise_var_raw.size > 0 else 0.005
print(f"Threshold (raw) based on 1.2 * max: {threshold_var_raw}")

# Find threshold trigger point for raw
start_idx_raw = 0
start_time_raw = None
start_value_raw = None
for i in range(len(variance_traj_raw)):
    if variance_traj_raw[i] > threshold_var_raw and t[i + window_size - 1] * 1e6 >= 20.0:
        start_idx_raw = i
        start_time_raw = t[i + window_size - 1] * 1e6
        start_value_raw = variance_traj_raw[i]
        break

# Normalize denoised variance trajectory
max_variance_denoised = np.max(variance_traj_denoised)
variance_traj_denoised = variance_traj_denoised / max_variance_denoised if max_variance_denoised > 0 else variance_traj_denoised

# Debug: Print VT values and ensure correct noise region
noise_var_denoised = variance_traj_denoised[noise_mask]
print(f"Denoised VT Values in Noise Region (16.05-19.95 µs): {noise_var_denoised}")
print(f"Number of Noise Region Samples: {len(noise_var_denoised)}")
noise_median_denoised = np.median(noise_var_denoised) if noise_var_denoised.size > 0 else 0.0
print(f"Denoised Noise Region Median VT (16.05-19.95 µs): {noise_median_denoised}")
# Use max VT in noise region for threshold to avoid early triggering
threshold_var = 1.2 * np.max(noise_var_denoised) if noise_var_denoised.size > 0 else 0.005
print(f"Threshold (denoised) based on 1.2 * max: {threshold_var}")

# Find threshold trigger point for denoised
start_idx_denoised = 0
start_time_denoised = None
start_value_denoised = None
for i in range(len(variance_traj_denoised)):
    if variance_traj_denoised[i] > threshold_var and t[i + window_size - 1] * 1e6 >= 20.0:
        start_idx_denoised = i
        start_time_denoised = t[i + window_size - 1] * 1e6
        start_value_denoised = variance_traj_denoised[i]
        break

# Create interactive figure
plt.figure(figsize=(12, 9))

# Subplot 1: Raw VT Path
plt.subplot(3, 1, 1)
line_raw, = plt.plot(t_var * 1e6, variance_traj_raw, label="Variance Trajectory (Raw)")
plt.axhline(threshold_var_raw, color="r", linestyle=":", label="Threshold")
# Add annotation for threshold trigger point
if start_time_raw is not None:
    plt.axvline(x=start_time_raw, color='g', linestyle='--', label='Threshold Trigger')
    plt.annotate(f'Trigger at {start_time_raw:.2f} µs\nVT={start_value_raw:.4f}',
                 xy=(start_time_raw, start_value_raw), xytext=(start_time_raw + 1, start_value_raw + 0.1),
                 arrowprops=dict(facecolor='green', shrink=0.05))
plt.xlabel("Time (µs)")
plt.ylabel("Variance")
plt.title("Variance Trajectory Detector (Raw Signal)")
plt.legend()
plt.xlim(0, 36)  # Full time range

# Subplot 2: Denoised VT Path
plt.subplot(3, 1, 2)
line_denoised, = plt.plot(t_var * 1e6, variance_traj_denoised, label="Variance Trajectory (Denoised)")
plt.axhline(threshold_var, color="r", linestyle=":", label="Threshold")
# Add annotation for threshold trigger point
if start_time_denoised is not None:
    plt.axvline(x=start_time_denoised, color='g', linestyle='--', label='Threshold Trigger')
    plt.annotate(f'Trigger at {start_time_denoised:.2f} µs\nVT={start_value_denoised:.4f}',
                 xy=(start_time_denoised, start_value_denoised), xytext=(start_time_denoised + 1, start_value_denoised + 0.1),
                 arrowprops=dict(facecolor='green', shrink=0.05))
plt.xlabel("Time (µs)")
plt.ylabel("Variance")
plt.title("Variance Trajectory Detector with Full DTCWT Denoising")
plt.legend()
plt.xlim(0, 36)  # Full time range

# Subplot 3: Matched Filter
plt.subplot(3, 1, 3)
plt.plot(t * 1e6, mf_output, label="Matched Filter Output")
plt.plot(t * 1e6, detected_mf * np.max(mf_output), label="Detected (MF)", linestyle="--")
plt.axhline(threshold_mf, color="r", linestyle=":", label="Threshold")
plt.xlabel("Time (µs)")
plt.ylabel("Normalized Output")
plt.title("Matched Filter Detector (Full STF)")
plt.legend()
plt.xlim(0, 36)

plt.tight_layout()

# Add interactive cursor to show variance values
cursor_raw = mplcursors.cursor(line_raw, hover=True)
cursor_raw.connect("add", lambda sel: sel.annotation.set_text(f'Variance: {variance_traj_raw[sel.target.index]:.4f} at {t_var[sel.target.index] * 1e6:.2f} µs'))
cursor_denoised = mplcursors.cursor(line_denoised, hover=True)
cursor_denoised.connect("add", lambda sel: sel.annotation.set_text(f'Variance: {variance_traj_denoised[sel.target.index]:.4f} at {t_var[sel.target.index] * 1e6:.2f} µs'))

plt.show()
# %%
