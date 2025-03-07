#%% Cell 1: Generate 802.11-like signal with 10 µs noise and full STF preamble
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import signal as sig
from dtcwt_utils import run_dtcwt

fs = 20e6
noise_duration = 10e-6  # 10 µs noise-only
stf_duration = 20e-6    # 20 µs total (10 µs noise + 10 µs with STF)
total_duration = noise_duration + stf_duration
n_samples = int(fs * total_duration)
if n_samples % 2 != 0:
    n_samples += 1
t = np.linspace(0, total_duration, n_samples, endpoint=False)

# Full STF: 10 symbols, each 0.8 µs, starting at 10 µs
stf_symbol_duration = 0.8e-6
stf_freq = 1.25e6
stf_samples_per_symbol = int(fs * stf_symbol_duration)
stf_symbol = np.sin(2 * np.pi * stf_freq * np.linspace(0, stf_symbol_duration, stf_samples_per_symbol, endpoint=False))
stf = np.tile(stf_symbol, 10)  # 10 repeats, 160 samples
signal = np.zeros(n_samples)
start_idx = int(noise_duration * fs)  # STF starts at 10 µs
signal[start_idx:start_idx + len(stf)] = stf
input_signal = signal + 0.1 * np.random.randn(n_samples)

plt.plot(t * 1e6, input_signal, label="Signal with Noise")
plt.plot(t * 1e6, signal, label="True STF (10 symbols)", linestyle="--")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("802.11-like Signal with 10 µs Noise and Full STF Preamble")
plt.show()
print("Main env NumPy version:", np.__version__)
print("Signal length:", len(input_signal))

mag_dtcwt = run_dtcwt(input_signal, t)

# Matched filter with full STF preamble
mf_output = sig.correlate(input_signal, stf, mode='same')
mf_output = np.abs(mf_output) / np.max(np.abs(mf_output))

#%% Cell 2: Compare detectors with peak-based detection
t_dtcwt = t[::2]

# DTCWT threshold with noise estimate from first 10 µs
noise_mask = t_dtcwt * 1e6 < 10.0  # Noise region up to 10 µs
noise_mag = mag_dtcwt[noise_mask]
threshold_dtcwt = np.mean(noise_mag) + 3 * np.std(noise_mag)
detected_dtcwt = mag_dtcwt > threshold_dtcwt
print("DTCWT threshold:", threshold_dtcwt)

# Matched Filter peak-based detection
max_mf = np.max(mf_output)
threshold_mf = 0.5 * max_mf
detected_mf = mf_output > threshold_mf
peak_idx = np.argmax(mf_output)
start_idx = np.where(mf_output[:peak_idx] < threshold_mf)[0][-1] if np.any(mf_output[:peak_idx] < threshold_mf) else 0
end_idx = np.where(mf_output[peak_idx:] < threshold_mf)[0][0] + peak_idx if np.any(mf_output[peak_idx:] < threshold_mf) else -1
detected_mf = np.zeros_like(mf_output, dtype=bool)
if end_idx != -1:
    detected_mf[start_idx:end_idx] = True

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t_dtcwt * 1e6, mag_dtcwt, label="DTCWT Coefficients")
plt.plot(t_dtcwt * 1e6, detected_dtcwt * np.max(mag_dtcwt), label="Detected (DTCWT)", linestyle="--")
plt.axhline(threshold_dtcwt, color='r', linestyle=':', label="Threshold")
plt.xlabel("Time (µs)")
plt.ylabel("Magnitude")
plt.title("DTCWT Energy Detector")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t * 1e6, mf_output, label="Matched Filter Output")
plt.plot(t * 1e6, detected_mf * max_mf, label="Detected (MF)", linestyle="--")
plt.axhline(threshold_mf, color='r', linestyle=':', label="Threshold")
plt.xlabel("Time (µs)")
plt.ylabel("Normalized Output")
plt.title("Matched Filter Detector (Full STF)")
plt.legend()
plt.tight_layout()
plt.show()

for name, detected, time_array in [("DTCWT", detected_dtcwt, t_dtcwt), ("Matched Filter", detected_mf, t)]:
    indices = np.where(detected)[0]
    if len(indices) > 0:
        start, end = time_array[indices[0]] * 1e6, time_array[indices[-1]] * 1e6
        print(f"{name}: Detected burst from {start:.2f}µs to {end:.2f}µs")
    else:
        print(f"{name}: No burst detected")
# %%
