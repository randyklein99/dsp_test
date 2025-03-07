#%% Cell 1: Generate signal and apply detectors
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import generate_80211_stf_signal
from detectors import dtcwt_detector, matched_filter_detector

# Generate signal
t, input_signal, signal = generate_80211_stf_signal()
stf = signal[int(10e-6 * 20e6):int(18e-6 * 20e6)]  # Extract STF for matched filter

plt.plot(t * 1e6, input_signal, label="Signal with Noise")
plt.plot(t * 1e6, signal, label="True STF (10 symbols)", linestyle="--")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("802.11-like Signal with 10 µs Noise and Full STF Preamble")
plt.show()
print("Main env NumPy version:", np.__version__)
print("Signal length:", len(input_signal))

# Apply detectors
detected_dtcwt, mag_dtcwt, threshold_dtcwt, t_dtcwt = dtcwt_detector(input_signal, t)
detected_mf, mf_output, threshold_mf = matched_filter_detector(input_signal, stf)

#%% Cell 2: Visualize detection results
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
plt.plot(t * 1e6, detected_mf * np.max(mf_output), label="Detected (MF)", linestyle="--")
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