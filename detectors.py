import numpy as np
from scipy import signal as sig
import dtcwt  # Full dtcwt module import

def variance_trajectory_detector(
    signal, t, window_size=320, threshold_multiplier=0.5, nlevels=4, debug_level=0
):
    """
    Detect bursts using a variance trajectory with DTCWT enhancement and denoising.

    Parameters:
    - signal: Input signal (numpy array, complex-valued)
    - t: Time vector (numpy array, in seconds)
    - window_size: Size of the sliding window (default 320 ~ 16 µs at 20 MHz)
    - threshold_multiplier: Multiplier for variance threshold (default 0.5)
    - nlevels: Number of DTCWT decomposition levels (default 4)
    - debug_level: Debugging level (0 = no debug, 1 = basic, 2 = detailed)

    Returns:
    - variance_traj: Variance trajectory over time
    - threshold_var: Variance threshold used for detection
    - denoised_mag: Magnitude of the denoised signal
    """
    if len(signal) == 0 or len(t) == 0:
        raise ValueError("Signal and time array must be non-empty")
    if len(signal) != len(t):
        raise ValueError("Signal and time array lengths must match")

    # Step 1: DT-CWT decomposition
    target_length = 2 ** int(np.ceil(np.log2(len(signal))))
    signal_padded = np.pad(signal, (0, target_length - len(signal)), mode="constant")
    
    transform = dtcwt.Transform1d()
    coeffs = transform.forward(signal_padded, nlevels=nlevels)
    
    # Step 2: Denoising (thresholding highpass coefficients)
    fs = 1 / (t[1] - t[0])  # Sampling frequency
    noise_samples = int(20e-6 * fs)  # 20 µs noise preamble
    noise_coeffs = [coeff[:noise_samples] for coeff in coeffs.highpasses]
    noise_std = np.mean([np.std(np.abs(coeff)) for coeff in noise_coeffs])
    threshold = noise_std * threshold_multiplier
    
    # Apply soft thresholding to all highpass levels
    denoised_coeffs = []
    for level_coeffs in coeffs.highpasses:
        mag = np.abs(level_coeffs)
        sign = np.sign(level_coeffs)
        mag_denoised = np.maximum(mag - threshold, 0)
        denoised_coeffs.append(mag_denoised * sign * (mag > threshold))
    
    # Step 3: Inverse DT-CWT to reconstruct denoised signal
    denoised_pyramid = dtcwt.Pyramid(lowpass=coeffs.lowpass, highpasses=tuple(denoised_coeffs))
    coeffs_denoised = transform.inverse(denoised_pyramid)
    denoised_signal = coeffs_denoised[:len(signal)]  # Truncate to original length
    # Center the denoised signal magnitude
    noise_denoised = np.abs(denoised_signal[:noise_samples])
    noise_mean_denoised = np.mean(noise_denoised)
    denoised_mag = np.abs(denoised_signal) - noise_mean_denoised
    denoised_mag = np.maximum(denoised_mag, 0)  # Ensure non-negative
    if debug_level >= 1:
        print(f"Denoised signal max magnitude: {np.max(denoised_mag)}")

    # Step 4: Compute variance trajectory on denoised signal
    variance_traj = np.zeros(len(signal) - window_size + 1)
    for i in range(len(variance_traj)):
        window = denoised_mag[i : i + window_size]
        variance_traj[i] = np.var(window)

    # Step 5: Threshold based on 1.2 * max of VT noise values, excluding initial artifact and 20 µs
    t_var = t[window_size - 1: len(variance_traj) + window_size - 1]  # Time at end of windows
    noise_mask = (t_var * 1e6 >= 16.05) & (t_var * 1e6 < 19.95)  # Exclude 20 µs
    noise_var = variance_traj[noise_mask] if np.any(noise_mask) else variance_traj[1:-1]  # Exclude first and last (20 µs)
    if debug_level >= 1:
        print(f"Denoised VT Values in Noise Region (16.05-19.95 µs): {noise_var}")
        print(f"Number of Noise Region Samples: {len(noise_var)}")
    if debug_level >= 2:
        for i, val in enumerate(noise_var):
            print(f"Denoised VT[{i}] at {t_var[noise_mask][i] * 1e6:.2f} µs: {val:.6f}")
    noise_median = np.median(noise_var) if noise_var.size > 0 else 0.0
    if debug_level >= 1:
        print(f"Noise Region Median VT (16.05-19.95 µs): {noise_median}")
    threshold_var = 1.2 * np.max(noise_var) if noise_var.size > 0 else 0.005  # 1.2 * max of noise VT values
    if debug_level >= 1:
        print(f"Threshold: {threshold_var}")
    if debug_level >= 1:
        print(f"Variance trajectory max: {np.max(variance_traj)}")

    return variance_traj, threshold_var, denoised_mag  # Return only needed values

def matched_filter_detector(signal, template, threshold_fraction=0.5, fs=20e6):
    """
    Detect bursts using a matched filter.

    Args:
        signal (np.ndarray): Input signal.
        template (np.ndarray): Template signal (e.g., STF preamble).
        threshold_fraction (float): Fraction of max output for detection threshold.
        fs (float): Sampling frequency (Hz).

    Returns:
        tuple: (detected, mf_output, threshold_mf)
    """
    mf_output = sig.correlate(signal, template, mode="same")
    mf_output = np.abs(mf_output) / np.max(np.abs(mf_output))
    max_mf = np.max(mf_output)
    threshold_mf = threshold_fraction * max_mf
    detected = mf_output > threshold_mf
    peak_idx = np.argmax(mf_output)
    template_duration = len(template) / fs
    start_idx = max(0, peak_idx - int(template_duration * fs / 2))
    end_idx = min(len(signal), peak_idx + int(template_duration * fs / 2))
    detected = np.zeros_like(mf_output, dtype=bool)
    detected[start_idx:end_idx] = True
    return detected, mf_output, threshold_mf