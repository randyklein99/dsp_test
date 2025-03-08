import numpy as np
from scipy import signal as sig
from dtcwt_utils import run_dtcwt


def dtcwt_detector(signal, t, threshold_multiplier=2.5):
    """
    Detect bursts using DTCWT energy (to be replaced).

    Args:
        signal (np.ndarray): Input signal.
        t (np.ndarray): Time array.
        threshold_multiplier (float): Multiplier for noise threshold (mean + k*std).

    Returns:
        tuple: (detected, mag_dtcwt, threshold_dtcwt, t_dtcwt)
    """
    mag_dtcwt = run_dtcwt(signal, t)
    t_dtcwt = t[::2]
    noise_mask = t_dtcwt * 1e6 < 10.0
    noise_mag = mag_dtcwt[noise_mask]
    threshold_dtcwt = np.mean(noise_mag) + threshold_multiplier * np.std(noise_mag)
    detected = mag_dtcwt > threshold_dtcwt
    return detected, mag_dtcwt, threshold_dtcwt, t_dtcwt


def variance_trajectory_detector(
    signal, t, window_size=320, threshold_multiplier=0.5, nlevels=4, venv_path=None
):
    """
    Detect bursts using a variance trajectory with DTCWT enhancement.

    Parameters:
    - signal: Input signal (numpy array)
    - t: Time vector (numpy array)
    - window_size: Size of the sliding window (320 ~ 16 µs at 20 MHz)
    - threshold_multiplier: Multiplier for variance threshold (0.5)
    - nlevels: Number of DTCWT decomposition levels
    - venv_path: Path to virtual environment for DTCWT subprocess (optional)

    Returns:
    - detected: Boolean array indicating detected bursts
    - variance_traj: Variance trajectory over time
    - threshold_var: Variance threshold used for detection
    - enhanced: Enhanced signal magnitude
    """
    # Step 1: Enhance signal with DTCWT highpass coefficients
    coeffs = run_dtcwt(signal, t, nlevels=nlevels, venv_path=venv_path)
    highpass_level_0 = np.array(coeffs[0], dtype=complex).flatten()

    # Upsample to match signal length
    enhanced = np.zeros_like(signal, dtype=complex)
    coeff_len = len(highpass_level_0)
    for i in range(len(signal)):
        idx = i // 2
        if idx < coeff_len:
            enhanced[i] = highpass_level_0[idx]

    enhanced_mag = np.abs(enhanced)
    print(f"Enhanced signal max magnitude: {np.max(enhanced_mag)}")

    # Step 2: Compute variance trajectory
    variance_traj = np.zeros(len(signal) - window_size + 1)
    for i in range(len(variance_traj)):
        window = enhanced_mag[i : i + window_size]
        variance_traj[i] = np.var(window)

    # Step 3: Threshold based on noise (incorrectly uses signal region)
    noise_mask = t[: len(variance_traj)] * 1e6 < -1.0  # This is broken as t starts at 0
    noise_var = (
        variance_traj[noise_mask] if np.any(noise_mask) else variance_traj[:window_size]
    )
    threshold_var = np.mean(noise_var) + threshold_multiplier * np.std(noise_var)
    print(f"Noise variance mean: {np.mean(noise_var)}, std: {np.std(noise_var)}")
    print(f"Variance threshold: {threshold_var}")
    print(f"Variance trajectory max: {np.max(variance_traj)}")

    # Step 4: Detect bursts
    detected = np.zeros_like(signal, dtype=bool)
    detected_var = variance_traj > threshold_var
    for i in range(len(detected_var)):
        if detected_var[i]:
            end_idx = min(len(signal), i + window_size)
            detected[i:end_idx] = True

    return detected, variance_traj, threshold_var, enhanced_mag


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
