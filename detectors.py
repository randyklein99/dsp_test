import numpy as np
from typing import Tuple, Optional
from scipy import signal as sig

def variance_trajectory_detector(
    signal: np.ndarray,
    t: np.ndarray,
    processed_mag: np.ndarray,
    window_size: int = 320,
    threshold_multiplier: float = 1.5,
    debug_level: int = 0
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Detect bursts using a variance trajectory on a pre-processed signal magnitude.

    Args:
        signal (np.ndarray): Input signal array (complex-valued, unused for computation but kept for context).
        t (np.ndarray): Time vector corresponding to the signal.
        processed_mag (np.ndarray): Pre-processed signal magnitude (e.g., raw or denoised).
        window_size (int): Size of the sliding window (default: 320 ~ 16 µs at 20 MHz).
        threshold_multiplier (float): Multiplier for variance threshold (default: 1.5).
        debug_level (int): Debugging level (0 = no debug, 1 = basic, 2 = detailed).

    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray]: 
            - variance_traj: Variance trajectory over time.
            - threshold: Variance threshold used for detection.
            - processed_mag: Input processed magnitude (passed through).
            - detected: Boolean array indicating detected bursts.
    """
    if len(signal) == 0 or len(t) == 0 or len(processed_mag) == 0:
        raise ValueError("Signal, time array, and processed magnitude must be non-empty")
    if len(signal) != len(t) or len(signal) != len(processed_mag):
        raise ValueError("Signal, time array, and processed magnitude lengths must match")

    # Step 1: Compute variance trajectory
    variance_traj = np.zeros(len(signal) - window_size + 1)
    for i in range(len(variance_traj)):
        window = processed_mag[i : i + window_size]
        variance_traj[i] = np.var(window)

    # Step 2: Threshold and detect bursts
    t_var = t[window_size - 1: len(variance_traj) + window_size - 1]  # Time at end of windows
    noise_mask = (t_var * 1e6 >= 16.05) & (t_var * 1e6 < 19.95)  # Exclude 20 µs
    noise_var = variance_traj[noise_mask] if np.any(noise_mask) else variance_traj[1:-1]
    if debug_level >= 1:
        print(f"VT Values in Noise Region (16.05-19.95 µs): {noise_var}")
        print(f"Number of Noise Region Samples: {len(noise_var)}")
    if debug_level >= 2:
        for i, val in enumerate(noise_var):
            print(f"VT[{i}] at {t_var[noise_mask][i] * 1e6:.2f} µs: {val:.6f}")
    noise_median = np.median(noise_var) if noise_var.size > 0 else 0.0
    if debug_level >= 1:
        print(f"Noise Region Median VT (16.05-19.95 µs): {noise_median}")
    threshold = 1.2 * np.max(noise_var) if noise_var.size > 0 else 0.005  # 1.2 * max of noise VT values
    if debug_level >= 1:
        print(f"Threshold: {threshold}")
    if debug_level >= 1:
        print(f"Variance trajectory max: {np.max(variance_traj)}")

    # Step 3: Detect bursts
    detected = np.zeros(len(signal), dtype=bool)
    for i in range(len(variance_traj)):
        if variance_traj[i] > threshold and t[i + window_size - 1] * 1e6 >= 20.0:
            start_idx = i
            end_time = t[i + window_size - 1] + 16e-6  # 16 µs after the window end
            end_idx = np.searchsorted(t, end_time, side='right')
            end_idx = min(len(signal), end_idx)
            detected[start_idx:end_idx] = True  # Use signal indices directly
            break

    return variance_traj, threshold, processed_mag, detected

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