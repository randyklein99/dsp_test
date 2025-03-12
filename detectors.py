import numpy as np
from typing import Tuple, Optional
from scipy import signal as sig

def variance_trajectory_detector(
    signal: np.ndarray,
    t: np.ndarray,
    window_size: int = 320,
    threshold: float = 0.01,
    debug_level: int = 0
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """
    Detect the first point where variance exceeds a threshold using a sliding window.

    Args:
        signal (np.ndarray): Complex-valued input signal array.
        t (np.ndarray): Time vector corresponding to the signal.
        window_size (int): Size of the sliding window (default: 320 ~ 16 µs at 20 MHz).
        threshold (float): Absolute variance threshold for detection (default: 0.01).
        debug_level (int): Debugging level (0 = no debug, 1 = basic, 2 = detailed).

    Returns:
        Tuple[np.ndarray, float, np.ndarray, int]: 
            - variance_traj: Variance trajectory over time.
            - threshold: Variance threshold used for detection (passed through).
            - mag: Magnitude of the input signal.
            - trigger_idx: Index where variance first exceeds threshold, or -1 if no detection.
    """
    if len(signal) == 0 or len(t) == 0:
        raise ValueError("Signal and time array must be non-empty")
    if len(signal) != len(t):
        raise ValueError("Signal and time array lengths must match")
    if window_size > len(signal):
        raise ValueError("Window size must not exceed signal length")

    # Compute magnitude internally
    mag = np.abs(signal)

    # Compute variance trajectory
    variance_traj = np.zeros(len(signal) - window_size + 1)
    for i in range(len(variance_traj)):
        window = mag[i : i + window_size]
        variance_traj[i] = np.var(window)

    if debug_level >= 1:
        print(f"Variance trajectory max: {np.max(variance_traj):.6f}")
        print(f"Threshold: {threshold:.6f}")

    # Find first threshold crossing
    trigger_idx = -1
    for i in range(len(variance_traj)):
        if variance_traj[i] > threshold:
            trigger_idx = i  # Index relative to variance_traj (shifted by window_size - 1 in signal)
            if debug_level >= 2:
                print(f"Trigger at index {i}, time {t[i + window_size - 1] * 1e6:.2f} µs, variance {variance_traj[i]:.6f}")
            break

    return variance_traj, threshold, mag, trigger_idx

def analyze_noise(
    signal: np.ndarray,
    t: np.ndarray,
    window_size: int = 320,
    noise_region: Optional[Tuple[float, float]] = None,
    debug_level: int = 0
) -> dict:
    """
    Analyze variance trajectory characteristics over a signal or specified noise region.

    Args:
        signal (np.ndarray): Complex-valued input signal array.
        t (np.ndarray): Time vector corresponding to the signal.
        window_size (int): Size of the sliding window (default: 320 ~ 16 µs at 20 MHz).
        noise_region (Optional[Tuple[float, float]]): Time range (start, end) in seconds to analyze as noise; if None, use entire signal.
        debug_level (int): Debugging level (0 = no debug, 1 = basic, 2 = detailed).

    Returns:
        dict: Noise VT statistics including 'mean_vt', 'max_vt', 'std_vt', 'percentile_95_vt'.
    """
    if len(signal) == 0 or len(t) == 0:
        raise ValueError("Signal and time array must be non-empty")
    if len(signal) != len(t):
        raise ValueError("Signal and time array lengths must match")
    if window_size > len(signal):
        raise ValueError("Window size must not exceed signal length")

    mag = np.abs(signal)
    variance_traj = np.zeros(len(signal) - window_size + 1)
    for i in range(len(variance_traj)):
        window = mag[i : i + window_size]
        variance_traj[i] = np.var(window)

    t_var = t[window_size - 1: len(variance_traj) + window_size - 1]
    if noise_region is not None:
        start, end = noise_region
        mask = (t_var >= start) & (t_var <= end)
        noise_vt = variance_traj[mask]
    else:
        noise_vt = variance_traj

    if len(noise_vt) == 0:
        noise_vt = np.array([0.0])  # Default to zero if no region specified or empty

    stats = {
        'mean_vt': float(np.mean(noise_vt)),
        'max_vt': float(np.max(noise_vt)),
        'std_vt': float(np.std(noise_vt)),
        'percentile_95_vt': float(np.percentile(noise_vt, 95))
    }

    if debug_level >= 1:
        print(f"Noise VT stats - Mean: {stats['mean_vt']:.6f}, Max: {stats['max_vt']:.6f}, Std: {stats['std_vt']:.6f}, 95th: {stats['percentile_95_vt']:.6f}")

    return stats

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