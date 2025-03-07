# detectors.py
import numpy as np
from scipy import signal as sig
from dtcwt_utils import run_dtcwt

def dtcwt_detector(signal, t, threshold_multiplier=3.0):
    """
    Detect bursts using DTCWT energy.

    Args:
        signal (np.ndarray): Input signal.
        t (np.ndarray): Time array.
        threshold_multiplier (float): Multiplier for noise threshold (mean + k*std).

    Returns:
        tuple: (detected, mag_dtcwt, threshold_dtcwt, t_dtcwt) where detected is a boolean array,
               mag_dtcwt is the magnitude coefficients, threshold_dtcwt is the threshold,
               and t_dtcwt is the downsampled time array.
    """
    mag_dtcwt = run_dtcwt(signal, t)
    t_dtcwt = t[::2]  # Downsampled time array
    noise_mask = t_dtcwt * 1e6 < 10.0  # Noise region up to 10 µs
    noise_mag = mag_dtcwt[noise_mask]
    threshold_dtcwt = np.mean(noise_mag) + threshold_multiplier * np.std(noise_mag)
    detected = mag_dtcwt > threshold_dtcwt
    return detected, mag_dtcwt, threshold_dtcwt, t_dtcwt

def matched_filter_detector(signal, template, threshold_fraction=0.5):
    """
    Detect bursts using a matched filter.

    Args:
        signal (np.ndarray): Input signal.
        template (np.ndarray): Template signal (e.g., STF preamble).
        threshold_fraction (float): Fraction of max output for detection threshold.

    Returns:
        tuple: (detected, mf_output, threshold_mf) where detected is a boolean array,
               mf_output is the correlation output, and threshold_mf is the threshold.
    """
    mf_output = sig.correlate(signal, template, mode='same')
    mf_output = np.abs(mf_output) / np.max(np.abs(mf_output))
    max_mf = np.max(mf_output)
    threshold_mf = threshold_fraction * max_mf
    detected = mf_output > threshold_mf
    peak_idx = np.argmax(mf_output)
    start_idx = np.where(mf_output[:peak_idx] < threshold_mf)[0][-1] if np.any(mf_output[:peak_idx] < threshold_mf) else 0
    end_idx = np.where(mf_output[peak_idx:] < threshold_mf)[0][0] + peak_idx if np.any(mf_output[peak_idx:] < threshold_mf) else -1
    detected = np.zeros_like(mf_output, dtype=bool)
    if end_idx != -1:
        detected[start_idx:end_idx] = True
    return detected, mf_output, threshold_mf