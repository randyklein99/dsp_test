# detectors.py
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
        tuple: (detected, mag_dtcwt, threshold_dtcwt, t_dtcwt) where detected is a boolean array,
               mag_dtcwt is the magnitude coefficients, threshold_dtcwt is the threshold,
               and t_dtcwt is the downsampled time array.
    """
    mag_dtcwt = run_dtcwt(signal, t)
    t_dtcwt = t[::2]
    noise_mask = t_dtcwt * 1e6 < 10.0
    noise_mag = mag_dtcwt[noise_mask]
    threshold_dtcwt = np.mean(noise_mag) + threshold_multiplier * np.std(noise_mag)
    detected = mag_dtcwt > threshold_dtcwt
    return detected, mag_dtcwt, threshold_dtcwt, t_dtcwt

def variance_trajectory_detector(signal, t, window_size=30, threshold_multiplier=4.0, nlevels=4):
    # Denoising with DTCWT
    coeffs = run_dtcwt(signal, t)
    coeff_len_per_level = len(signal) // (2 ** nlevels)
    highpass_level_0 = coeffs[:coeff_len_per_level]
    threshold = np.std(np.abs(highpass_level_0)) * np.sqrt(2 * np.log(len(signal)))
    denoised = signal.copy()
    for i in range(len(signal)):
        idx = i // 2
        if idx < len(highpass_level_0) and np.abs(highpass_level_0[idx]) < threshold:
            denoised[i] = 0

    # Variance trajectory
    variance_traj = np.zeros(len(signal) - window_size + 1)
    for i in range(len(variance_traj)):
        variance_traj[i] = np.var(denoised[i:i + window_size])

    # Thresholding
    noise_mask = t[:len(variance_traj)] * 1e6 < 10.0
    noise_var = variance_traj[noise_mask]
    threshold_var = np.mean(noise_var) + threshold_multiplier * np.std(noise_var)
    detected = np.zeros_like(signal, dtype=bool)
    detected_var = variance_traj > threshold_var
    for i in range(len(detected_var)):
        if detected_var[i]:
            detected[i:i + window_size] = True

    return detected, variance_traj, threshold_var, denoised
    
def matched_filter_detector(signal, template, threshold_fraction=0.5, fs=20e6):
    """
    Detect bursts using a matched filter.

    Args:
        signal (np.ndarray): Input signal.
        template (np.ndarray): Template signal (e.g., STF preamble).
        threshold_fraction (float): Fraction of max output for detection threshold.
        fs (float): Sampling frequency (Hz).

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
    template_duration = len(template) / fs
    start_idx = max(0, peak_idx - int(template_duration * fs / 2))
    end_idx = min(len(signal), peak_idx + int(template_duration * fs / 2))
    detected = np.zeros_like(mf_output, dtype=bool)
    detected[start_idx:end_idx] = True
    return detected, mf_output, threshold_mf