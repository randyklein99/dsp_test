import numpy as np
from scipy import stats
from dtcwt import Transform1d
from scipy.signal import medfilt
from numpy.polynomial.polynomial import Polynomial

def compute_characteristics(coeffs, fs, nlevels=5):
    """
    Compute amplitude, phase, and frequency characteristics from DTCWT coefficients.

    Args:
        coeffs (list): List of highpass coefficients for each level.
        fs (float): Sampling frequency in Hz.
        nlevels (int): Number of decomposition levels.

    Returns:
        list: List of dictionaries containing 'amplitude', 'phase_unwrapped', 'frequency', and 't' for each level.
    """
    characteristics = []
    for j in range(nlevels):
        highpass_j = np.squeeze(coeffs[j])  # Remove extra dimensions to ensure 1D array
        amplitude = np.abs(highpass_j)
        phase = np.angle(highpass_j)
        # Simplified phase unwrapping
        phase_unwrapped = np.unwrap(phase)
        print(f"Level {j} - Phase unwrapped (before trend removal): min={np.min(phase_unwrapped):.4f}, max={np.max(phase_unwrapped):.4f}")
        # MATLAB-style phase detrending using mean frequency
        delta_t = max((2**j) / fs, 1 / fs)  # Minimum delta_t
        if len(phase_unwrapped) > 1:
            # Compute instantaneous frequency using gradient (MATLAB style)
            t_samples = np.arange(len(phase_unwrapped)) * delta_t
            frequency = np.gradient(phase_unwrapped, delta_t) / (2 * np.pi)
            print(f"Level {j} - Frequency (before centering): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
            # Center frequency
            mu_f = np.mean(frequency)
            frequency = frequency - mu_f
            print(f"Level {j} - Frequency (after centering): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
            # Light smoothing to reduce spikiness
            if len(frequency) > 3:
                frequency = medfilt(frequency, kernel_size=3)
            print(f"Level {j} - Frequency (after smoothing): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
            # Widen clipping range to capture more detail
            frequency = np.clip(frequency, -20000, 20000)
            print(f"Level {j} - Frequency (after clipping): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
            # Detrend phase using mean frequency
            t_samples_indices = np.arange(len(phase_unwrapped))
            phase_detrended = phase_unwrapped - 2 * np.pi * t_samples_indices * mu_f / fs
        else:
            frequency = np.array([])
            phase_detrended = phase_unwrapped
        print(f"Level {j} - Phase unwrapped (after detrending): min={np.min(phase_detrended):.4f}, max={np.max(phase_detrended):.4f}")
        # Amplify LTF amplitude
        amplitude_ltf = amplitude.copy()
        ltf_start_idx = int(8e-6 / delta_t)  # Adjust for downsampling at this level
        if len(amplitude) > ltf_start_idx:
            amplitude_ltf[ltf_start_idx:] *= 3  # Amplify LTF portion
        characteristics.append(
            {
                "amplitude": amplitude_ltf,
                "phase_unwrapped": phase_detrended,  # Use the detrended phase
                "frequency": frequency,
                "t": np.linspace(0, 16, len(highpass_j)),  # Ensure consistent time axis
            }
        )
    return characteristics

def center_characteristics(characteristics):
    """
    Center the characteristics by removing means, matching MATLAB logic.

    Args:
        characteristics (list): List of dictionaries containing 'amplitude', 'phase_unwrapped', 'frequency', and 't'.

    Returns:
        list: List of dictionaries with centered 'amplitude', 'phase', and 'frequency'.
    """
    centered = []
    for char in characteristics:
        amp = char["amplitude"]
        phi = char["phase_unwrapped"]
        freq = char["frequency"]
        t = char["t"]
        # Center all features (MATLAB style)
        centered_amp = amp - np.mean(amp)
        centered_phi = phi - np.mean(phi)
        centered_freq = freq  # Already centered in compute_characteristics
        centered.append(
            {
                "amplitude": centered_amp,
                "phase": centered_phi,
                "frequency": centered_freq,
                "t": t,
            }
        )
    return centered

def compute_statistics(centered_characteristics):
    """
    Compute variance, skewness, and kurtosis for each characteristic.

    Args:
        centered_characteristics (list): List of dictionaries with centered 'amplitude', 'phase', and 'frequency'.

    Returns:
        list: List of lists containing [var_amp, skew_amp, kurt_amp, var_phi, skew_phi, kurt_phi, var_freq, skew_freq, kurt_freq].
    """
    stats_list = []
    for level, char in enumerate(centered_characteristics):
        amp = char["amplitude"]
        phi = char["phase"]
        freq = char["frequency"]
        # Compute stats only if arrays are not empty
        amp_stats = (
            [np.var(amp), stats.skew(amp), stats.kurtosis(amp)]
            if len(amp) > 1
            else [0.0, 0.0, 0.0]
        )
        phi_stats = (
            [np.var(phi), stats.skew(phi), stats.kurtosis(phi)]
            if len(phi) > 1
            else [0.0, 0.0, 0.0]
        )
        print(f"Level {level} - Frequency data: min={np.min(freq):.4f}, max={np.max(freq):.4f}, var={np.var(freq):.4f}")
        freq_stats = (
            [np.var(freq), stats.skew(freq), stats.kurtosis(freq)]
            if len(freq) > 1 and np.var(freq) > 1e-10
            else [0.0, 0.0, 0.0]
        )
        # Replace nan/inf values with 0 to ensure finite feature vector
        all_stats = amp_stats + phi_stats + freq_stats
        adjusted_stats = [0.0 if np.isnan(x) or np.isinf(x) else x for x in all_stats]
        stats_list.append(adjusted_stats)
    return stats_list

def extract_features(signal, t, fs, preamble_start=10e-6):
    """
    Extract a 135-element feature vector from the signal based on DTCWT characteristics.

    Args:
        signal (np.ndarray): Input signal array.
        t (np.ndarray): Time array corresponding to the signal.
        fs (float): Sampling frequency in Hz.
        preamble_start (float): Start time of the preamble in seconds (default: 10e-6).

    Returns:
        np.ndarray: 135-element feature vector.
    """
    if len(signal) != len(t):
        raise ValueError(
            f"Signal length ({len(signal)}) does not match time array length ({len(t)})"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or inf values")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling frequency fs={fs} must be positive and finite")

    # Define subregion durations
    short_duration = 8e-6  # 8 µs for short subregion (STF)
    long_duration = 8e-6  # 8 µs for long subregion (LTF with guard)
    combined_duration = short_duration + long_duration  # 16 µs for combined subregion

    # Compute expected subregion lengths (ensure divisible by 2)
    short_len = int(fs * short_duration)
    if short_len % 2 != 0:
        short_len -= 1
    combined_len = int(fs * combined_duration)
    if combined_len % 2 != 0:
        combined_len -= 1

    # Compute preamble start index
    idx_start = int(preamble_start * fs)
    idx_start = max(0, min(idx_start, len(signal) - combined_len))

    # Compute subregion indices
    idx_sub1_end = idx_start + short_len  # End of short subregion (STF)
    idx_sub2_end = idx_start + combined_len  # End of combined subregion
    idx_sub2_start = idx_sub1_end  # Start of long subregion (LTF)

    # Ensure indices are within bounds
    idx_sub2_end = min(idx_sub2_end, len(signal))
    idx_sub1_end = min(idx_sub1_end, idx_sub2_end)

    # Extract subregions
    sub1 = signal[idx_start:idx_sub1_end]  # Short subregion (STF)
    sub2 = signal[idx_sub2_start:idx_sub2_end]  # Long subregion (LTF)
    sub3 = signal[idx_start:idx_sub2_end]  # Combined subregion
    t_sub1 = t[idx_start:idx_sub1_end]
    t_sub2 = t[idx_sub2_start:idx_sub2_end]
    t_sub3 = t[idx_start:idx_sub2_end]

    # Validate subregion lengths
    if len(sub1) != short_len:
        raise ValueError(f"Short subregion length is {len(sub1)}, expected {short_len}")
    if len(sub2) != (combined_len - short_len):
        raise ValueError(
            f"Long subregion length is {len(sub2)}, expected {combined_len - short_len}"
        )
    if len(sub3) != combined_len:
        raise ValueError(
            f"Combined subregion length is {len(sub3)}, expected {combined_len}"
        )

    # Perform DC bias removal (MATLAB style: Sig = Sig - mean(Sig))
    sub1 = sub1 - np.mean(sub1)
    sub2 = sub2 - np.mean(sub2)
    sub3 = sub3 - np.mean(sub3)

    # Perform DTCWT decomposition for each subregion directly
    transform = Transform1d()
    nlevels = 5
    # Subregion 1 (STF)
    target_length1 = 2 ** int(np.ceil(np.log2(len(sub1))))
    sub1_padded = np.pad(sub1, (0, target_length1 - len(sub1)), mode="reflect")
    coeffs1 = transform.forward(sub1_padded, nlevels=nlevels)
    coeffs1 = [np.array(c) for c in coeffs1.highpasses]
    # Subregion 2 (LTF)
    target_length2 = 2 ** int(np.ceil(np.log2(len(sub2))))
    sub2_padded = np.pad(sub2, (0, target_length2 - len(sub2)), mode="reflect")
    coeffs2 = transform.forward(sub2_padded, nlevels=nlevels)
    coeffs2 = [np.array(c) for c in coeffs2.highpasses]
    # Subregion 3 (Combined)
    target_length3 = 2 ** int(np.ceil(np.log2(len(sub3))))
    sub3_padded = np.pad(sub3, (0, target_length3 - len(sub3)), mode="reflect")
    coeffs3 = transform.forward(sub3_padded, nlevels=nlevels)
    coeffs3 = [np.array(c) for c in coeffs3.highpasses]

    # Compute characteristics and center them
    chars1 = compute_characteristics(coeffs1, fs, nlevels=5)
    chars2 = compute_characteristics(coeffs2, fs, nlevels=5)
    chars3 = compute_characteristics(coeffs3, fs, nlevels=5)
    centered1 = [center_characteristics([c])[0] for c in chars1]
    centered2 = [center_characteristics([c])[0] for c in chars2]
    centered3 = [center_characteristics([c])[0] for c in chars3]

    # Compute statistics for all levels (to match MATLAB's 135-element feature vector)
    stats1 = compute_statistics(centered1)  # All levels for subregion 1
    stats2 = compute_statistics(centered2)  # All levels for subregion 2
    stats3 = compute_statistics(centered3)  # All levels for subregion 3

    # Flatten the feature vector (3 subregions x 5 levels x 9 stats = 135 elements)
    feature_vector = np.array(stats1 + stats2 + stats3).flatten()
    return feature_vector