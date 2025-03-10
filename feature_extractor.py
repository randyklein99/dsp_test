import numpy as np
from scipy import stats
from dtcwt import Transform1d
from scipy.signal import medfilt, butter, filtfilt
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
        # Apply light smoothing to reduce noise before unwrapping
        phase_smooth = medfilt(phase, kernel_size=3)
        # Detect and correct large phase jumps before unwrapping
        phase_diff = np.diff(phase_smooth)
        phase_jump_threshold = 0.1 * np.pi  # Aggressive jump detection
        jump_indices = np.where(np.abs(phase_diff) > phase_jump_threshold)[0]
        max_iterations = 5  # Limit iterations to prevent infinite loops
        iteration = 0
        while len(jump_indices) > 0 and iteration < max_iterations:
            for idx in jump_indices:
                correction = -2 * np.pi * np.sign(phase_diff[idx])
                phase_smooth[idx + 1 :] += correction
            phase_diff = np.diff(phase_smooth)
            jump_indices = np.where(np.abs(phase_diff) > phase_jump_threshold)[0]
            iteration += 1
        phase_unwrapped = np.unwrap(phase_smooth)
        # Apply minimal smoothing before differentiation to retain variation
        phase_unwrapped = medfilt(phase_unwrapped, kernel_size=3)
        # Remove trend
        t_samples = np.arange(len(phase_unwrapped))
        poly = Polynomial.fit(t_samples, phase_unwrapped, deg=5)  # Quintic trend removal
        phase_unwrapped = phase_unwrapped - poly(t_samples)
        # Normalize phase to [-π, π]
        phase_unwrapped = np.mod(phase_unwrapped + np.pi, 2 * np.pi) - np.pi
        delta_t = max((2**j) / fs, 1 / fs)  # Minimum delta_t to prevent excessive amplification
        if len(phase_unwrapped) > 1:  # Need at least 2 points for fitting
            # Estimate frequency from phase difference
            frequency = np.diff(phase_unwrapped) / (2 * np.pi * delta_t)
            frequency = np.pad(frequency, (0, 1), mode="edge")  # Extend to match length
            # Apply light smoothing after differentiation to retain variation
            frequency = medfilt(frequency, kernel_size=3)
            # Apply low-pass filter (cutoff at 50 Hz) only if length is sufficient
            if len(frequency) > 9:  # padlen is 9 for order 2 Butterworth filter
                b, a = butter(2, 50 / (fs / 2), btype="low")
                frequency = filtfilt(b, a, frequency)
            # Additional light moving average for stability
            window_size = 2
            frequency = np.convolve(frequency, np.ones(window_size) / window_size, mode="same")
            # Clip to expected range (±75 Hz to allow more variation)
            frequency = np.clip(frequency, -75, 75)
        else:
            frequency = np.array([])
        # Amplify LTF amplitude
        amplitude_ltf = amplitude.copy()
        if len(amplitude) > 160:
            amplitude_ltf[160:] *= 3  # Increased scaling factor
        t_level = np.arange(len(highpass_j)) * delta_t * 1e6  # Convert to µs for 16 µs range
        characteristics.append(
            {
                "amplitude": amplitude_ltf,
                "phase_unwrapped": phase_unwrapped,
                "frequency": frequency,
                "t": t_level,
            }
        )
    return characteristics

def center_characteristics(characteristics):
    """
    Center the characteristics by removing means and correcting phase.

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
        # Center amplitude only if mean is significant
        amp_mean = np.mean(amp)
        centered_amp = amp - amp_mean if abs(amp_mean) > 0.1 else amp
        # Preserve phase and frequency variations without mean subtraction
        centered_phi = phi
        centered_freq = freq
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
        freq_stats = (
            [np.var(freq), stats.skew(freq), stats.kurtosis(freq)]
            if len(freq) > 1
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

    # Perform DTCWT decomposition for each subregion directly
    transform = Transform1d()
    nlevels = 5
    # Subregion 1 (STF)
    target_length1 = 2 ** int(np.ceil(np.log2(len(sub1))))
    sub1_padded = np.pad(sub1, (0, target_length1 - len(sub1)), mode="constant")
    coeffs1 = transform.forward(sub1_padded, nlevels=nlevels)
    coeffs1 = [np.array(c) for c in coeffs1.highpasses]
    # Subregion 2 (LTF)
    target_length2 = 2 ** int(np.ceil(np.log2(len(sub2))))
    sub2_padded = np.pad(sub2, (0, target_length2 - len(sub2)), mode="constant")
    coeffs2 = transform.forward(sub2_padded, nlevels=nlevels)
    coeffs2 = [np.array(c) for c in coeffs2.highpasses]
    # Subregion 3 (Combined)
    target_length3 = 2 ** int(np.ceil(np.log2(len(sub3))))
    sub3_padded = np.pad(sub3, (0, target_length3 - len(sub3)), mode="constant")
    coeffs3 = transform.forward(sub3_padded, nlevels=nlevels)
    coeffs3 = [np.array(c) for c in coeffs3.highpasses]

    # Compute characteristics and center them
    chars1 = compute_characteristics(coeffs1, fs, nlevels=5)
    chars2 = compute_characteristics(coeffs2, fs, nlevels=5)
    chars3 = compute_characteristics(coeffs3, fs, nlevels=5)
    centered1 = [center_characteristics([c])[0] for c in chars1]
    centered2 = [center_characteristics([c])[0] for c in chars2]
    centered3 = [center_characteristics([c])[0] for c in chars3]

    # Compute statistics for each level and subregion, prioritizing level 3
    stats1 = compute_statistics([centered1[3]])
    stats2 = compute_statistics([centered2[3]])
    stats3 = compute_statistics([centered3[3]])

    # Flatten the feature vector (3 subregions x 1 level x 3 characteristics x 3 stats)
    feature_vector = np.array(stats1 + stats2 + stats3).flatten()
    return feature_vector