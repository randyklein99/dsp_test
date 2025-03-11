import numpy as np
from scipy import stats
from dtcwt import Transform1d
from scipy.signal import medfilt
from numpy.polynomial.polynomial import Polynomial

def compute_characteristics(coeffs, fs, nlevels=5, debug_level=0):
    """
    Compute amplitude, phase, and frequency characteristics from DTCWT coefficients.

    Args:
        coeffs (list): List of highpass coefficients for each level.
        fs (float): Sampling frequency in Hz.
        nlevels (int): Number of decomposition levels.
        debug_level (int): Debugging level (0 = no debug, 1 = basic, 2 = detailed).

    Returns:
        list: List of dictionaries containing 'amplitude', 'phase_unwrapped', 'frequency', and 't' for each level.
    """
    characteristics = []
    for j in range(nlevels):
        highpass_j = np.squeeze(coeffs[j])  # Remove extra dimensions to ensure 1D array
        if debug_level >= 2:
            print(f"Level {j} - highpass_j type: {type(highpass_j)}, shape: {highpass_j.shape}")
        amplitude = np.abs(highpass_j)
        phase = np.angle(highpass_j)
        # Simplified phase unwrapping
        phase_unwrapped = np.unwrap(phase)
        if debug_level >= 2:
            print(f"Level {j} - Phase unwrapped (before trend removal): min={np.min(phase_unwrapped):.4f}, max={np.max(phase_unwrapped):.4f}")
        # MATLAB-style phase detrending using polynomial fit
        delta_t = max((2**j) / fs, 1 / fs)  # Minimum delta_t
        if len(phase_unwrapped) > 1:
            t_samples = np.arange(len(phase_unwrapped)) * delta_t
            # Fit a polynomial (degree 2 for linear + quadratic trend) to remove the dominant phase trend
            coeffs_poly = Polynomial.fit(t_samples, phase_unwrapped, deg=2)
            phase_detrended = phase_unwrapped - coeffs_poly(t_samples)
            if debug_level >= 2:
                print(f"Level {j} - Phase unwrapped (after polynomial detrending): min={np.min(phase_detrended):.4f}, max={np.max(phase_detrended):.4f}")
            # Compute instantaneous frequency using gradient on detrended phase
            frequency = np.gradient(phase_detrended, delta_t) / (2 * np.pi)
            if debug_level >= 2:
                print(f"Level {j} - Frequency (after gradient): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
            # Center frequency
            mu_f = np.mean(frequency)
            frequency = frequency - mu_f
            if debug_level >= 2:
                print(f"Level {j} - Frequency (after centering): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
            # Light smoothing to reduce spikiness, with adaptive kernel size
            if len(frequency) > 3:
                # Adjust kernel size based on level to avoid over-smoothing at higher levels
                base_kernel = max(3 - j, 1)
                kernel_size = base_kernel if base_kernel % 2 == 1 else base_kernel + 1  # Ensure odd
                frequency = medfilt(frequency, kernel_size=kernel_size)
            if debug_level >= 2:
                print(f"Level {j} - Frequency (after smoothing with kernel size {kernel_size}): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
            # Scale frequency by the center frequency of the wavelet band at this level
            center_freq = fs / (2**(j+1))  # Center frequency of the highpass band
            frequency = frequency / center_freq  # Normalize to a relative deviation (unitless)
            if debug_level >= 1:
                print(f"Level {j} - Frequency (after scaling by center freq {center_freq:.2f} Hz): min={np.min(frequency):.4f}, max={np.max(frequency):.4f}")
        else:
            frequency = np.array([])
            phase_detrended = phase_unwrapped
        if debug_level >= 2:
            print(f"Level {j} - Phase unwrapped (final): min={np.min(phase_detrended):.4f}, max={np.max(phase_detrended):.4f}")
        # Amplify LTF amplitude
        amplitude_ltf = amplitude.copy()
        ltf_start_idx = int(8e-6 / delta_t)
        if len(amplitude) > ltf_start_idx:
            amplitude_ltf[ltf_start_idx:] *= 3
        characteristics.append(
            {
                "amplitude": amplitude_ltf,
                "phase_unwrapped": phase_detrended,
                "frequency": frequency,
                "t": np.linspace(0, 16, len(highpass_j)),
            }
        )
    return characteristics

def center_characteristics(characteristics):
    """
    Center the characteristics by removing means, matching MATLAB logic.

    Args:
        characteristics (dict): Dictionary containing characteristics.

    Returns:
        dict: Centered characteristics.
    """
    amp = characteristics["amplitude"]
    phi = characteristics["phase_unwrapped"]
    freq = characteristics["frequency"]
    t = characteristics["t"]
    centered_amp = amp - np.mean(amp)
    centered_phi = phi - np.mean(phi)
    centered_freq = freq  # Already centered in compute_characteristics
    return {
        "amplitude": centered_amp,
        "phase": centered_phi,
        "frequency": centered_freq,
        "t": t,
    }

def compute_statistics(centered_characteristics, debug_level=0):
    """
    Compute variance, skewness, and kurtosis for each characteristic.

    Args:
        centered_characteristics (list): List of centered characteristics.
        debug_level (int): Debugging level (0 = no debug, 1 = basic, 2 = detailed).

    Returns:
        list: List of statistics for each level.
    """
    stats_list = []
    for level, char in enumerate(centered_characteristics):
        amp = char["amplitude"]
        phi = char["phase"]
        freq = char["frequency"]
        amp_stats = [np.var(amp), stats.skew(amp), stats.kurtosis(amp)] if len(amp) > 1 else [0.0, 0.0, 0.0]
        phi_stats = [np.var(phi), stats.skew(phi), stats.kurtosis(phi)] if len(phi) > 1 else [0.0, 0.0, 0.0]
        if debug_level >= 1:
            print(f"Level {level} - Frequency data: min={np.min(freq):.4f}, max={np.max(freq):.4f}, var={np.var(freq):.4f}")
        freq_stats = [np.var(freq), stats.skew(freq), stats.kurtosis(freq)] if len(freq) > 1 and np.var(freq) > 1e-10 else [0.0, 0.0, 0.0]
        all_stats = amp_stats + phi_stats + freq_stats
        adjusted_stats = [0.0 if np.isnan(x) or np.isinf(x) else x for x in all_stats]
        stats_list.append(adjusted_stats)
    return stats_list

def extract_features(signal, t, fs, preamble_start=10e-6, debug_level=0):
    """
    Extract a 135-element feature vector and centered characteristics from the signal based on DTCWT.

    Args:
        signal (np.ndarray): Input signal array.
        t (np.ndarray): Time array corresponding to the signal.
        fs (float): Sampling frequency in Hz.
        preamble_start (float): Start time of the preamble in seconds (default: 10e-6).
        debug_level (int): Debugging level (0 = no debug, 1 = basic, 2 = detailed).

    Returns:
        tuple: (feature_vector, centered_chars) where:
            - feature_vector: 135-element 1D NumPy array.
            - centered_chars: Dictionary with keys 'sub1', 'sub2', 'sub3', each containing a list of centered characteristics for each level.
    """
    if len(signal) != len(t):
        raise ValueError(f"Signal length ({len(signal)}) does not match time array length ({len(t)})")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or inf values")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling frequency fs={fs} must be positive and finite")

    short_duration = 8e-6
    long_duration = 8e-6
    combined_duration = short_duration + long_duration

    short_len = int(fs * short_duration)
    if short_len % 2 != 0:
        short_len -= 1
    combined_len = int(fs * combined_duration)
    if combined_len % 2 != 0:
        combined_len -= 1

    idx_start = int(preamble_start * fs)
    idx_start = max(0, min(idx_start, len(signal) - combined_len))

    idx_sub1_end = idx_start + short_len
    idx_sub2_end = idx_start + combined_len
    idx_sub2_start = idx_sub1_end

    idx_sub2_end = min(idx_sub2_end, len(signal))
    idx_sub1_end = min(idx_sub1_end, idx_sub2_end)

    sub1 = signal[idx_start:idx_sub1_end]
    sub2 = signal[idx_sub2_start:idx_sub2_end]
    sub3 = signal[idx_start:idx_sub2_end]
    t_sub1 = t[idx_start:idx_sub1_end]
    t_sub2 = t[idx_sub2_start:idx_sub2_end]
    t_sub3 = t[idx_start:idx_sub2_end]

    if len(sub1) != short_len:
        raise ValueError(f"Short subregion length is {len(sub1)}, expected {short_len}")
    if len(sub2) != (combined_len - short_len):
        raise ValueError(f"Long subregion length is {len(sub2)}, expected {combined_len - short_len}")
    if len(sub3) != combined_len:
        raise ValueError(f"Combined subregion length is {len(sub3)}, expected {combined_len}")

    sub1 = sub1 - np.mean(sub1)
    sub2 = sub2 - np.mean(sub2)
    sub3 = sub3 - np.mean(sub3)

    if debug_level >= 1:
        print(f"Subregion length: {len(sub3)}, Expected samples: {int(combined_duration * fs)}")
        print(f"Subregion t_sub range: min={t_sub3[0]*1e6:.2f}µs, max={t_sub3[-1]*1e6:.2f}µs")
        print(f"Subregion amplitude range: min={np.min(np.abs(sub3)):.4f}, max={np.max(np.abs(sub3)):.4f}")

    transform = Transform1d()
    nlevels = 5
    target_length1 = 2 ** int(np.ceil(np.log2(len(sub1))))
    sub1_padded = np.pad(sub1, (0, target_length1 - len(sub1)), mode="reflect")
    coeffs1 = transform.forward(sub1_padded, nlevels=nlevels)
    coeffs1 = [np.array(c) for c in coeffs1.highpasses]
    target_length2 = 2 ** int(np.ceil(np.log2(len(sub2))))
    sub2_padded = np.pad(sub2, (0, target_length2 - len(sub2)), mode="reflect")
    coeffs2 = transform.forward(sub2_padded, nlevels=nlevels)
    coeffs2 = [np.array(c) for c in coeffs2.highpasses]
    target_length3 = 2 ** int(np.ceil(np.log2(len(sub3))))
    sub3_padded = np.pad(sub3, (0, target_length3 - len(sub3)), mode="reflect")
    coeffs3 = transform.forward(sub3_padded, nlevels=nlevels)
    coeffs3 = [np.array(c) for c in coeffs3.highpasses]

    if debug_level >= 2:
        print(f"coeffs1 type: {type(coeffs1)}, len: {len(coeffs1)}")
        for i, coeff in enumerate(coeffs1):
            print(f"coeffs1[{i}] type: {type(coeff)}, shape: {coeff.shape}")

    chars1 = compute_characteristics(coeffs1, fs, nlevels=5, debug_level=debug_level)
    chars2 = compute_characteristics(coeffs2, fs, nlevels=5, debug_level=debug_level)
    chars3 = compute_characteristics(coeffs3, fs, nlevels=5, debug_level=debug_level)

    if debug_level >= 2:
        print(f"chars1 type: {type(chars1)}, len: {len(chars1)}")
        for i, char in enumerate(chars1):
            print(f"chars1[{i}] type: {type(char)}")

    # Fix: Remove the extra list layer by calling center_characteristics directly
    centered1 = [center_characteristics(c) for c in chars1]
    centered2 = [center_characteristics(c) for c in chars2]
    centered3 = [center_characteristics(c) for c in chars3]

    stats1 = compute_statistics(centered1, debug_level=debug_level)
    stats2 = compute_statistics(centered2, debug_level=debug_level)
    stats3 = compute_statistics(centered3, debug_level=debug_level)

    feature_vector = np.array(stats1 + stats2 + stats3).flatten()

    centered_chars = {
        'sub1': centered1,
        'sub2': centered2,
        'sub3': centered3
    }
    return feature_vector, centered_chars