import numpy as np
from dtcwt import Transform1d  # Explicit import
import dtcwt  # Test import to verify module availability

# Test the import to raise an error if dtcwt is not available
if not hasattr(dtcwt, 'Transform1d'):
    raise ImportError("dtcwt module or Transform1d class not found. Ensure 'dtcwt' is installed.")

def denoise_signal(
    signal: np.ndarray,
    t: np.ndarray,
    nlevels: int = 4,
    threshold_multiplier: float = 0.5,
    debug_level: int = 0
) -> np.ndarray:
    """
    Denoise a signal using Dual-Tree Complex Wavelet Transform (DTCWT).

    Args:
        signal (np.ndarray): Input signal array (complex-valued).
        t (np.ndarray): Time vector corresponding to the signal.
        nlevels (int): Number of DTCWT decomposition levels (default: 4).
        threshold_multiplier (float): Multiplier for variance threshold (default: 0.5).
        debug_level (int): Debugging level (0 = no debug, 1 = basic, 2 = detailed).

    Returns:
        np.ndarray: Denoised signal magnitude, centered by noise mean.
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
    
    return denoised_mag