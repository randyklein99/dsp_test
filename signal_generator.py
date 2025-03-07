# signal_generator.py
import numpy as np

def generate_80211_stf_signal(fs=20e6, noise_duration=10e-6, stf_duration=20e-6, noise_std=0.1):
    """
    Generate an 802.11-like signal with a noise-only period and STF preamble.

    Args:
        fs (float): Sampling frequency (Hz).
        noise_duration (float): Duration of noise-only period (seconds).
        stf_duration (float): Duration of STF + buffer period (seconds).
        noise_std (float): Standard deviation of Gaussian noise.

    Returns:
        tuple: (t, input_signal, signal) where t is time array, input_signal is noisy signal,
               and signal is the clean STF signal.
    """
    total_duration = noise_duration + stf_duration
    n_samples = int(fs * total_duration)
    if n_samples % 2 != 0:  # Ensure even length for DTCWT
        n_samples += 1
    t = np.linspace(0, total_duration, n_samples, endpoint=False)

    # STF: 10 symbols, each 0.8 µs
    stf_symbol_duration = 0.8e-6
    stf_freq = 1.25e6
    stf_samples_per_symbol = int(fs * stf_symbol_duration)
    stf_symbol = np.sin(2 * np.pi * stf_freq * np.linspace(0, stf_symbol_duration, stf_samples_per_symbol, endpoint=False))
    stf = np.tile(stf_symbol, 10)  # 10 repeats, 160 samples
    signal = np.zeros(n_samples)
    start_idx = int(noise_duration * fs)  # STF starts at 10 µs
    signal[start_idx:start_idx + len(stf)] = stf
    input_signal = signal + noise_std * np.random.randn(n_samples)

    return t, input_signal, signal