# signal_generator.py
import numpy as np


def generate_80211ag_preamble(fs=20e6, add_rf_fingerprint=False, seed=None):
    """
    Generate an 802.11a/g preamble with short and long training sequences, optionally with RF fingerprint simulation.

    Args:
        fs (float): Sampling frequency in Hz (default: 20e6).
        add_rf_fingerprint (bool): Whether to apply RF fingerprint simulation (default: False).
        seed (int or None): Random seed for reproducibility of RF fingerprint (default: None).

    Returns:
        tuple: (t, preamble) where t is the time array and preamble is the signal.
    """
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    fft_size = 64
    # Scale samples per symbol based on sampling frequency (3.2 µs is standard long symbol duration)
    samples_per_symbol = int(fs * 3.2e-6)  # 64 samples at 20 MHz, scales with fs
    cp_long = int(fs * 1.6e-6)  # Cyclic prefix = 1.6 µs, scales with fs

    # Short training sequence: subcarriers -24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24
    short_subcarriers = np.zeros(fft_size, dtype=complex)
    short_indices = [-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]
    short_values = np.sqrt(13 / 6) * np.array(
        [
            1 + 1j,
            -1 - 1j,
            1 + 1j,
            -1 - 1j,
            1 + 1j,
            1 + 1j,
            1 + 1j,
            -1 - 1j,
            -1 - 1j,
            1 + 1j,
            -1 - 1j,
            1 + 1j,
        ]
    )
    for k, v in zip(short_indices, short_values):
        idx = k if k >= 0 else fft_size + k
        short_subcarriers[idx] = v
    short_symbol = np.fft.ifft(short_subcarriers) * np.sqrt(fft_size)
    # Use 16 samples per short symbol, repeated 10 times (total duration 8 µs)
    short_samples_per_symbol = int(fs * 0.8e-6)  # 0.8 µs per short symbol
    short_sequence = np.tile(short_symbol[:short_samples_per_symbol], 10)

    # Long training sequence: all 52 subcarriers (-26 to -1, 1 to 26)
    long_subcarriers = np.zeros(fft_size, dtype=complex)
    long_indices = list(range(-26, 0)) + list(range(1, 27))
    long_values = np.array(
        [
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
        ],
        dtype=complex,
    )
    # Normalize power to match STF: STF has 12 subcarriers with power 13/6, LTF has 52
    power_scale = np.sqrt((12 * (13 / 6)) / 52)  # sqrt((12 * (13/6)) / 52) ≈ 0.707
    long_values *= power_scale
    for k, v in zip(long_indices, long_values):
        idx = k if k >= 0 else fft_size + k
        long_subcarriers[idx] = v
    long_symbol = np.fft.ifft(long_subcarriers) * np.sqrt(fft_size)
    cp = long_symbol[-cp_long:]  # Cyclic prefix: last 1.6 µs samples
    long_sequence = np.concatenate(
        [cp, long_symbol, long_symbol]
    )  # 2 long symbols + CP

    # Combine preamble
    preamble = np.concatenate([short_sequence, long_sequence])
    # Ensure total duration is approximately 16 µs (adjust if needed)
    total_samples = int(16e-6 * fs)
    if len(preamble) < total_samples:
        padding = np.zeros(total_samples - len(preamble), dtype=complex)
        preamble = np.concatenate([preamble, padding])
    elif len(preamble) > total_samples:
        preamble = preamble[:total_samples]

    # Apply RF fingerprint simulation if requested
    if add_rf_fingerprint:
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        # Phase noise: random phase variation, scaled to ±25 degrees
        phase_noise = np.random.randn(len(preamble)) * np.deg2rad(25)
        print(f"Phase noise std (degrees): {np.std(phase_noise) * 180/np.pi:.2f}")
        preamble = preamble * np.exp(1j * phase_noise)
        # Frequency offset: small random offset, ±50 Hz
        freq_offset = (
            np.random.uniform(-50, 50)
            if seed is None
            else np.random.uniform(-50, 50, size=1)[0]
        )
        print(f"Applied frequency offset: {freq_offset:.2f} Hz")
        t = np.arange(len(preamble)) / fs
        freq_shift = np.exp(1j * 2 * np.pi * freq_offset * t)
        preamble = preamble * freq_shift

    # Add AWGN (distinct from RF fingerprint), reduced noise level
    noise = (
        np.random.randn(len(preamble)) + 1j * np.random.randn(len(preamble))
    ) * 0.05
    preamble += noise
    t = np.arange(0, len(preamble) / fs, 1 / fs)
    return t, preamble