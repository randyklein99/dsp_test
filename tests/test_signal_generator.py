# tests/test_signal_generator.py
import unittest
import numpy as np
from signal_generator import generate_80211_stf_signal

class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.fs = 20e6
        self.noise_duration = 10e-6
        self.stf_duration = 20e-6
        self.noise_std = 0.1
        self.t, self.input_signal, self.signal = generate_80211_stf_signal(
            self.fs, self.noise_duration, self.stf_duration, self.noise_std
        )

    def test_signal_length(self):
        """Test that the signal length is correct."""
        total_duration = self.noise_duration + self.stf_duration
        expected_length = int(self.fs * total_duration)
        if expected_length % 2 != 0:
            expected_length += 1
        self.assertEqual(len(self.input_signal), expected_length)
        self.assertEqual(len(self.t), expected_length)

    def test_stf_placement(self):
        """Test that STF is placed correctly at 10 µs."""
        stf_start_idx = int(self.noise_duration * self.fs)
        stf_end_idx = stf_start_idx + int(8e-6 * self.fs)  # 8 µs STF
        self.assertTrue(np.all(self.signal[:stf_start_idx] == 0))  # Noise-only before
        self.assertTrue(np.any(self.signal[stf_start_idx:stf_end_idx] != 0))  # STF present
        self.assertTrue(np.all(self.signal[stf_end_idx:] == 0))  # Zero after STF

    def test_noise_properties(self):
        """Test that noise has expected properties (std ≈ 0.1)."""
        noise_region = self.input_signal[:int(self.noise_duration * self.fs)]
        noise_std = np.std(noise_region)
        self.assertAlmostEqual(noise_std, self.noise_std, delta=0.02)

if __name__ == "__main__":
    unittest.main()