# tests/test_signal_generator.py
import unittest
import numpy as np
from signal_generator import generate_80211ag_preamble


class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.fs = 20e6
        self.t, self.signal = generate_80211ag_preamble(
            fs=self.fs, add_rf_fingerprint=True, seed=42
        )

    def test_signal_length(self):
        """Test that the signal length is correct."""
        expected_length = int(16e-6 * self.fs)  # 16 µs total duration
        if expected_length % 2 != 0:
            expected_length += 1
        self.assertEqual(len(self.signal), expected_length)
        self.assertEqual(len(self.t), expected_length)

    def test_stf_ltf_structure(self):
        """Test that STF (0-8 µs) and LTF (8-16 µs) are present."""
        stf_end_idx = int(8e-6 * self.fs)  # 8 µs STF
        ltf_start_idx = stf_end_idx
        ltf_end_idx = int(16e-6 * self.fs)  # 16 µs total
        self.assertTrue(np.any(self.signal[:stf_end_idx] != 0))  # STF present
        self.assertTrue(
            np.any(self.signal[ltf_start_idx:ltf_end_idx] != 0)
        )  # LTF present

    def test_rf_fingerprint(self):
        """Test that RF fingerprint adds variation (basic check)."""
        signal_no_rf = generate_80211ag_preamble(
            fs=self.fs, add_rf_fingerprint=False, seed=42
        )[1]
        phase_diff = np.angle(self.signal / signal_no_rf)
        self.assertTrue(
            np.std(phase_diff) > 0, "RF fingerprint should add phase variation"
        )


if __name__ == "__main__":
    unittest.main()
