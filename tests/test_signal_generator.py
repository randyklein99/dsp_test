# tests/test_signal_generator.py
import unittest
import numpy as np
from signal_generator import generate_80211ag_preamble

class TestSignalGenerator(unittest.TestCase):
    def setUp(self):
        self.fs = 20e6
        self.t, self.signal = generate_80211ag_preamble(
            fs=self.fs, add_rf_fingerprint=True, seed=42, debug_level=0
        )
        self.preamble_duration = 16e-6
        self.expected_length = int(self.preamble_duration * self.fs)

    def test_signal_length(self):
        """Test that the signal length matches the expected preamble duration."""
        self.assertEqual(len(self.signal), self.expected_length, "Signal length incorrect")
        self.assertEqual(len(self.t), self.expected_length, "Time array length incorrect")

    def test_stf_ltf_structure(self):
        """Test that STF (0-8 µs) and LTF (8-16 µs) have non-zero power."""
        stf_end_idx = int(8e-6 * self.fs)
        ltf_end_idx = self.expected_length
        stf_power = np.mean(np.abs(self.signal[:stf_end_idx])**2)
        ltf_power = np.mean(np.abs(self.signal[stf_end_idx:ltf_end_idx])**2)
        self.assertGreater(stf_power, 0, "STF should have non-zero power")
        self.assertGreater(ltf_power, 0, "LTF should have non-zero power")

    def test_rf_fingerprint(self):
        """Test that RF fingerprint adds phase noise and frequency offset."""
        signal_no_rf = generate_80211ag_preamble(fs=self.fs, add_rf_fingerprint=False, seed=42, debug_level=0)[1]
        phase_diff = np.angle(self.signal / signal_no_rf) * 180 / np.pi  # Degrees
        phase_std = np.std(phase_diff)
        self.assertGreater(phase_std, 0, "RF fingerprint should add phase variation")
        self.assertLess(phase_std, 40, "Phase noise should be within ±40° range (adjusted for AWGN)")
        # Basic frequency offset check (non-zero difference in phase slope)
        phase_slope_diff = np.abs(np.diff(np.unwrap(np.angle(self.signal))) - np.diff(np.unwrap(np.angle(signal_no_rf))))
        self.assertGreater(np.mean(phase_slope_diff), 0, "RF fingerprint should add frequency offset")

    def test_invalid_fs(self):
        """Test signal generator with invalid sampling frequency."""
        with self.assertRaises(ValueError):
            generate_80211ag_preamble(fs=0, add_rf_fingerprint=False, debug_level=0)
        with self.assertRaises(ValueError):
            generate_80211ag_preamble(fs=-1, add_rf_fingerprint=False, debug_level=0)

if __name__ == "__main__":
    unittest.main()