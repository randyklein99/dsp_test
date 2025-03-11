# tests/test_detectors.py
import unittest
import numpy as np
from signal_generator import generate_80211ag_preamble
from detectors import variance_trajectory_detector, matched_filter_detector

class TestDetectors(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.fs = 20e6
        self.t, self.input_signal = generate_80211ag_preamble(
            fs=self.fs, add_rf_fingerprint=True, seed=42
        )
        self.preamble_duration = 16e-6  # STF (8 µs) + LTF (8 µs)
        self.stf_duration = 8e-6

    def test_variance_trajectory_detector_normal(self):
        """Test variance trajectory detector with normal signal."""
        detected, variance_traj, threshold_var, enhanced_mag = variance_trajectory_detector(
            self.input_signal, self.t
        )
        indices = np.where(detected)[0]
        self.assertTrue(len(indices) > 0, "Should detect preamble")
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(start, 0.0, delta=1.0, msg="Start time incorrect")
        self.assertAlmostEqual(end, self.preamble_duration * 1e6, delta=1.0, msg="End time incorrect")
        self.assertTrue(np.all(np.isfinite(variance_traj)), "Variance trajectory should be finite")
        self.assertGreater(threshold_var, 0, "Threshold should be positive")

    def test_variance_trajectory_detector_no_signal(self):
        """Test variance trajectory detector with noise only."""
        noise_signal = 0.1 * np.random.randn(len(self.input_signal))
        detected, _, _, _ = variance_trajectory_detector(noise_signal, self.t)
        indices = np.where(detected)[0]
        self.assertLess(len(indices), 100, "Should detect minimal false positives in noise")

    def test_variance_trajectory_detector_high_noise(self):
        """Test variance trajectory detector with high noise."""
        noisy_signal = self.input_signal + np.random.randn(len(self.input_signal))
        detected, _, _, _ = variance_trajectory_detector(noisy_signal, self.t, threshold_multiplier=5.0)
        self.assertTrue(np.any(detected), "Should detect preamble in high noise")
        indices = np.where(detected)[0]
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(start, 0.0, delta=5.0, msg="Start time incorrect")
        self.assertAlmostEqual(end, self.preamble_duration * 1e6, delta=5.0, msg="End time incorrect")

    def test_matched_filter_detector_normal(self):
        """Test matched filter detector with normal signal."""
        stf_template = self.input_signal[:int(self.stf_duration * self.fs)]
        detected, mf_output, threshold_mf = matched_filter_detector(
            self.input_signal, stf_template, fs=self.fs
        )
        indices = np.where(detected)[0]
        self.assertTrue(len(indices) > 0, "Should detect preamble")
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(start, 0.0, delta=0.5, msg="Start time incorrect")
        self.assertAlmostEqual(end, self.stf_duration * 1e6, delta=1.5, msg="End time incorrect")
        self.assertTrue(np.all(np.isfinite(mf_output)), "MF output should be finite")

    def test_matched_filter_detector_high_noise(self):
        """Test matched filter detector with high noise."""
        noisy_signal = self.input_signal + np.random.randn(len(self.input_signal))
        stf_template = self.input_signal[:int(self.stf_duration * self.fs)]
        detected, _, _ = matched_filter_detector(noisy_signal, stf_template, fs=self.fs)
        indices = np.where(detected)[0]
        self.assertTrue(len(indices) > 0, "Should detect preamble in high noise")
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(start, 0.0, delta=1.5, msg="Start time incorrect")
        self.assertAlmostEqual(end, self.stf_duration * 1e6, delta=2.5, msg="End time incorrect")

    def test_variance_trajectory_detector_invalid_input(self):
        """Test variance trajectory detector with invalid input."""
        with self.assertRaises(ValueError):
            variance_trajectory_detector(np.array([]), self.t)  # Empty signal
        with self.assertRaises(ValueError):
            variance_trajectory_detector(self.input_signal, np.array([]))  # Empty time array

if __name__ == "__main__":
    unittest.main()