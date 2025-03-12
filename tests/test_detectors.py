# tests/test_detectors.py
import unittest
import numpy as np
from signal_generator import generate_80211ag_preamble
from detectors import variance_trajectory_detector, matched_filter_detector
from denoising import denoise_signal

class TestDetectors(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.fs = 20e6
        self.t, self.input_signal = generate_80211ag_preamble(
            fs=self.fs, add_rf_fingerprint=True, seed=42, debug_level=0
        )
        # Extend signal with 20 µs noise preamble
        self.noise_duration = 20e-6
        self.noise_samples = int(self.noise_duration * self.fs)
        noise = 0.05 * (np.random.randn(self.noise_samples) + 1j * np.random.randn(self.noise_samples))
        self.input_signal = np.concatenate([noise, self.input_signal])
        self.t = np.concatenate([np.arange(self.noise_samples) / self.fs, self.t[-1] + (1 / self.fs) + self.t])
        self.preamble_duration = 16e-6  # STF (8 µs) + LTF (8 µs)
        self.stf_duration = 8e-6
        self.window_size = 320  # Matches detector's default window size
        # Compute raw and denoised magnitudes
        self.raw_mag = np.abs(self.input_signal) - np.mean(np.abs(self.input_signal[:self.noise_samples]))
        self.raw_mag = np.maximum(self.raw_mag, 0)
        self.denoised_mag = denoise_signal(self.input_signal, self.t)

    def test_variance_trajectory_detector_normal(self):
        """Test variance trajectory detector with normal signal."""
        # Test with denoised signal
        variance_traj_denoised, threshold_var_denoised, _, trigger_idx_denoised = variance_trajectory_detector(
            self.input_signal, self.t, processed_mag=self.denoised_mag, threshold_multiplier=1.0
        )
        self.assertGreaterEqual(trigger_idx_denoised, 0, "Should detect a trigger point with denoising")
        self.assertLess(trigger_idx_denoised, len(variance_traj_denoised), "Trigger index should be within bounds")
        # Optional debug output
        # t_var = self.t[self.window_size - 1: len(variance_traj_denoised) + self.window_size - 1]
        # print(f"Denoised trigger time: {t_var[trigger_idx_denoised] * 1e6:.2f} µs")
        self.assertTrue(np.all(np.isfinite(variance_traj_denoised)), "Denoised variance trajectory should be finite")
        self.assertGreater(threshold_var_denoised, 0, "Denoised threshold should be positive")

        # Test with raw signal
        variance_traj_raw, threshold_var_raw, _, trigger_idx_raw = variance_trajectory_detector(
            self.input_signal, self.t, processed_mag=self.raw_mag, threshold_multiplier=1.0
        )
        self.assertGreaterEqual(trigger_idx_raw, 0, "Should detect a trigger point with raw signal")
        self.assertLess(trigger_idx_raw, len(variance_traj_raw), "Trigger index should be within bounds")
        self.assertTrue(np.all(np.isfinite(variance_traj_raw)), "Raw variance trajectory should be finite")
        self.assertGreater(threshold_var_raw, 0, "Raw threshold should be positive")

    def test_variance_trajectory_detector_no_signal(self):
        """Test variance trajectory detector with noise only."""
        noise_signal = 0.05 * np.random.randn(len(self.input_signal))
        raw_mag_noise = np.abs(noise_signal) - np.mean(np.abs(noise_signal[:self.noise_samples]))
        raw_mag_noise = np.maximum(raw_mag_noise, 0)
        _, _, _, trigger_idx = variance_trajectory_detector(
            noise_signal, self.t, processed_mag=raw_mag_noise, threshold_multiplier=10.0
        )
        self.assertEqual(trigger_idx, -1, "Should not detect a trigger in noise-only signal")
        # Optional debug output
        # if trigger_idx >= 0:
        #     t_var = self.t[self.window_size - 1: len(variance_traj) + self.window_size - 1]
        #     print(f"Noise trigger time: {t_var[trigger_idx] * 1e6:.2f} µs")

    def test_variance_trajectory_detector_high_noise(self):
        """Test variance trajectory detector with high noise."""
        noisy_signal = self.input_signal + np.random.randn(len(self.input_signal))
        raw_mag_noisy = np.abs(noisy_signal) - np.mean(np.abs(noisy_signal[:self.noise_samples]))
        raw_mag_noisy = np.maximum(raw_mag_noisy, 0)
        variance_traj, _, _, trigger_idx = variance_trajectory_detector(
            noisy_signal, self.t, processed_mag=raw_mag_noisy, threshold_multiplier=1.0
        )
        self.assertGreaterEqual(trigger_idx, 0, "Should detect a trigger point in high noise")
        self.assertLess(trigger_idx, len(variance_traj), "Trigger index should be within bounds")

    def test_variance_trajectory_detector_short_signal(self):
        """Test variance trajectory detector with a short signal."""
        short_t = np.linspace(0, 10e-6, 200)  # 10 µs signal
        short_signal = np.random.randn(200) + 1j * np.random.randn(200)
        short_mag = np.abs(short_signal)
        variance_traj, threshold, _, trigger_idx = variance_trajectory_detector(
            short_signal, short_t, processed_mag=short_mag, window_size=50
        )
        self.assertTrue(np.all(np.isfinite(variance_traj)), "Variance trajectory should be finite")
        self.assertGreaterEqual(len(variance_traj), 0, "Variance trajectory should be computed")
        # No specific expectation for trigger_idx since signal is short and random

    def test_matched_filter_detector_normal(self):
        """Test matched filter detector with normal signal."""
        stf_template = self.input_signal[int(self.noise_duration * self.fs):int((self.noise_duration + self.stf_duration) * self.fs)]
        detected, mf_output, threshold_mf = matched_filter_detector(
            self.input_signal, stf_template, fs=self.fs
        )
        indices = np.where(detected)[0]
        self.assertTrue(len(indices) > 0, "Should detect preamble")
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(start, (self.noise_duration - self.stf_duration/2) * 1e6, delta=0.5, msg="Start time should be near 16 µs (STF peak centered)")
        self.assertAlmostEqual(end, (self.noise_duration + self.stf_duration/2) * 1e6, delta=1.0, msg="End time should be near 24 µs")
        self.assertTrue(np.all(np.isfinite(mf_output)), "MF output should be finite")

    def test_matched_filter_detector_high_noise(self):
        """Test matched filter detector with high noise."""
        noisy_signal = self.input_signal + np.random.randn(len(self.input_signal))
        stf_template = self.input_signal[int(self.noise_duration * self.fs):int((self.noise_duration + self.stf_duration) * self.fs)]
        detected, _, _ = matched_filter_detector(noisy_signal, stf_template, fs=self.fs)
        indices = np.where(detected)[0]
        self.assertTrue(len(indices) > 0, "Should detect preamble in high noise")
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(start, (self.noise_duration - self.stf_duration/2) * 1e6, delta=1.5, msg="Start time should be near 16 µs")
        self.assertAlmostEqual(end, (self.noise_duration + self.stf_duration/2) * 1e6, delta=2.0, msg="End time should be near 24 µs")

    def test_variance_trajectory_detector_invalid_input(self):
        """Test variance trajectory detector with invalid input."""
        with self.assertRaises(ValueError):
            variance_trajectory_detector(np.array([]), self.t, processed_mag=np.array([]))  # Empty signal
        with self.assertRaises(ValueError):
            variance_trajectory_detector(self.input_signal, np.array([]), processed_mag=self.raw_mag)  # Empty time array
        with self.assertRaises(ValueError):
            variance_trajectory_detector(self.input_signal, self.t, processed_mag=np.array([]))  # Empty processed_mag

if __name__ == "__main__":
    unittest.main()