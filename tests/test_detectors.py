# tests/test_detectors.py
import unittest
import numpy as np
from signal_generator import generate_80211ag_preamble
from detectors import variance_trajectory_detector, matched_filter_detector, analyze_noise
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
        self.preamble_duration = 16e-6
        self.stf_duration = 8e-6
        self.window_size = 320
        self.denoised_signal = denoise_signal(self.input_signal, self.t)

    def test_variance_trajectory_detector_normal(self):
        """Test variance trajectory detector with normal signal."""
        # Test with raw signal
        noise_stats_raw = analyze_noise(self.input_signal, self.t, window_size=self.window_size, noise_region=(0, self.noise_duration))
        threshold_raw = noise_stats_raw['percentile_95_vt'] * 1.5
        variance_traj_raw, _, _, trigger_idx_raw = variance_trajectory_detector(
            self.input_signal, self.t, window_size=self.window_size, threshold=threshold_raw
        )
        self.assertGreaterEqual(trigger_idx_raw, 0, "Should detect a trigger point with raw signal")
        self.assertLess(trigger_idx_raw, len(variance_traj_raw), "Trigger index should be within bounds")
        self.assertTrue(np.all(np.isfinite(variance_traj_raw)), "Raw variance trajectory should be finite")

        # Test with denoised signal
        noise_stats_denoised = analyze_noise(self.denoised_signal, self.t, window_size=self.window_size, noise_region=(0, self.noise_duration))
        threshold_denoised = noise_stats_denoised['percentile_95_vt'] * 1.5
        variance_traj_denoised, _, _, trigger_idx_denoised = variance_trajectory_detector(
            self.denoised_signal, self.t, window_size=self.window_size, threshold=threshold_denoised
        )
        self.assertGreaterEqual(trigger_idx_denoised, 0, "Should detect a trigger point with denoising")
        self.assertLess(trigger_idx_denoised, len(variance_traj_denoised), "Trigger index should be within bounds")
        self.assertTrue(np.all(np.isfinite(variance_traj_denoised)), "Denoised variance trajectory should be finite")

    def test_variance_trajectory_detector_no_signal(self):
        """Test variance trajectory detector with noise only."""
        noise_signal = 0.05 * np.random.randn(len(self.input_signal))
        noise_stats = analyze_noise(noise_signal, self.t, window_size=self.window_size)
        threshold = noise_stats['percentile_95_vt'] * 10.0
        _, _, _, trigger_idx = variance_trajectory_detector(
            noise_signal, self.t, window_size=self.window_size, threshold=threshold
        )
        self.assertEqual(trigger_idx, -1, "Should not detect a trigger in noise-only signal")

    def test_variance_trajectory_detector_high_noise(self):
        """Test variance trajectory detector with high noise (SNR >= 10 dB)."""
        noisy_signal = self.input_signal + 0.2 * np.random.randn(len(self.input_signal))  # Scaled for ~10 dB SNR
        noise_stats = analyze_noise(noisy_signal, self.t, window_size=self.window_size, noise_region=(0, self.noise_duration))
        threshold = noise_stats['percentile_95_vt'] * 1.5
        variance_traj, _, _, trigger_idx = variance_trajectory_detector(
            noisy_signal, self.t, window_size=self.window_size, threshold=threshold
        )
        # Debug output (uncomment if needed)
        # t_var = self.t[self.window_size - 1: len(variance_traj) + self.window_size - 1]
        # preamble_mask = (t_var >= 20e-6) & (t_var <= 36e-6)
        # print(f"Noisy 75th VT: {noise_stats['percentile_75_vt']:.6f}, Threshold: {threshold:.6f}")
        # print(f"Noisy max preamble VT: {np.max(variance_traj[preamble_mask]):.6f}, Trigger idx: {trigger_idx}")
        self.assertGreaterEqual(trigger_idx, 0, "Should detect a trigger point in high noise")
        self.assertLess(trigger_idx, len(variance_traj), "Trigger index should be within bounds")

    def test_variance_trajectory_detector_short_signal(self):
        """Test variance trajectory detector with a short signal."""
        short_t = np.linspace(0, 10e-6, 200)
        short_signal = np.random.randn(200) + 1j * np.random.randn(200)
        noise_stats = analyze_noise(short_signal, short_t, window_size=50)
        threshold = noise_stats['percentile_95_vt'] * 2.0
        variance_traj, _, _, trigger_idx = variance_trajectory_detector(
            short_signal, short_t, window_size=50, threshold=threshold
        )
        self.assertTrue(np.all(np.isfinite(variance_traj)), "Variance trajectory should be finite")
        self.assertGreaterEqual(len(variance_traj), 0, "Variance trajectory should be computed")

    def test_analyze_noise(self):
        """Test noise analysis function."""
        stats = analyze_noise(self.input_signal, self.t, window_size=self.window_size, noise_region=(0, 20e-6))
        self.assertIn('mean_vt', stats)
        self.assertIn('max_vt', stats)
        self.assertIn('std_vt', stats)
        self.assertIn('percentile_95_vt', stats)
        self.assertGreaterEqual(stats['mean_vt'], 0)
        self.assertGreaterEqual(stats['max_vt'], 0)
        self.assertGreaterEqual(stats['std_vt'], 0)
        self.assertGreaterEqual(stats['percentile_95_vt'], 0)

        stats_full = analyze_noise(self.input_signal, self.t, window_size=self.window_size)
        self.assertGreater(stats_full['max_vt'], stats['max_vt'], "Full signal should have higher max VT due to preamble")

        short_t = np.linspace(0, 10e-6, 100)
        short_signal = np.random.randn(100) + 1j * np.random.randn(100)
        stats_short = analyze_noise(short_signal, short_t, window_size=50)
        self.assertGreaterEqual(stats_short['mean_vt'], 0)

    def test_matched_filter_detector_normal(self):
        """Test matched filter detector with normal signal."""
        stf_template = self.input_signal[int(self.noise_duration * self.fs):int((self.noise_duration + self.stf_duration) * self.fs)]
        detected, mf_output, threshold_mf = matched_filter_detector(
            self.input_signal, stf_template, fs=self.fs
        )
        indices = np.where(detected)[0]
        self.assertTrue(len(indices) > 0, "Should detect preamble")
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(start, (self.noise_duration - self.stf_duration/2) * 1e6, delta=0.5, msg="Start time should be near 16 µs")
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
            variance_trajectory_detector(np.array([]), self.t, window_size=self.window_size, threshold=0.01)
        with self.assertRaises(ValueError):
            variance_trajectory_detector(self.input_signal, np.array([]), window_size=self.window_size, threshold=0.01)
        with self.assertRaises(ValueError):
            variance_trajectory_detector(self.input_signal, self.t, window_size=len(self.input_signal) + 1, threshold=0.01)

if __name__ == "__main__":
    unittest.main()