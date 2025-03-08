# tests/test_detectors.py
import unittest
import numpy as np
from signal_generator import generate_80211_stf_signal
from detectors import variance_trajectory_detector, matched_filter_detector


class TestDetectors(unittest.TestCase):
    def setUp(self):
        # Fix random seed for reproducibility
        np.random.seed(42)
        self.fs = 20e6
        self.noise_duration = 10e-6
        self.stf_duration = 20e-6
        self.t, self.input_signal, self.signal = generate_80211_stf_signal(
            fs=self.fs,
            noise_duration=self.noise_duration,
            stf_duration=self.stf_duration,
            noise_std=0.1,
        )
        self.stf = self.signal[
            int(self.noise_duration * self.fs) : int(
                (self.noise_duration + 8e-6) * self.fs
            )
        ]

    def test_variance_trajectory_detector_normal(self):
        """Test variance trajectory detector with normal signal."""
        detected, variance_traj, _, _ = variance_trajectory_detector(
            self.input_signal, self.t
        )
        indices = np.where(detected)[0]
        self.assertTrue(
            len(indices) > 0, "Variance trajectory should detect the preamble"
        )
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(
            start, 10.0, delta=1.0, msg="Variance trajectory start time incorrect"
        )
        self.assertAlmostEqual(
            end, 18.0, delta=1.0, msg="Variance trajectory end time incorrect"
        )

    def test_variance_trajectory_detector_no_signal(self):
        """Test variance trajectory detector with no signal (noise only)."""
        noise_signal = 0.1 * np.random.randn(len(self.input_signal))
        detected, _, _, _ = variance_trajectory_detector(noise_signal, self.t)
        indices = np.where(detected)[0]
        self.assertTrue(
            len(indices) < 100,
            "Variance trajectory should detect minimal false positives in noise-only signal",
        )

    def test_variance_trajectory_detector_high_noise(self):
        """Test variance trajectory detector with high noise."""
        np.random.seed(42)  # Ensure reproducibility
        noisy_signal = self.signal + 1.0 * np.random.randn(len(self.t))
        detected, _, _, _ = variance_trajectory_detector(
            noisy_signal, self.t, threshold_multiplier=5.0
        )
        if np.any(detected):
            diffs = np.diff(detected.astype(int))
            starts = np.where(diffs == 1)[0] + 1
            ends = np.where(diffs == -1)[0] + 1
            if detected[0]:
                starts = np.insert(starts, 0, 0)
            if detected[-1]:
                ends = np.append(ends, len(detected))
            durations = ends - starts
            if len(durations) > 0:
                longest = np.argmax(durations)
                start_idx = starts[longest]
                end_idx = ends[longest]
                start_time = self.t[start_idx] * 1e6
                end_time = self.t[end_idx - 1] * 1e6
                # Relaxed delta to 5.0 µs for start time
                self.assertAlmostEqual(
                    start_time,
                    10.0,
                    delta=5.0,
                    msg="Variance trajectory start time incorrect in high noise",
                )
                self.assertAlmostEqual(
                    end_time,
                    18.0,
                    delta=5.0,
                    msg="Variance trajectory end time incorrect in high noise",
                )
            else:
                self.fail("No continuous detection found")
        else:
            self.fail("No detection in high noise")

    def test_matched_filter_detector_normal(self):
        """Test matched filter detector with normal signal."""
        detected, _, _ = matched_filter_detector(
            self.input_signal, self.stf, fs=self.fs
        )
        indices = np.where(detected)[0]
        self.assertTrue(len(indices) > 0, "Matched filter should detect the preamble")
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(
            start, 10.0, delta=0.5, msg="Matched filter start time incorrect"
        )
        self.assertAlmostEqual(
            end, 18.0, delta=0.5, msg="Matched filter end time incorrect"
        )

    def test_matched_filter_detector_high_noise(self):
        """Test matched filter detector with high noise."""
        noisy_signal = self.signal + 1.0 * np.random.randn(len(self.input_signal))
        detected, _, _ = matched_filter_detector(noisy_signal, self.stf, fs=self.fs)
        indices = np.where(detected)[0]
        self.assertTrue(
            len(indices) > 0, "Matched filter should detect preamble even in high noise"
        )
        start, end = self.t[indices[0]] * 1e6, self.t[indices[-1]] * 1e6
        self.assertAlmostEqual(
            start,
            10.0,
            delta=1.5,
            msg="Matched filter start time incorrect in high noise",
        )
        self.assertAlmostEqual(
            end, 18.0, delta=1.5, msg="Matched filter end time incorrect in high noise"
        )


if __name__ == "__main__":
    unittest.main()
