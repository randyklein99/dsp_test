# tests/test_feature_extractor.py
import unittest
import numpy as np
from signal_generator import generate_80211ag_preamble
from feature_extractor import compute_characteristics, center_characteristics, compute_statistics, extract_features
from dtcwt import Transform1d

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.fs = 20e6
        self.nlevels = 5
        self.t, self.signal = generate_80211ag_preamble(fs=self.fs, add_rf_fingerprint=True, seed=42, debug_level=0)
        transform = Transform1d()
        target_length = 2 ** int(np.ceil(np.log2(len(self.signal))))
        signal_padded = np.pad(self.signal, (0, target_length - len(self.signal)), mode="reflect")
        coeffs_obj = transform.forward(signal_padded, nlevels=self.nlevels)
        self.coeffs = [np.array(c) for c in coeffs_obj.highpasses]

    def test_compute_characteristics(self):
        """Test compute_characteristics function."""
        chars = compute_characteristics(self.coeffs, self.fs, nlevels=self.nlevels)
        self.assertEqual(len(chars), self.nlevels, "Should return one dict per level")
        for char in chars:
            self.assertIn("amplitude", char)
            self.assertIn("phase_unwrapped", char)
            self.assertIn("frequency", char)
            self.assertIn("t", char)
            self.assertGreater(len(char["amplitude"]), 0, "Amplitude should be non-empty")
            self.assertTrue(np.all(np.isfinite(char["amplitude"])), "Amplitude should be finite")
            self.assertTrue(np.all(np.isfinite(char["phase_unwrapped"])), "Phase should be finite")
            self.assertTrue(np.all(np.isfinite(char["frequency"])), "Frequency should be finite")

    def test_center_characteristics(self):
        """Test center_characteristics function."""
        chars = compute_characteristics(self.coeffs, self.fs, nlevels=self.nlevels)
        centered = [center_characteristics(char) for char in chars]
        self.assertEqual(len(centered), self.nlevels, "Should match number of levels")
        for i, c in enumerate(centered):
            self.assertAlmostEqual(np.mean(c["amplitude"]), 0, delta=1e-10, msg="Amplitude not centered")
            self.assertAlmostEqual(np.mean(c["phase"]), 0, delta=1e-10, msg="Phase not centered")
            self.assertEqual(len(c["frequency"]), len(chars[i]["phase_unwrapped"]), "Frequency length should match phase length")
            self.assertTrue(np.all(np.isfinite(c["amplitude"])), "Centered amplitude should be finite")

    def test_compute_statistics(self):
        """Test compute_statistics function."""
        chars = compute_characteristics(self.coeffs, self.fs, nlevels=self.nlevels)
        centered = [center_characteristics(c) for c in chars]
        stats = compute_statistics(centered)
        self.assertEqual(len(stats), self.nlevels, "Should have stats for each level")
        self.assertEqual(len(stats[0]), 9, "Should have 9 stats per level")
        self.assertTrue(np.all(np.isfinite(stats)), "Statistics should be finite")

    def test_extract_features(self):
        """Test extract_features function."""
        feature_vector, centered_chars = extract_features(self.signal, self.t, self.fs, preamble_start=0)
        self.assertEqual(len(feature_vector), 135, "Feature vector should have 135 elements")
        self.assertTrue(np.all(np.isfinite(feature_vector)), "Feature vector should be finite")
        self.assertIn("sub1", centered_chars)
        self.assertIn("sub2", centered_chars)
        self.assertIn("sub3", centered_chars)
        self.assertEqual(len(centered_chars["sub1"]), self.nlevels, "Sub1 should have 5 levels")
        self.assertEqual(len(centered_chars["sub2"]), self.nlevels, "Sub2 should have 5 levels")
        self.assertEqual(len(centered_chars["sub3"]), self.nlevels, "Sub3 should have 5 levels")
        for sub in ["sub1", "sub2", "sub3"]:
            for level in centered_chars[sub]:
                self.assertIn("amplitude", level)
                self.assertIn("phase", level)
                self.assertIn("frequency", level)
                self.assertIn("t", level)

    def test_extract_features_short_signal(self):
        """Test extract_features with a short signal meeting minimum length."""
        short_t = np.linspace(0, 16e-6, 320)  # 16 µs signal, 320 samples meets sub1 (160) and sub3 (320) requirements
        short_signal = np.random.randn(320) + 1j * np.random.randn(320)
        feature_vector, centered_chars = extract_features(short_signal, short_t, self.fs, preamble_start=0)
        self.assertEqual(len(feature_vector), 135, "Feature vector should still be 135 elements")
        self.assertTrue(np.all(np.isfinite(feature_vector)), "Feature vector should be finite")
        self.assertEqual(len(centered_chars["sub3"]), self.nlevels, "Sub3 should have 5 levels")

    def test_extract_features_invalid_input(self):
        """Test extract_features with invalid inputs."""
        with self.assertRaises(ValueError):
            extract_features(np.array([]), self.t, self.fs)  # Empty signal
        with self.assertRaises(ValueError):
            extract_features(self.signal, np.array([]), self.fs)  # Empty time array
        with self.assertRaises(ValueError):
            extract_features(self.signal, self.t, fs=0)  # Invalid fs
        with self.assertRaises(ValueError):
            extract_features(np.array([np.inf]), np.array([0]), self.fs)  # Non-finite signal

if __name__ == "__main__":
    unittest.main()