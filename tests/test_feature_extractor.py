# tests/test_feature_extractor.py
import unittest
import numpy as np
from signal_generator import generate_80211ag_preamble
from feature_extractor import (
    compute_characteristics,
    center_characteristics,
    compute_statistics,
    extract_features,
)
from dtcwt_utils import run_dtcwt


class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        # Set up test parameters
        self.fs = 20e6
        self.nlevels = 5
        self.t_start = 0
        self.t_end = 16e-6
        self.venv_path = "/home/randy/code/test/.venv_dtcwt"
        self.t, self.signal = generate_80211ag_preamble(
            fs=self.fs, add_rf_fingerprint=True, seed=42
        )
        self.idx_combined = (self.t >= self.t_start) & (self.t < self.t_end)
        self.subregion = self.signal[self.idx_combined]
        self.t_sub = self.t[self.idx_combined]
        self.coeffs = run_dtcwt(
            self.subregion, self.t_sub, nlevels=self.nlevels, venv_path=self.venv_path
        )

    def test_compute_characteristics(self):
        """Test compute_characteristics function."""
        chars = compute_characteristics(self.coeffs, self.fs, nlevels=self.nlevels)
        self.assertEqual(len(chars), self.nlevels)
        for char in chars:
            self.assertIn("amplitude", char)
            self.assertIn("phase_unwrapped", char)
            self.assertIn("frequency", char)
            self.assertIn("t", char)
            self.assertTrue(len(char["amplitude"]) > 0)
            self.assertTrue(len(char["phase_unwrapped"]) > 0)
            self.assertTrue(len(char["frequency"]) >= 0)  # Can be empty if length < 2

    def test_center_characteristics(self):
        """Test center_characteristics function."""
        chars = compute_characteristics(self.coeffs, self.fs, nlevels=self.nlevels)
        centered = center_characteristics(chars)
        self.assertEqual(len(centered), len(chars))
        for centered_char in centered:
            self.assertIn("amplitude", centered_char)
            self.assertIn("phase", centered_char)
            self.assertIn("frequency", centered_char)
            self.assertIn("t", centered_char)

    def test_compute_statistics(self):
        """Test compute_statistics function."""
        chars = compute_characteristics(self.coeffs, self.fs, nlevels=self.nlevels)
        centered = center_characteristics(chars)
        stats = compute_statistics(centered)
        self.assertEqual(len(stats), self.nlevels)
        for stat in stats:
            self.assertEqual(
                len(stat), 9
            )  # 3 stats per characteristic (var, skew, kurt)
            self.assertTrue(np.all(np.isfinite(stat)), "Statistics should be finite")

    def test_extract_features(self):
        """Test extract_features function."""
        feature_vector = extract_features(
            self.signal, self.t, self.fs, preamble_start=0, venv_path=self.venv_path
        )
        self.assertEqual(
            len(feature_vector), 27
        )  # 3 subregions x 1 level (3) x 3 characteristics x 3 stats
        self.assertTrue(
            np.all(np.isfinite(feature_vector)),
            "Feature vector should contain only finite values",
        )

    def test_signal_generator(self):
        """Test signal_generator function."""
        t, signal = generate_80211ag_preamble(
            fs=self.fs, add_rf_fingerprint=True, seed=42
        )
        self.assertEqual(len(t), len(signal))
        self.assertTrue(np.all(np.isfinite(signal)))
        self.assertGreater(len(signal), 0)


if __name__ == "__main__":
    unittest.main()
