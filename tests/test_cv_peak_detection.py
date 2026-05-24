import importlib.util
import unittest
from pathlib import Path

import numpy as np


APP_PATH = Path(__file__).resolve().parents[1] / "MVP 1" / "voltscope_ai_mvp.py"
SPEC = importlib.util.spec_from_file_location("voltscope_ai_mvp", APP_PATH)
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


def clean_cv_standard_columns_trace() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 01_clean_cv_standard_columns.csv regression trace."""
    forward = np.linspace(-0.2, 0.8, 500)
    reverse = np.linspace(0.8, -0.2, 500)
    potential = np.concatenate([forward, reverse])
    noise = 0.12e-6 * np.sin(np.arange(potential.size) * 1.7)

    forward_current = (
        3.2e-6
        + 0.2e-6 * forward
        + 23.6e-6 * np.exp(-0.5 * ((forward - 0.3852) / 0.045) ** 2)
    )
    reverse_current = (
        -4.5e-6
        + 0.1e-6 * reverse
        - 18.8e-6 * np.exp(-0.5 * ((reverse - 0.295) / 0.045) ** 2)
    )
    current = np.concatenate([forward_current, reverse_current]) + noise
    return potential, current


class CVPeakDetectionRegressionTest(unittest.TestCase):
    def test_clean_cv_primary_peaks_are_correct_and_not_low_confidence(self) -> None:
        potential, current = clean_cv_standard_columns_trace()
        params = APP.auto_tune_cv_peak_parameters(potential, current)
        peaks = APP.detect_peaks(
            potential,
            current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
        )

        oxidation_peaks = [peak for peak in peaks if peak.peak_type == "oxidation"]
        reduction_peaks = [peak for peak in peaks if peak.peak_type == "reduction"]
        selected_oxidation = oxidation_peaks[APP.default_primary_peak_index(oxidation_peaks, "oxidation")]
        selected_reduction = reduction_peaks[APP.default_primary_peak_index(reduction_peaks, "reduction")]

        self.assertAlmostEqual(selected_oxidation.potential, 0.3852, delta=0.01)
        self.assertAlmostEqual(selected_oxidation.raw_current * 1e6, 26.8, delta=1.0)
        self.assertAlmostEqual(selected_reduction.potential, 0.295, delta=0.01)
        self.assertAlmostEqual(selected_reduction.raw_current * 1e6, -23.28, delta=1.0)
        self.assertNotEqual(selected_oxidation.confidence, "low")
        self.assertNotEqual(selected_reduction.confidence, "low")
        self.assertIn(selected_oxidation.confidence, {"medium-high", "high"})
        self.assertIn(selected_reduction.confidence, {"medium-high", "high"})

        _rows, metrics = APP.build_peak_metrics_rows(
            selected_oxidation,
            selected_reduction,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        self.assertEqual(APP.cv_analysis_quality_messages(selected_oxidation, selected_reduction, metrics, potential), [])


if __name__ == "__main__":
    unittest.main()
