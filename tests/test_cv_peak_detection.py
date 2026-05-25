import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


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

    def test_clean_cv_scan_rate_and_cycle_count_are_detected(self) -> None:
        potential, current = clean_cv_standard_columns_trace()
        time_s = np.arange(potential.size) * (0.002004 / 0.1)
        dataset = APP.ParsedDataset(
            filename="01_clean_cv_standard_columns.csv",
            dataframe=pd.DataFrame(
                {
                    "time_s": time_s,
                    "Potential_V": potential,
                    "Current_A": current,
                    "cycle": np.ones_like(potential),
                }
            ),
            headers=["time_s", "Potential_V", "Current_A", "cycle"],
            metadata={},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=0,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_A",
            detected_time_col="time_s",
            detected_units={"potential:Potential_V": "V", "current:Current_A": "A", "time:time_s": "s"},
            warnings=[],
        )

        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)

        self.assertEqual(details["cycle_count"], "1")
        self.assertIn("100", details["scan_rate_estimate"])
        self.assertAlmostEqual(
            APP.representative_scan_rate_v_s(time_s, potential),
            0.1,
            delta=0.002,
        )

    def test_candidate_peak_display_filter_keeps_selected_and_limits_top_candidates(self) -> None:
        peaks = [
            APP.Peak(
                id=f"peak_{idx}",
                peak_type="oxidation" if idx % 2 == 0 else "reduction",
                index=idx,
                potential=float(idx) / 100,
                raw_current=float(idx) * 1e-6,
                prominence=float(idx) * 1e-6,
                segment_index=1,
                segment_direction="forward" if idx % 2 == 0 else "reverse",
                confidence="high" if idx in {4, 8, 12} else "medium",
            )
            for idx in range(15)
        ]

        filtered = APP.filter_candidate_peaks_for_display(
            peaks,
            peaks[0],
            peaks[1],
            "Top candidates",
        )

        self.assertEqual([peak.id for peak in filtered[:2]], ["peak_0", "peak_1"])
        self.assertEqual(len(filtered), 12)
        self.assertEqual([peak.id for peak in filtered[2:]], [f"peak_{idx}" for idx in range(14, 4, -1)])

        high_confidence = APP.filter_candidate_peaks_for_display(
            peaks,
            peaks[0],
            peaks[1],
            "High-confidence candidates",
        )

        self.assertEqual([peak.id for peak in high_confidence], ["peak_4", "peak_8", "peak_12"])

    def test_selected_peak_metrics_table_hides_baseline_columns_when_inactive(self) -> None:
        peak_metric_df = pd.DataFrame(
            [
                {
                    "peak": "oxidation",
                    "potential_V": 0.3411,
                    "current_µA": 41.39,
                    "baseline_current_µA": 0.0,
                    "corrected_current_µA": 41.39,
                    "scan": "Forward",
                    "confidence": "high",
                    "method": "Maximum current on forward scan",
                }
            ]
        )

        inactive = APP.selected_peak_metrics_display_table(peak_metric_df, "uA", baseline_active=False)
        self.assertEqual(
            list(inactive.columns),
            ["peak", "potential_V", "current_µA", "scan", "confidence", "method"],
        )

        active = APP.selected_peak_metrics_display_table(peak_metric_df, "uA", baseline_active=True)
        self.assertEqual(
            list(active.columns),
            [
                "peak",
                "potential_V",
                "raw_current_µA",
                "baseline_current_µA",
                "corrected_current_µA",
                "scan",
                "confidence",
                "method",
            ],
        )


if __name__ == "__main__":
    unittest.main()
