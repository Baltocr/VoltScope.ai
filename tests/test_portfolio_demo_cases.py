import unittest
from pathlib import Path

import numpy as np

from src.utils.legacy_app import load_legacy_app


APP = load_legacy_app()
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


class UploadedFile:
    def __init__(self, path: Path, name: str | None = None) -> None:
        self.path = path
        self.name = name or path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


class InlineUploadedFile:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text

    def getvalue(self) -> bytes:
        return self.text.encode("utf-8")


def parse_sample(filename: str):
    return APP.parse_electrochem_file(UploadedFile(SAMPLE_DIR / filename))


def analyze_cv_sample(filename: str):
    dataset = parse_sample(filename)
    columns = APP.detect_experiment_columns(dataset, "cv")
    normalized = APP.normalize_dataset(
        dataset,
        columns["potential"],
        columns["current"],
        "Auto",
        "Auto",
        "Current",
        "None",
        11,
    )
    params = APP.auto_tune_cv_peak_parameters(normalized.potential_v, normalized.raw_current_a)
    analysis = APP.analyze_cv_default_peaks(
        normalized.potential_v,
        normalized.raw_current_a,
        params["min_prominence"],
        int(params["min_distance"]),
        params["min_abs_current"],
        "uA",
        baseline_background_suspected=APP.detect_sloped_baseline_background(
            normalized.potential_v,
            normalized.raw_current_a,
        ),
    )
    status = APP.cv_analysis_status(
        analysis.oxidation_peak,
        analysis.reduction_peak,
        analysis.metrics,
        normalized.potential_v,
        candidate_peaks=analysis.candidate_peaks,
        cv_behavior=analysis.behavior,
    )
    return dataset, normalized, analysis, status


class PortfolioDemoRegressionTest(unittest.TestCase):
    def test_clean_reversible_cv_demo_returns_valid_pair_metrics(self) -> None:
        dataset, _normalized, analysis, status = analyze_cv_sample("cv_clean_reversible.csv")

        self.assertEqual(dataset.rows_dropped, 0)
        self.assertEqual(analysis.behavior.behavior, "reversible_like")
        self.assertEqual(status, "Passed")
        self.assertAlmostEqual(analysis.metrics["epa_V"], 0.387, delta=0.01)
        self.assertAlmostEqual(analysis.metrics["epc_V"], 0.293, delta=0.01)
        self.assertAlmostEqual(analysis.metrics["delta_ep_V"] * 1000, 94.2, delta=12.0)
        self.assertAlmostEqual(analysis.metrics["ipa_A"] * 1e6, 27.0, delta=2.0)
        self.assertAlmostEqual(analysis.metrics["ipc_A"] * 1e6, -23.4, delta=2.0)
        self.assertIsNone(
            APP.cv_pair_metrics_review_reason(
                analysis.metrics,
                analysis.behavior.pair_confidence,
            )
        )

    def test_noisy_cv_demo_requires_review_and_invalid_pair_metrics(self) -> None:
        _dataset, _normalized, analysis, status = analyze_cv_sample("cv_noisy_ambiguous.csv")

        self.assertEqual(analysis.behavior.behavior, "noisy_ambiguous")
        self.assertEqual(status, "Review recommended")
        review_reason = APP.cv_pair_metrics_review_reason(
            analysis.metrics,
            analysis.behavior.pair_confidence,
        )
        self.assertIsNotNone(review_reason)
        self.assertIn("Invalid", review_reason)

    def test_oxidation_only_cv_demo_suppresses_reversible_pair_metrics(self) -> None:
        _dataset, _normalized, analysis, status = analyze_cv_sample("cv_oxidation_only.csv")

        self.assertEqual(analysis.behavior.behavior, "irreversible_oxidation_only")
        self.assertEqual(status, "Review recommended")
        self.assertIsNotNone(analysis.oxidation_peak)
        self.assertIsNone(analysis.reduction_peak)
        self.assertIsNone(analysis.metrics["epc_V"])
        self.assertIsNone(analysis.metrics["delta_ep_V"])
        self.assertIsNone(analysis.metrics["ipa_ipc_ratio"])

    def test_lsv_demo_reports_reliable_anodic_onset_with_metadata(self) -> None:
        dataset = parse_sample("lsv_anodic_onset_with_metadata.txt")
        columns = APP.detect_experiment_columns(dataset, "lsv")
        normalized = APP.normalize_dataset(
            dataset,
            columns["potential"],
            columns["current"],
            "Auto",
            "Auto",
            "Current",
            "None",
            11,
        )
        record = {
            "filename": dataset.filename,
            "sample_name": dataset.filename.rsplit(".", 1)[0],
            "columns": columns,
            "data": {"potential": normalized.potential_v, "current": normalized.raw_current_a},
            "parsed_dataset": dataset,
        }
        display_units = {
            "current_source": "Current",
            "x_axis": "Potential",
            "y_axis": "Current",
            "potential": "V",
            "current": "uA",
            "electrode_area_cm2": 1.0,
        }
        threshold_a = float(np.nanmax(normalized.raw_current_a) * 0.1)

        summary = APP.build_lsv_analysis_summary(
            record,
            dataset,
            threshold_a,
            display_units,
            "SCE",
            reference_electrode_source="metadata",
        )

        self.assertEqual(summary["status"], "Passed")
        self.assertEqual(summary["direction"], "Anodic")
        self.assertEqual(summary["onset_confidence"], "Reliable")
        self.assertIn("0.477 V vs SCE", summary["onset_potential"])
        self.assertEqual(summary["scan_rate"], "10 mV/s")
        self.assertEqual(summary["reference_electrode"], "SCE (from metadata)")
        self.assertEqual(summary["metadata_completeness"], "Complete")

    def test_ambiguous_current_density_requires_confirmation_before_units_are_trusted(self) -> None:
        text = "\n".join(
            [
                "Technique: Linear sweep voltammetry",
                "Electrode area: 0.196 cm^2",
                "-0.100, -0.563",
                "0.100, 0.250",
                "0.300, 2.500",
                "0.443, 3.529",
                "0.700, 20.000",
                "1.100, 35.290",
            ]
        )
        dataset = APP.parse_electrochem_file(
            InlineUploadedFile("21_lsv_current_density_do_not_normalize.csv", text)
        )
        columns = APP.detect_experiment_columns(dataset, "lsv")
        current_source = APP.current_source_guess_for_dataset(dataset, columns["current"])

        self.assertEqual(current_source, "Current density")
        self.assertEqual(APP.current_unit_confidence(dataset, columns["current"], "Auto"), "Low")
        self.assertEqual(APP.suggested_current_unit_for_context(
            dataset.dataframe[columns["current"]].to_numpy(dtype=float),
            current_source,
        ), "mA")
        self.assertIn("mA/cm^2", APP.confirmation_unit_options(current_source))
        self.assertEqual(
            APP.uploaded_signal_kind(dataset, columns["current"], current_source, "Auto"),
            "current_density",
        )
        self.assertEqual(
            APP.lsv_axis_title(
                "Current density",
                {
                    "unit_review_required": True,
                    "current_source": "Current density",
                    "current": "mA/cm^2",
                    "potential": "V",
                },
                "Ag/AgCl",
            ),
            "Current density (unit unconfirmed)",
        )


if __name__ == "__main__":
    unittest.main()
