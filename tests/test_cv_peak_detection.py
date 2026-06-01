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


def weak_noisy_cv_trace() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic weak/noisy CV regression trace similar to 10_cv_weak_peaks_noisy_uA.csv."""
    rng = np.random.default_rng(10)
    forward = np.linspace(-0.2, 0.8, 500)
    reverse = np.linspace(0.8, -0.2, 500)
    potential = np.concatenate([forward, reverse])
    noise_scale = 2.0e-6
    forward_current = (
        0.2e-6 * forward
        + 3.0e-6 * np.exp(-0.5 * ((forward - 0.32) / 0.06) ** 2)
        + rng.normal(0.0, noise_scale, forward.size)
    )
    reverse_current = (
        -0.2e-6 * reverse
        - 3.0e-6 * np.exp(-0.5 * ((reverse - 0.27) / 0.06) ** 2)
        + rng.normal(0.0, noise_scale, reverse.size)
    )
    return potential, np.concatenate([forward_current, reverse_current])


def irreversible_oxidation_only_trace() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 11_cv_irreversible_oxidation_only.csv regression trace."""
    rng = np.random.default_rng(11)
    forward = np.linspace(-0.2, 0.8, 500)
    reverse = np.linspace(0.8, -0.2, 500)
    potential = np.concatenate([forward, reverse])
    forward_current = (
        0.6e-6
        + 0.25e-6 * forward
        + 28.5e-6 * np.exp(-0.5 * ((forward - 0.3852) / 0.044) ** 2)
        + rng.normal(0.0, 0.55e-6, forward.size)
    )
    reverse_current = (
        -3.4e-6
        + 0.55e-6 * reverse
        + rng.normal(0.0, 0.75e-6, reverse.size)
    )
    return potential, np.concatenate([forward_current, reverse_current])


def irreversible_reduction_only_trace() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 12_cv_irreversible_reduction_only.csv regression trace."""
    rng = np.random.default_rng(12)
    forward = np.linspace(0.8, -0.2, 500)
    reverse = np.linspace(-0.2, 0.8, 500)
    potential = np.concatenate([forward, reverse])
    forward_current = (
        -0.8e-6
        + 0.15e-6 * forward
        - 27.0e-6 * np.exp(-0.5 * ((forward - 0.285) / 0.045) ** 2)
        + rng.normal(0.0, 0.55e-6, forward.size)
    )
    reverse_current = (
        3.2e-6
        + 0.45e-6 * reverse
        + rng.normal(0.0, 0.75e-6, reverse.size)
    )
    return potential, np.concatenate([forward_current, reverse_current])


def two_redox_couples_trace() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 13_cv_two_redox_couples.csv regression trace."""
    rng = np.random.default_rng(13)
    forward = np.linspace(-0.2, 0.8, 500)
    reverse = np.linspace(0.8, -0.2, 500)
    potential = np.concatenate([forward, reverse])
    forward_current = (
        1.0e-6
        + 0.25e-6 * forward
        + 14.5e-6 * np.exp(-0.5 * ((forward - 0.2168) / 0.04) ** 2)
        + 23.5e-6 * np.exp(-0.5 * ((forward - 0.5134) / 0.045) ** 2)
        + rng.normal(0.0, 0.6e-6, forward.size)
    )
    reverse_current = (
        -1.0e-6
        + 0.1e-6 * reverse
        - 14.0e-6 * np.exp(-0.5 * ((reverse - 0.1487) / 0.04) ** 2)
        - 22.0e-6 * np.exp(-0.5 * ((reverse - 0.4333) / 0.045) ** 2)
        + rng.normal(0.0, 0.6e-6, reverse.size)
    )
    return potential, np.concatenate([forward_current, reverse_current])


def sloped_baseline_background_trace() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 14_cv_sloped_baseline_background.csv regression trace."""
    rng = np.random.default_rng(14)
    forward = np.linspace(-0.2, 0.8, 500)
    reverse = np.linspace(0.8, -0.2, 500)
    potential = np.concatenate([forward, reverse])
    baseline_forward = 12.0e-6 * forward + 2.0e-6
    baseline_reverse = 12.0e-6 * reverse + 1.5e-6
    forward_current = (
        baseline_forward
        + 24.0e-6 * np.exp(-0.5 * ((forward - 0.39) / 0.05) ** 2)
        + rng.normal(0.0, 0.35e-6, forward.size)
    )
    reverse_current = (
        baseline_reverse
        - 5.0e-6 * np.exp(-0.5 * ((reverse - 0.30) / 0.06) ** 2)
        + rng.normal(0.0, 0.35e-6, reverse.size)
    )
    return potential, np.concatenate([forward_current, reverse_current])


def reverse_scan_first_trace() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 15_cv_reverse_scan_first.csv regression trace with branch order reversed."""
    potential, current = clean_cv_standard_columns_trace()
    forward_potential = potential[:500]
    forward_current = current[:500]
    reverse_potential = potential[500:]
    reverse_current = current[500:]
    return (
        np.concatenate([reverse_potential[::-1], forward_potential[::-1]]),
        np.concatenate([reverse_current[::-1], forward_current[::-1]]),
    )


def multicycle_no_cycle_column_trace() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic 16_cv_multicycle_no_cycle_column.csv regression trace."""
    single_cycle_potential, single_cycle_current = clean_cv_standard_columns_trace()
    scales = [0.75, 1.0, 1.25]
    potential = np.concatenate([single_cycle_potential for _scale in scales])
    current = np.concatenate([single_cycle_current * scale for scale in scales])
    inferred_cycles = APP.infer_cycle_values_from_potential(potential)
    if inferred_cycles is None:
        inferred_cycles = np.ones_like(potential)
    return potential, current, inferred_cycles


class FakeUploadedFile:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._data = text.encode("utf-8")

    def getvalue(self) -> bytes:
        return self._data


class CVPeakDetectionRegressionTest(unittest.TestCase):
    def test_lsv_analysis_summary_reports_onset_and_readable_units(self) -> None:
        potential = np.linspace(-0.1, 1.1, 650)
        max_current_a = 148.17590015810895e-6
        threshold_a = max_current_a * 0.1
        slope_width = 0.045
        inflection = 0.477062626 + 2.197224577 * slope_width
        current = max_current_a / (1.0 + np.exp(-(potential - inflection) / slope_width))
        time_s = (potential - potential[0]) / 0.01
        dataset = APP.ParsedDataset(
            filename="05_clean_lsv_standard.csv",
            dataframe=pd.DataFrame(
                {
                    "time_s": time_s,
                    "Potential_V": potential,
                    "Current_A": current,
                }
            ),
            headers=["time_s", "Potential_V", "Current_A"],
            metadata={},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_A",
            detected_time_col="time_s",
            detected_units={"potential:Potential_V": "V", "current:Current_A": "A", "time:time_s": "s"},
            warnings=[],
        )
        record = {
            "filename": "05_clean_lsv_standard.csv",
            "sample_name": "05_clean_lsv_standard",
            "columns": {"potential": "Potential_V", "current": "Current_A"},
            "data": {"potential": potential, "current": current},
        }
        display_units = {
            "current_source": "Current",
            "x_axis": "Potential",
            "y_axis": "Current density",
            "potential": "V",
            "current": "A/cm^2",
            "electrode_area_cm2": 1.0,
        }

        summary = APP.build_lsv_analysis_summary(record, dataset, threshold_a, display_units, "SCE")

        self.assertEqual(summary["status"], "Passed")
        self.assertEqual(summary["direction"], "Anodic")
        self.assertEqual(summary["onset_label"], "Anodic onset potential")
        self.assertEqual(summary["threshold"], "14.82 µA/cm²")
        self.assertEqual(summary["primary_current_label"], "Max anodic current density")
        self.assertEqual(summary["primary_current_value"], "148.2 µA/cm²")
        self.assertEqual(summary["primary_potential_label"], "Potential at max anodic current density")
        self.assertEqual(summary["max_value"], "148.2 µA/cm²")
        self.assertEqual(summary["potential_range"], "-0.100 to 1.100 V")
        self.assertEqual(summary["scan_rate"], "10 mV/s")
        self.assertEqual(summary["reference_electrode"], "SCE (user-selected)")
        self.assertIn("0.477", summary["onset_potential"])
        self.assertIn("vs SCE", summary["onset_potential"])
        self.assertIn("anodic current density first crosses 14.82 µA/cm²", summary["method_note"])
        self.assertIn("Reported vs SCE", summary["method_note"])
        self.assertEqual(APP.current_axis_label("uA/cm^2"), "Current density (µA/cm²)")

        parser_display_unit, parser_current_source = APP.readable_parser_current_unit_for_lsv([dataset])
        self.assertEqual(parser_display_unit, "uA")
        self.assertEqual(parser_current_source, "Current")
        parser_details = APP.parser_summary_details(dataset, "lsv", parser_display_unit, parser_current_source, 1.0)
        self.assertEqual(parser_details["y_quantity"], "Current")
        self.assertIn("µA", parser_details["current_range"])
        self.assertIn("A", parser_details["raw_current_range"])

        annotation_display_units = dict(display_units)
        annotation_display_units["current"] = "uA/cm^2"
        annotation = APP.build_lsv_onset_annotation(record, threshold_a, annotation_display_units, "SCE")
        self.assertIsNotNone(annotation)
        self.assertAlmostEqual(annotation["onset_potential_display"], 0.477062626, delta=0.002)
        self.assertAlmostEqual(annotation["threshold_display"], 14.817590015810896, delta=0.1)
        self.assertIn("Onset = 0.477 V vs SCE", annotation["label"])

        detailed_metrics = APP.build_lsv_detailed_metrics(
            record,
            dataset,
            threshold_a,
            annotation_display_units,
            "SCE",
        )
        self.assertEqual(list(detailed_metrics.columns), ["Metric", "Value", "Method"])
        metric_values = dict(zip(detailed_metrics["Metric"], detailed_metrics["Value"]))
        metric_methods = dict(zip(detailed_metrics["Metric"], detailed_metrics["Method"]))
        self.assertIn("Anodic onset potential", metric_values)
        self.assertIn("0.477 V vs SCE", metric_values["Anodic onset potential"])
        self.assertIn("threshold", metric_methods["Anodic onset potential"])
        self.assertEqual(metric_values["Threshold"], "14.82 µA/cm²")
        self.assertEqual(metric_values["Threshold source"], "User-defined")
        self.assertEqual(metric_methods["Threshold"], "User-defined threshold.")
        self.assertEqual(metric_values["Direction"], "Anodic")
        self.assertEqual(metric_values["Maximum anodic current density"], "148.2 µA/cm²")
        self.assertEqual(metric_values["Potential range"], "-0.100 to 1.100 V")
        self.assertEqual(metric_values["Scan rate"], "10 mV/s")
        self.assertEqual(metric_values["Number of points"], "650")

        results = APP.build_generic_results(
            [record],
            "lsv",
            lsv_threshold_a=threshold_a,
            lsv_threshold_display=14.817590015810896,
            lsv_threshold_unit="uA/cm^2",
            lsv_current_source="Current",
        )
        self.assertIn("onset_potential_V", results.columns)
        self.assertNotIn("anodic_limit_V", results.columns)
        self.assertAlmostEqual(float(results.loc[0, "onset_potential_V"]), 0.477062626, delta=0.002)

    def test_lsv_sustained_onset_rejects_boundary_crossing(self) -> None:
        potential = np.linspace(-0.1, 1.1, 300)
        threshold = 1.0e-6
        current = np.full_like(potential, 0.2e-6)
        current[:18] = 1.4e-6
        record = {
            "filename": "17_lsv_no_threshold_crossing.csv",
            "sample_name": "17_lsv_no_threshold_crossing",
            "columns": {"potential": "Potential_V", "current": "j_mA_cm2"},
            "data": {"potential": potential, "current": current},
        }
        display_units = {
            "current_source": "Current density",
            "x_axis": "Potential",
            "y_axis": "Current density",
            "potential": "V",
            "current": "uA/cm^2",
            "electrode_area_cm2": 1.0,
        }

        onset = APP.analyze_lsv_onset(potential, current, threshold, "anodic")
        self.assertEqual(onset.status, "Review needed")
        self.assertIsNone(onset.onset_potential)
        self.assertIn("scan start", onset.rejected_reason)

        summary = APP.build_lsv_analysis_summary(record, None, threshold, display_units, "Ag/AgCl")
        self.assertEqual(summary["status"], "Review needed")
        self.assertEqual(summary["onset_potential"], "Not detected")
        self.assertEqual(summary["onset_confidence"], "Not detected")
        self.assertIn("No reliable sustained threshold crossing", summary["review_note"])
        self.assertIsNone(APP.build_lsv_onset_annotation(record, threshold, display_units, "Ag/AgCl"))

    def test_lsv_sustained_onset_rejects_false_spike_and_finds_later_rise(self) -> None:
        potential = np.linspace(-0.1, 1.1, 650)
        threshold = 10e-6
        baseline = 0.4e-6 * np.sin(np.linspace(0, 8 * np.pi, potential.size))
        narrow_spike = 90e-6 * np.exp(-((potential - 0.171) / 0.005) ** 2)
        late_rise = 95e-6 / (1.0 + np.exp(-(potential - 0.62) / 0.035))
        current = baseline + narrow_spike + late_rise
        record = {
            "filename": "18_lsv_noisy_false_crossing.csv",
            "sample_name": "18_lsv_noisy_false_crossing",
            "columns": {"potential": "Potential_V", "current": "Current_A"},
            "data": {"potential": potential, "current": current},
        }
        display_units = {
            "current_source": "Current",
            "x_axis": "Potential",
            "y_axis": "Current",
            "potential": "V",
            "current": "uA",
            "electrode_area_cm2": 1.0,
        }

        onset = APP.analyze_lsv_onset(potential, current, threshold, "anodic")
        self.assertEqual(onset.status, "Passed")
        self.assertIsNotNone(onset.onset_potential)
        self.assertGreater(float(onset.onset_potential), 0.5)
        self.assertLess(float(onset.rejected_potential), 0.25)
        self.assertIn("drops back below threshold", onset.rejected_reason)

        annotation = APP.build_lsv_onset_annotation(record, threshold, display_units, "Ag/AgCl")
        self.assertIsNotNone(annotation)
        self.assertGreater(float(annotation["onset_potential_v"]), 0.5)

        metrics = APP.build_lsv_detailed_metrics(record, None, threshold, display_units, "Ag/AgCl")
        metric_values = dict(zip(metrics["Metric"], metrics["Value"]))
        self.assertEqual(metric_values["Onset reliability"], "Reliable")
        self.assertIn("drops back below threshold", metric_values["Rejected crossing reason"])

    def test_cathodic_lsv_summary_emphasizes_max_cathodic_current(self) -> None:
        potential = np.array([1.1, 1.026, 0.398, 0.2, -0.1])
        current = np.array([-20.0e-6, -94.84e-6, -10.0e-6, -3.0e-6, 5.8e-6])
        dataset = APP.ParsedDataset(
            filename="07_lsv_whitespace_no_header.dat",
            dataframe=pd.DataFrame({"Potential_V": potential, "Current_A": current}),
            headers=["Potential_V", "Current_A"],
            metadata={},
            delimiter="whitespace",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_A",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V", "current:Current_A": "A"},
            warnings=[],
        )
        record = {
            "filename": "07_lsv_whitespace_no_header.dat",
            "sample_name": "07_lsv_whitespace_no_header",
            "columns": {"potential": "Potential_V", "current": "Current_A"},
            "data": {"potential": potential, "current": current},
        }
        display_units = {
            "current_source": "Current",
            "x_axis": "Potential",
            "y_axis": "Current",
            "potential": "V",
            "current": "uA",
            "electrode_area_cm2": 1.0,
        }

        source, rule = APP.lsv_threshold_source_details("Auto-selected", "current", "cathodic")
        self.assertEqual(APP.lsv_dominant_response_direction(current), "cathodic")
        summary = APP.build_lsv_analysis_summary(
            record,
            dataset,
            10.0e-6,
            display_units,
            "Ag/AgCl",
            source,
            rule,
            reference_electrode_source="default",
        )

        self.assertEqual(summary["direction"], "Cathodic")
        self.assertEqual(summary["onset_label"], "Cathodic onset potential")
        self.assertIn("0.398 V vs Ag/AgCl", summary["onset_potential"])
        self.assertEqual(summary["threshold_rule"], "Auto-selected as 10% of maximum cathodic current magnitude.")
        self.assertEqual(summary["threshold_magnitude"], "10 µA")
        self.assertEqual(summary["crossing_threshold"], "-10 µA")
        self.assertEqual(summary["crossing_threshold_label"], "Cathodic crossing threshold")
        self.assertEqual(summary["reference_electrode"], "Ag/AgCl (default)")
        self.assertIn("using default Ag/AgCl", summary["reference_note"])
        self.assertEqual(summary["scan_rate"], "Not available")
        self.assertEqual(
            summary["scan_rate_note"],
            "Scan rate could not be estimated because no time column was found.",
        )
        self.assertEqual(summary["primary_current_label"], "Max cathodic current")
        self.assertEqual(summary["primary_current_value"], "-94.84 µA")
        self.assertEqual(summary["primary_potential_label"], "Potential at max cathodic current")
        self.assertEqual(summary["primary_potential"], "1.026 V vs Ag/AgCl")
        self.assertEqual(summary["max_value"], "5.8 µA")
        self.assertIn("cathodic current first crosses -10 µA", summary["method_note"])

        detailed_metrics = APP.build_lsv_detailed_metrics(
            record,
            dataset,
            10.0e-6,
            display_units,
            "Ag/AgCl",
            source,
            rule,
        )
        detailed_methods = dict(zip(detailed_metrics["Metric"], detailed_metrics["Method"]))
        detailed_values = dict(zip(detailed_metrics["Metric"], detailed_metrics["Value"]))
        self.assertEqual(detailed_methods["Threshold magnitude"], "Auto-selected as 10% of maximum cathodic current magnitude.")
        self.assertEqual(detailed_values["Threshold magnitude"], "10 µA")
        self.assertEqual(detailed_values["Cathodic crossing threshold"], "-10 µA")
        self.assertEqual(detailed_values["Maximum cathodic current / minimum current"], "-94.84 µA")
        self.assertEqual(detailed_values["Scan rate"], "Not available")
        self.assertEqual(
            detailed_methods["Scan rate"],
            "Scan rate could not be estimated because no time column was found.",
        )

        parser_details = APP.parser_summary_details(dataset, "lsv", "uA", "Current", 1.0)
        self.assertEqual(parser_details["scan_rate_estimate"], "Not available")
        self.assertEqual(
            parser_details["scan_rate_note"],
            "Scan rate could not be estimated because no time column was found.",
        )

        raw_axis_options, raw_default_axis = APP.lsv_axis_options_for_current_source("Current")
        density_axis_options, density_default_axis = APP.lsv_axis_options_for_current_source("Current density")
        self.assertIn("Current", raw_axis_options)
        self.assertEqual(raw_default_axis, "Current")
        self.assertNotIn("Current", density_axis_options)
        self.assertEqual(density_default_axis, "Current density")

        raw_display_state = APP.format_lsv_display_state(display_units, 9.484e-6, "Ag/AgCl")
        self.assertIn("Display: Current", raw_display_state)
        self.assertIn("threshold = 9.484 µA", raw_display_state)
        self.assertNotIn("area =", raw_display_state)

        density_from_raw_state = APP.format_lsv_display_state(
            {
                "current_source": "Current",
                "y_axis": "Current density",
                "current": "uA/cm^2",
                "electrode_area_cm2": 0.196,
            },
            9.484e-6,
            "Ag/AgCl",
        )
        self.assertIn("Display: Current density", density_from_raw_state)
        self.assertIn("area = 0.196 cm²", density_from_raw_state)

        self.assertEqual(
            APP.reference_electrode_from_metadata({"Reference electrode": "SCE"}),
            "SCE",
        )
        self.assertEqual(
            APP.reference_electrode_from_metadata({"Reference": "Ag/AgCl (3 M KCl)"}),
            "Ag/AgCl",
        )
        self.assertEqual(
            APP.reference_electrode_source_label("Ag/AgCl", "metadata"),
            "Ag/AgCl (from metadata)",
        )

    def test_lsv_scan_rate_uses_metadata_when_time_column_is_missing(self) -> None:
        potential = np.array([-0.1, 0.1, 0.3, 0.5, 0.7])
        current = np.array([0.0, 1.0e-6, 4.0e-6, 12.0e-6, 40.0e-6])
        dataset = APP.ParsedDataset(
            filename="17_lsv_no_time_scan_rate_metadata.csv",
            dataframe=pd.DataFrame({"Potential_V": potential, "Current_A": current}),
            headers=["Potential_V", "Current_A"],
            metadata={"Scan rate": "10 mV/s"},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_A",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V", "current:Current_A": "A"},
            warnings=[],
        )
        record = {
            "filename": dataset.filename,
            "sample_name": "17_lsv_no_time_scan_rate_metadata",
            "columns": {"potential": "Potential_V", "current": "Current_A"},
            "data": {"potential": potential, "current": current},
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

        parser_details = APP.parser_summary_details(dataset, "lsv", "uA", "Current", 1.0)
        self.assertEqual(APP.summarize_scan_rate_for_card(parser_details["scan_rate_estimate"]), "10 mV/s")
        self.assertEqual(parser_details["scan_rate_note"], "")

        summary = APP.build_lsv_analysis_summary(record, dataset, 4.0e-6, display_units, "Ag/AgCl")
        self.assertEqual(summary["scan_rate"], "10 mV/s")
        self.assertEqual(summary["scan_rate_note"], "")

        detailed_metrics = APP.build_lsv_detailed_metrics(record, dataset, 4.0e-6, display_units, "Ag/AgCl")
        detailed_values = dict(zip(detailed_metrics["Metric"], detailed_metrics["Value"]))
        detailed_methods = dict(zip(detailed_metrics["Metric"], detailed_metrics["Method"]))
        self.assertEqual(detailed_values["Scan rate"], "10 mV/s")
        self.assertEqual(detailed_methods["Scan rate"], "Parsed from file metadata.")

        dataset.metadata = {"scan_rate_mV_s": "10"}
        parser_details = APP.parser_summary_details(dataset, "lsv", "uA", "Current", 1.0)
        self.assertEqual(APP.summarize_scan_rate_for_card(parser_details["scan_rate_estimate"]), "10 mV/s")
        self.assertEqual(parser_details["scan_rate_note"], "")
        summary = APP.build_lsv_analysis_summary(record, dataset, 4.0e-6, display_units, "Ag/AgCl")
        self.assertEqual(summary["scan_rate"], "10 mV/s")
        self.assertEqual(summary["scan_rate_note"], "")

    def test_manual_metadata_text_can_supply_scan_rate(self) -> None:
        manual_metadata = APP.parse_manual_metadata_text("Scan rate: 25 mV/s\nElectrode: glassy carbon")
        dataset = APP.ParsedDataset(
            filename="manual_scan_rate_lsv.csv",
            dataframe=pd.DataFrame(
                {
                    "Potential_V": np.linspace(-0.1, 1.1, 10),
                    "Current_A": np.linspace(0.0, 10.0e-6, 10),
                }
            ),
            headers=["Potential_V", "Current_A"],
            metadata=manual_metadata,
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_A",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V", "current:Current_A": "A"},
            warnings=[],
        )

        details = APP.parser_summary_details(dataset, "lsv", "uA", "Current", 1.0)

        self.assertEqual(manual_metadata["Scan rate"], "25 mV/s")
        self.assertEqual(manual_metadata["Electrode"], "glassy carbon")
        self.assertEqual(APP.summarize_scan_rate_for_card(details["scan_rate_estimate"]), "25 mV/s")
        self.assertEqual(APP.scan_rate_method_summary(dataset, "Potential_V"), "Parsed from file metadata.")

    def test_lsv_current_density_parser_summary_uses_density_labels(self) -> None:
        potential = np.linspace(-0.1, 1.1, 50)
        current_density_ma_cm2 = np.linspace(-0.78, 57.95, 50)
        dataset = APP.ParsedDataset(
            filename="06_lsv_current_density_metadata.csv",
            dataframe=pd.DataFrame(
                {
                    "Potential_V": potential,
                    "j_mA_cm2": current_density_ma_cm2,
                }
            ),
            headers=["Potential_V", "j_mA_cm2"],
            metadata={"Technique": "Linear sweep voltammetry"},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="j_mA_cm2",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V", "current:j_mA_cm2": "mA"},
            warnings=[],
        )

        parser_display_unit, parser_current_source = APP.readable_parser_current_unit_for_lsv([dataset])
        self.assertEqual(parser_display_unit, "mA/cm^2")
        self.assertEqual(parser_current_source, "Current density")

        details = APP.parser_summary_details(dataset, "lsv", parser_display_unit, parser_current_source, 1.0)
        self.assertEqual(details["y_quantity"], "Current density")
        self.assertEqual(details["current_column"], "j_mA_cm2")
        self.assertEqual(details["raw_current_unit"], "mA/cm²")
        self.assertEqual(details["analysis_current_unit"], "A/cm²")
        self.assertEqual(details["display_current_unit"], "mA/cm²")
        self.assertEqual(details["current_range"], "-0.78 to 57.95 mA/cm²")
        self.assertIn("Current density", details["units"])
        self.assertEqual(APP.electrode_area_metadata_summary(dataset.metadata), None)

        dataset.metadata["Electrode area"] = "0.196 cm^2"
        self.assertEqual(APP.electrode_area_metadata_summary(dataset.metadata), "0.196 cm²")
        display_state = APP.format_lsv_display_state(
            {
                "current_source": "Current density",
                "y_axis": "Current density",
                "current": "mA/cm^2",
                "electrode_area_cm2": 1.0,
            },
            0.005795,
            "Ag/AgCl",
            APP.electrode_area_metadata_summary(dataset.metadata),
        )
        self.assertIn("Current density from uploaded column", display_state)
        self.assertIn("threshold = 5.795 mA/cm²", display_state)
        self.assertIn("Electrode area metadata: 0.196 cm²", display_state)
        self.assertIn("No additional normalization applied", display_state)
        self.assertNotIn("area = 1.0 cm²", display_state)

        density_display_units = {
            "current_source": "Current density",
            "x_axis": "Potential",
            "y_axis": "Current density",
            "potential": "V",
            "current": "mA/cm^2",
            "electrode_area_cm2": 1.0,
        }
        density_record = {
            "filename": "06_lsv_current_density_metadata.csv",
            "sample_name": "06_lsv_current_density_metadata",
            "columns": {"potential": "Potential_V", "current": "j_mA_cm2"},
            "data": {
                "potential": potential,
                "current": current_density_ma_cm2 * 1e-3,
            },
        }
        threshold_base = 5.795e-3
        source, rule = APP.lsv_threshold_source_details("Auto-selected", "current density", "anodic")
        self.assertEqual(
            APP.lsv_dominant_response_direction(current_density_ma_cm2 * 1e-3),
            "anodic",
        )
        summary = APP.build_lsv_analysis_summary(
            density_record,
            dataset,
            threshold_base,
            density_display_units,
            "Ag/AgCl",
            source,
            rule,
            reference_electrode_source="metadata",
        )
        self.assertEqual(summary["threshold_source"], "Auto-selected")
        self.assertEqual(summary["threshold_rule"], "Auto-selected as 10% of maximum anodic current density.")
        self.assertEqual(summary["reference_electrode"], "Ag/AgCl (from metadata)")
        self.assertEqual(summary["threshold"], "5.795 mA/cm²")
        self.assertIn("anodic current density first crosses 5.795 mA/cm²", summary["method_note"])
        self.assertIn("Reported vs Ag/AgCl", summary["method_note"])

        density_metrics = APP.build_lsv_detailed_metrics(
            density_record,
            dataset,
            threshold_base,
            density_display_units,
            "Ag/AgCl",
            source,
            rule,
        )
        density_metric_values = dict(zip(density_metrics["Metric"], density_metrics["Value"]))
        density_metric_methods = dict(zip(density_metrics["Metric"], density_metrics["Method"]))
        self.assertEqual(density_metric_values["Threshold source"], "Auto-selected")
        self.assertEqual(density_metric_methods["Threshold"], "Auto-selected as 10% of maximum anodic current density.")
        self.assertIn("1 mA/cm²", APP.lsv_threshold_preset_options("mA/cm^2"))
        self.assertIn("100 µA", APP.lsv_threshold_preset_options("uA"))

    def test_raw_current_lsv_metadata_is_visible_and_exported(self) -> None:
        potential = np.linspace(-0.1, 1.1, 50)
        time_s = (potential - potential[0]) / 0.01
        current = np.linspace(-2.0e-6, 80.0e-6, 50)
        dataset = APP.ParsedDataset(
            filename="20_lsv_txt_metadata_current.txt",
            dataframe=pd.DataFrame(
                {
                    "time_s": time_s,
                    "Potential_V": potential,
                    "Current_A": current,
                }
            ),
            headers=["time_s", "Potential_V", "Current_A"],
            metadata={"Reference electrode": "SCE", "Electrode area": "0.071 cm^2"},
            delimiter="tab / whitespace",
            has_header=True,
            header_row=4,
            data_start_row=5,
            rows_skipped=4,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_A",
            detected_time_col="time_s",
            detected_units={"time:time_s": "s", "potential:Potential_V": "V", "current:Current_A": "A"},
            warnings=[],
        )
        record = {
            "filename": "20_lsv_txt_metadata_current.txt",
            "sample_name": "20_lsv_txt_metadata_current",
            "columns": {"potential": "Potential_V", "current": "Current_A"},
            "data": {"potential": potential, "current": current},
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

        details = APP.parser_summary_details(dataset, "lsv", "uA", "Current", 1.0)
        self.assertEqual(details["reference_electrode_metadata"], "SCE")
        self.assertEqual(details["electrode_area_metadata"], "0.071 cm²")
        self.assertEqual(APP.summarize_scan_rate_for_card(details["scan_rate_estimate"]), "10 mV/s")

        display_state = APP.format_lsv_display_state(
            display_units,
            8.0e-6,
            "SCE",
            APP.electrode_area_metadata_summary(dataset.metadata),
        )
        self.assertIn("Display: Current", display_state)
        self.assertIn("Electrode area detected: 0.071 cm²", display_state)
        self.assertIn("Current density display is available", display_state)

        metrics = APP.build_lsv_detailed_metrics(record, dataset, 8.0e-6, display_units, "SCE")
        metric_values = dict(zip(metrics["Metric"], metrics["Value"]))
        self.assertEqual(metric_values["Reference electrode metadata"], "SCE")
        self.assertEqual(metric_values["Electrode area metadata"], "0.071 cm²")
        self.assertEqual(metric_values["Scan rate"], "10 mV/s")

        results = APP.build_generic_results(
            [record],
            "lsv",
            lsv_threshold_a=8.0e-6,
            lsv_threshold_display=8.0,
            lsv_threshold_unit="uA",
            lsv_current_source="Current",
        )
        self.assertEqual(results.loc[0, "reference_electrode_metadata"], "SCE")
        self.assertEqual(results.loc[0, "electrode_area_metadata"], "0.071 cm²")
        self.assertIn("10 mV/s", results.loc[0, "scan_rate_estimate"])
        self.assertEqual(results.loc[0, "uploaded_signal_kind"], "raw_current")
        self.assertFalse(bool(results.loc[0, "normalization_applied"]))

        density_results = APP.build_generic_results(
            [record],
            "lsv",
            lsv_threshold_a=8.0e-6,
            lsv_threshold_display=112.676,
            lsv_threshold_unit="uA/cm^2",
            lsv_current_source="Current",
            lsv_electrode_area_cm2=0.071,
            lsv_selected_current_unit="A",
        )
        self.assertEqual(density_results.loc[0, "uploaded_signal_kind"], "raw_current")
        self.assertTrue(bool(density_results.loc[0, "normalization_applied"]))
        self.assertAlmostEqual(float(density_results.loc[0, "electrode_area_cm2"]), 0.071)

    def test_no_header_lsv_unknown_signal_stays_unconfirmed(self) -> None:
        potential = np.linspace(-0.1, 1.1, 50)
        signal = np.linspace(0.2, 35.29, 50)
        current_col = "Inferred column 2"
        dataset = APP.ParsedDataset(
            filename="21_lsv_unknown_signal.csv",
            dataframe=pd.DataFrame({"Inferred column 1": potential, current_col: signal}),
            headers=["Inferred column 1", current_col],
            metadata={},
            delimiter=",",
            has_header=False,
            header_row=None,
            data_start_row=1,
            rows_skipped=0,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Inferred column 1",
            detected_current_col=current_col,
            detected_time_col=None,
            detected_units={"potential:Inferred column 1": "V"},
            warnings=[],
        )

        self.assertEqual(APP.current_source_guess_for_dataset(dataset, current_col), "Current")
        details = APP.parser_summary_details(dataset, "lsv", "A", None, 1.0)
        self.assertEqual(details["y_quantity"], "Current")
        self.assertEqual(details["current_unit_confidence"], "Low")
        self.assertEqual(details["suggested_current_unit"], "µA based on value magnitude")

        display_units = {
            "current_source": "Current",
            "x_axis": "Potential",
            "y_axis": "Current",
            "potential": "V",
            "current": "A",
            "electrode_area_cm2": 1.0,
            "unit_review_required": True,
        }
        self.assertEqual(APP.lsv_axis_title("Current", display_units, "Ag/AgCl"), "Signal (unit unconfirmed)")
        record = {
            "filename": dataset.filename,
            "sample_name": "21_lsv_unknown_signal",
            "columns": {"potential": "Inferred column 1", "current": current_col},
            "data": {"potential": potential, "current": signal},
            "parsed_dataset": dataset,
        }
        summary = APP.build_lsv_analysis_summary(
            record,
            dataset,
            3.529,
            display_units,
            "Ag/AgCl",
            unit_review_needed=True,
        )
        self.assertEqual(summary["status"], "Review needed")
        self.assertEqual(summary["primary_current_value"], "Pending unit confirmation")
        self.assertEqual(summary["threshold"], "3.529 unit unconfirmed")

        fig = APP.go.Figure()
        APP.add_generic_trace(fig, record, "lsv", display_units, False)
        self.assertIn("Signal: %{y:.4g} unit unconfirmed", fig.data[0].hovertemplate)

        results = APP.build_generic_results(
            [record],
            "lsv",
            lsv_threshold_a=3.529,
            lsv_threshold_display=3.529,
            lsv_threshold_unit="A",
            lsv_current_source="Current",
            lsv_selected_current_unit="Auto",
        )
        self.assertEqual(results.loc[0, "uploaded_signal_kind"], "unknown")
        self.assertTrue(bool(results.loc[0, "unit_review_required"]))
        self.assertEqual(results.loc[0, "y_axis_quantity"], "Current")
        self.assertFalse(bool(results.loc[0, "normalization_applied"]))

    def test_current_density_filename_skips_area_normalization_until_unit_confirmed(self) -> None:
        potential = np.linspace(-0.1, 1.1, 400)
        min_density = -0.563
        max_density = 35.29
        threshold_display = 3.52867
        onset_target = 0.443
        width = 0.04
        fraction_at_onset = (threshold_display - min_density) / (max_density - min_density)
        inflection = onset_target - width * np.log(fraction_at_onset / (1.0 - fraction_at_onset))
        density_ma_cm2 = min_density + (max_density - min_density) / (
            1.0 + np.exp(-(potential - inflection) / width)
        )
        current_col = "Inferred column 2"
        dataset = APP.ParsedDataset(
            filename="21_lsv_current_density_do_not_normalize.csv",
            dataframe=pd.DataFrame({"Inferred column 1": potential, current_col: density_ma_cm2}),
            headers=["Inferred column 1", current_col],
            metadata={"Electrode area": "0.196 cm^2"},
            delimiter=",",
            has_header=False,
            header_row=None,
            data_start_row=1,
            rows_skipped=0,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Inferred column 1",
            detected_current_col=current_col,
            detected_time_col=None,
            detected_units={"potential:Inferred column 1": "V"},
            warnings=[],
        )

        self.assertEqual(APP.current_source_guess_for_dataset(dataset, current_col), "Current density")
        self.assertEqual(APP.suggested_current_unit_for_context(density_ma_cm2, "Current density"), "mA")
        self.assertEqual(APP.confirmation_unit_options("Current density"), ["A/cm^2", "mA/cm^2", "uA/cm^2", "nA/cm^2", "pA/cm^2"])
        self.assertEqual(APP.unit_input_from_confirmed_unit("mA/cm^2", "Current density"), "mA")
        self.assertEqual(APP.confirmation_unit_type("Current density"), "current_density")
        details = APP.parser_summary_details(dataset, "lsv", "mA/cm^2", None, 1.0)
        self.assertEqual(details["y_quantity"], "Current density")
        self.assertEqual(details["current_unit_confidence"], "Low")
        self.assertEqual(details["suggested_current_unit"], "mA/cm² based on value magnitude")
        self.assertEqual(details["electrode_area_metadata"], "0.196 cm²")

        warning = APP.current_unit_magnitude_warning(
            dataset,
            current_col,
            "Auto",
            "A",
            "Current density",
            1.0,
        )
        self.assertIn("A/cm²", warning)
        explanation = APP.current_unit_review_explanation(current_col, "mA/cm^2", warning)
        self.assertIn("current-density column", explanation)

        display_units = {
            "current_source": "Current density",
            "x_axis": "Potential",
            "y_axis": "Current density",
            "potential": "V",
            "current": "A/cm^2",
            "electrode_area_cm2": 1.0,
            "unit_review_required": True,
        }
        self.assertEqual(
            APP.lsv_axis_title("Current density", display_units, "Ag/AgCl"),
            "Current density (unit unconfirmed)",
        )
        display_state = APP.format_lsv_display_state(
            display_units,
            3.529,
            "Ag/AgCl",
            APP.electrode_area_metadata_summary(dataset.metadata),
            unit_review_required=True,
        )
        self.assertIn("Current density from uploaded column", display_state)
        self.assertIn("threshold = 3.529 unit unconfirmed", display_state)
        self.assertIn("No additional normalization applied", display_state)
        self.assertNotIn("area = 1.0 cm²", display_state)

        confirmed_details = APP.parser_summary_details(dataset, "lsv", "mA/cm^2", None, 1.0, "mA")
        self.assertEqual(confirmed_details["current_unit_confidence"], "Confirmed by user")
        self.assertEqual(confirmed_details["current_range"], "-0.56 to 35.29 mA/cm²")
        self.assertEqual(confirmed_details["raw_current_unit"], "mA/cm²")
        self.assertIn("Current unit was confirmed by user as mA/cm²", confirmed_details["unit_header_audit"])

        record = {
            "filename": dataset.filename,
            "sample_name": "21_lsv_current_density_do_not_normalize",
            "columns": {"potential": "Inferred column 1", "current": current_col},
            "data": {"potential": potential, "current": density_ma_cm2 * 1e-3},
            "parsed_dataset": dataset,
        }
        confirmed_display_units = {
            "current_source": "Current density",
            "x_axis": "Potential",
            "y_axis": "Current density",
            "potential": "V",
            "current": "mA/cm^2",
            "electrode_area_cm2": 1.0,
            "unit_review_required": False,
        }
        confirmed_summary = APP.build_lsv_analysis_summary(
            record,
            dataset,
            threshold_display * 1e-3,
            confirmed_display_units,
            "Ag/AgCl",
        )
        self.assertNotEqual(confirmed_summary["status"], "Review needed")
        self.assertEqual(confirmed_summary["threshold"], "3.529 mA/cm²")
        self.assertEqual(confirmed_summary["primary_current_value"], "35.29 mA/cm²")
        self.assertEqual(confirmed_summary["min_value"], "-0.563 mA/cm²")
        self.assertIn("0.443 V vs Ag/AgCl", confirmed_summary["onset_potential"])
        self.assertEqual(APP.lsv_axis_title("Current density", confirmed_display_units, "Ag/AgCl"), "Current density (mA/cm²)")

        confirmed_display_state = APP.format_lsv_display_state(
            confirmed_display_units,
            threshold_display * 1e-3,
            "Ag/AgCl",
            APP.electrode_area_metadata_summary(dataset.metadata),
            unit_review_required=False,
        )
        self.assertIn("threshold = 3.529 mA/cm²", confirmed_display_state)
        self.assertIn("Uploaded signal was treated as current density", confirmed_display_state)
        self.assertIn("Electrode-area normalization was skipped", confirmed_display_state)

        results = APP.build_generic_results(
            [record],
            "lsv",
            lsv_threshold_a=threshold_display * 1e-3,
            lsv_threshold_display=threshold_display,
            lsv_threshold_unit="mA/cm^2",
            lsv_current_source="Current density",
            lsv_electrode_area_cm2=1.0,
            lsv_selected_current_unit="mA",
        )
        self.assertEqual(results.loc[0, "uploaded_signal_kind"], "current_density")
        self.assertFalse(bool(results.loc[0, "normalization_applied"]))
        self.assertFalse(bool(results.loc[0, "unit_review_required"]))
        self.assertEqual(results.loc[0, "y_axis_quantity"], "Current density")
        self.assertAlmostEqual(float(results.loc[0, "electrode_area_cm2"]), 0.196)
        self.assertEqual(results.loc[0, "confirmed_unit"], "mA/cm²")
        self.assertEqual(results.loc[0, "confirmed_unit_type"], "current_density")
        self.assertEqual(results.loc[0, "unit_confirmation_source"], "user")

    def test_lsv_confirmed_density_unit_uses_user_selected_signal_type(self) -> None:
        dataset = APP.ParsedDataset(
            filename="manual_density_signal.csv",
            dataframe=pd.DataFrame(
                {
                    "Potential_V": np.linspace(-0.1, 1.1, 20),
                    "Signal": np.linspace(-0.5, 35.0, 20),
                }
            ),
            headers=["Potential_V", "Signal"],
            metadata={},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Signal",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V"},
            warnings=[],
        )
        source_key = APP.lsv_current_source_state_key(
            "project",
            "experiment",
            dataset.filename,
            "Signal",
        )

        source = APP.confirmed_unit_source_for_lsv(
            dataset,
            "Signal",
            source_key,
            {source_key: "Current density"},
        )

        self.assertEqual(APP.current_source_guess_for_dataset(dataset, "Signal"), "Current")
        self.assertEqual(source, "Current density")
        self.assertEqual(APP.unit_input_from_confirmed_unit("mA/cm^2", source), "mA")
        self.assertEqual(APP.confirmation_unit_type(source), "current_density")

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
        behavior = APP.classify_cv_behavior(selected_oxidation, selected_reduction, metrics, potential, peaks)
        self.assertEqual(behavior.behavior, "reversible_like")
        self.assertEqual(
            APP.cv_analysis_quality_messages(selected_oxidation, selected_reduction, metrics, potential, peaks, behavior),
            [],
        )
        self.assertEqual(
            APP.cv_analysis_status(selected_oxidation, selected_reduction, metrics, potential, False, peaks, behavior),
            "Passed",
        )
        self.assertEqual(
            APP.cv_analysis_quality_label(selected_oxidation, selected_reduction, metrics, potential, False, peaks, behavior),
            "High",
        )

    def test_weak_noisy_cv_requires_peak_review(self) -> None:
        potential, current = weak_noisy_cv_trace()
        params = APP.auto_tune_cv_peak_parameters(potential, current)
        peaks = APP.detect_peaks(
            potential,
            current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
        )
        selected_oxidation, selected_reduction = APP.select_default_primary_peaks(peaks)
        self.assertIsNotNone(selected_oxidation)
        self.assertIsNotNone(selected_reduction)

        _rows, metrics = APP.build_peak_metrics_rows(
            selected_oxidation,
            selected_reduction,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        messages = APP.cv_analysis_quality_messages(
            selected_oxidation,
            selected_reduction,
            metrics,
            potential,
            peaks,
        )
        behavior = APP.classify_cv_behavior(selected_oxidation, selected_reduction, metrics, potential, peaks)

        self.assertIn(selected_oxidation.confidence, {"medium", "low"})
        self.assertIn(selected_reduction.confidence, {"medium", "low"})
        self.assertEqual(behavior.behavior, "noisy_ambiguous")
        self.assertEqual(
            APP.cv_analysis_status(selected_oxidation, selected_reduction, metrics, potential, False, peaks, behavior),
            "Review recommended",
        )
        self.assertNotEqual(
            APP.cv_analysis_quality_label(selected_oxidation, selected_reduction, metrics, potential, False, peaks, behavior),
            "High",
        )
        self.assertTrue(any("Weak/noisy CV" in message for message in messages))
        self.assertTrue(any("similar prominence" in message for message in messages))

    def test_irreversible_oxidation_only_rejects_weak_return_peak(self) -> None:
        potential, current = irreversible_oxidation_only_trace()
        params = APP.auto_tune_cv_peak_parameters(potential, current)
        peaks = APP.detect_peaks(
            potential,
            current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
        )
        raw_oxidation, raw_reduction = APP.select_default_primary_peaks(peaks)
        self.assertIsNotNone(raw_oxidation)
        self.assertIsNotNone(raw_reduction)

        _raw_rows, raw_metrics = APP.build_peak_metrics_rows(
            raw_oxidation,
            raw_reduction,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        behavior = APP.classify_cv_behavior(raw_oxidation, raw_reduction, raw_metrics, potential, peaks)
        self.assertEqual(behavior.behavior, "irreversible_oxidation_only")
        self.assertIsNotNone(behavior.oxidation_peak)
        self.assertIsNone(behavior.reduction_peak)
        self.assertIs(behavior.rejected_reduction_peak, raw_reduction)
        self.assertAlmostEqual(behavior.oxidation_peak.potential, 0.3852, delta=0.02)
        self.assertAlmostEqual(behavior.oxidation_peak.raw_current * 1e6, 29.05, delta=2.0)

        _rows, metrics = APP.build_peak_metrics_rows(
            behavior.oxidation_peak,
            behavior.reduction_peak,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        self.assertIsNone(metrics["epc_V"])
        self.assertIsNone(metrics["ipc_A"])
        self.assertIsNone(metrics["delta_ep_V"])
        self.assertIsNone(metrics["ipa_ipc_ratio"])

        messages = APP.cv_analysis_quality_messages(
            behavior.oxidation_peak,
            behavior.reduction_peak,
            metrics,
            potential,
            peaks,
            behavior,
        )
        self.assertTrue(any("Irreversible or oxidation-only CV" in message for message in messages))
        self.assertFalse(any("Weak/noisy CV" in message for message in messages))
        self.assertEqual(
            APP.cv_analysis_status(
                behavior.oxidation_peak,
                behavior.reduction_peak,
                metrics,
                potential,
                False,
                peaks,
                behavior,
            ),
            "Review recommended",
        )

        compact_metrics = APP.metrics_table(metrics, "uA", 1.0, "Current", behavior.behavior)
        metric_values = dict(zip(compact_metrics["metric"], compact_metrics["value"]))
        self.assertEqual(metric_values["Epc"], "No reliable cathodic peak")
        self.assertEqual(metric_values["Ipc"], "Not detected")
        self.assertEqual(metric_values["\u0394Ep"], "Not applicable")
        self.assertEqual(metric_values["E\u00b0\u2032"], "Not applicable")
        self.assertEqual(metric_values["|Ipa/Ipc|"], "Not meaningful")
        self.assertEqual(APP.selected_peak_legend_label(behavior), "Selected Epa")
        self.assertIn("Only the selected Epa is used", APP.candidate_peak_explanation(behavior))
        self.assertIn(
            "no reliable cathodic return peak",
            APP.candidate_rejection_reasons(behavior)[raw_reduction.id],
        )

    def test_irreversible_reduction_only_rejects_weak_return_peak(self) -> None:
        potential, current = irreversible_reduction_only_trace()
        params = APP.auto_tune_cv_peak_parameters(potential, current)
        peaks = APP.detect_peaks(
            potential,
            current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
        )
        raw_oxidation, raw_reduction = APP.select_default_primary_peaks(peaks)
        self.assertIsNotNone(raw_oxidation)
        self.assertIsNotNone(raw_reduction)

        _raw_rows, raw_metrics = APP.build_peak_metrics_rows(
            raw_oxidation,
            raw_reduction,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        behavior = APP.classify_cv_behavior(raw_oxidation, raw_reduction, raw_metrics, potential, peaks)
        self.assertEqual(behavior.behavior, "irreversible_reduction_only")
        self.assertIsNone(behavior.oxidation_peak)
        self.assertIsNotNone(behavior.reduction_peak)
        self.assertIs(behavior.rejected_oxidation_peak, raw_oxidation)
        self.assertAlmostEqual(behavior.reduction_peak.potential, 0.285, delta=0.03)
        self.assertLess(behavior.reduction_peak.raw_current, -20e-6)

        _rows, metrics = APP.build_peak_metrics_rows(
            behavior.oxidation_peak,
            behavior.reduction_peak,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        self.assertIsNone(metrics["epa_V"])
        self.assertIsNone(metrics["ipa_A"])
        self.assertIsNone(metrics["delta_ep_V"])
        self.assertIsNone(metrics["ipa_ipc_ratio"])

        messages = APP.cv_analysis_quality_messages(
            behavior.oxidation_peak,
            behavior.reduction_peak,
            metrics,
            potential,
            peaks,
            behavior,
        )
        self.assertTrue(any("Irreversible or reduction-only CV" in message for message in messages))
        self.assertTrue(any("not applicable because no reliable anodic return peak" in message for message in messages))
        self.assertFalse(any("Weak/noisy CV" in message for message in messages))
        self.assertEqual(
            APP.cv_analysis_status(
                behavior.oxidation_peak,
                behavior.reduction_peak,
                metrics,
                potential,
                False,
                peaks,
                behavior,
            ),
            "Review recommended",
        )

        compact_metrics = APP.metrics_table(metrics, "uA", 1.0, "Current", behavior.behavior)
        metric_values = dict(zip(compact_metrics["metric"], compact_metrics["value"]))
        self.assertEqual(metric_values["Epa"], "No reliable anodic peak")
        self.assertEqual(metric_values["Ipa"], "Not detected")
        self.assertEqual(metric_values["\u0394Ep"], "Not applicable")
        self.assertEqual(metric_values["E\u00b0\u2032"], "Not applicable")
        self.assertEqual(metric_values["|Ipa/Ipc|"], "Not meaningful")
        self.assertEqual(APP.selected_peak_legend_label(behavior), "Selected Epc")
        self.assertIn("Only the selected Epc is used", APP.candidate_peak_explanation(behavior))
        self.assertIn(
            "no reliable anodic return peak",
            APP.candidate_rejection_reasons(behavior)[raw_oxidation.id],
        )

    def test_two_redox_couples_are_not_classified_irreversible(self) -> None:
        potential, current = two_redox_couples_trace()
        params = APP.auto_tune_cv_peak_parameters(potential, current)
        peaks = APP.detect_peaks(
            potential,
            current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
        )
        raw_oxidation, raw_reduction = APP.select_default_primary_peaks(peaks)
        self.assertIsNotNone(raw_oxidation)
        self.assertIsNotNone(raw_reduction)

        _raw_rows, raw_metrics = APP.build_peak_metrics_rows(
            raw_oxidation,
            raw_reduction,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        behavior = APP.classify_cv_behavior(raw_oxidation, raw_reduction, raw_metrics, potential, peaks)
        self.assertEqual(behavior.behavior, "multiple_redox_couples")
        self.assertEqual(APP.selected_peak_legend_label(behavior), "Selected redox couple")
        self.assertIn("Only the selected redox couple", APP.candidate_peak_explanation(behavior))
        self.assertGreaterEqual(len(behavior.redox_couples or []), 2)

        selected_couple = behavior.redox_couples[behavior.selected_couple_index or 0]
        self.assertAlmostEqual(selected_couple.oxidation_peak.potential, 0.5134, delta=0.02)
        self.assertAlmostEqual(selected_couple.oxidation_peak.raw_current * 1e6, 25.48, delta=1.0)
        self.assertAlmostEqual(selected_couple.reduction_peak.potential, 0.4333, delta=0.02)
        self.assertAlmostEqual(selected_couple.reduction_peak.raw_current * 1e6, -23.69, delta=1.5)
        self.assertAlmostEqual(selected_couple.delta_ep_v * 1000, 80.0, delta=10.0)
        self.assertAlmostEqual(selected_couple.formal_potential_v, 0.473, delta=0.02)
        self.assertAlmostEqual(selected_couple.current_ratio, 1.08, delta=0.15)
        option_label = APP.redox_couple_option_label(selected_couple)
        self.assertIn("E°", option_label)
        self.assertIn("ΔEp", option_label)
        self.assertIn("|Ipa/Ipc|", option_label)
        self.assertNotIn("Epa", option_label)
        self.assertEqual(
            list(APP.redox_couples_display_table(behavior.redox_couples, 0).columns),
            ["Couple", "Epa", "Epc", "ΔEp", "E°", "|Ipa/Ipc|", "Confidence", "Selected?"],
        )

        _rows, metrics = APP.build_peak_metrics_rows(
            behavior.oxidation_peak,
            behavior.reduction_peak,
            None,
            current,
            "uA",
            1.0,
            "Current",
        )
        self.assertIsNotNone(metrics["epc_V"])
        self.assertIsNotNone(metrics["ipc_A"])
        self.assertIsNotNone(metrics["delta_ep_V"])
        self.assertIsNotNone(metrics["ipa_ipc_ratio"])
        self.assertEqual(
            APP.cv_analysis_status(
                behavior.oxidation_peak,
                behavior.reduction_peak,
                metrics,
                potential,
                False,
                peaks,
                behavior,
            ),
            "Review recommended",
        )
        messages = APP.cv_analysis_quality_messages(
            behavior.oxidation_peak,
            behavior.reduction_peak,
            metrics,
            potential,
            peaks,
            behavior,
        )
        self.assertTrue(any("Multiple redox couples detected" in message for message in messages))
        self.assertFalse(any("Irreversible" in message for message in messages))

    def test_reversible_pair_metrics_are_tentative_when_pair_confidence_is_not_high(self) -> None:
        metrics = {
            "epa_V": 0.385,
            "ipa_A": 26.8e-6,
            "epc_V": 0.295,
            "ipc_A": -23.3e-6,
            "delta_ep_V": 0.090,
            "ipa_ipc_ratio": 1.15,
        }

        table = APP.metrics_table(metrics, "uA", 1.0, "Current", "quasi_reversible", "Medium")
        metric_values = dict(zip(table["metric"], table["value"]))
        metric_methods = dict(zip(table["metric"], table["method"]))

        self.assertEqual(metric_values["\u0394Ep"], "Tentative: 90 mV")
        self.assertEqual(metric_values["E\u00b0\u2032"], "Tentative: 0.34 V")
        self.assertEqual(metric_values["|Ipa/Ipc|"], "Tentative: 1.15")
        self.assertIn("pair confidence is Medium", metric_methods["\u0394Ep"])

    def test_reversible_pair_metrics_are_invalid_when_epa_is_lower_than_epc(self) -> None:
        metrics = {
            "epa_V": 0.200,
            "ipa_A": 12.0e-6,
            "epc_V": 0.300,
            "ipc_A": -10.0e-6,
            "delta_ep_V": -0.100,
            "ipa_ipc_ratio": 1.2,
        }

        table = APP.metrics_table(metrics, "uA", 1.0, "Current", "reversible_like", "High")
        metric_values = dict(zip(table["metric"], table["value"]))
        metric_methods = dict(zip(table["metric"], table["method"]))

        self.assertEqual(metric_values["\u0394Ep"], "Invalid")
        self.assertEqual(metric_values["E\u00b0\u2032"], "Invalid")
        self.assertEqual(metric_values["|Ipa/Ipc|"], "Invalid")
        self.assertIn("Epa is lower than Epc", metric_methods["\u0394Ep"])

    def test_sloped_baseline_background_adds_review_recommendation(self) -> None:
        potential, current = sloped_baseline_background_trace()
        baseline_suspected = APP.detect_sloped_baseline_background(potential, current)
        self.assertTrue(baseline_suspected)

        params = APP.auto_tune_cv_peak_parameters(potential, current)
        analysis = APP.analyze_cv_default_peaks(
            potential,
            current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
            "uA",
            1.0,
            "Current",
            baseline_background_suspected=baseline_suspected,
        )

        messages = APP.cv_analysis_quality_messages(
            analysis.oxidation_peak,
            analysis.reduction_peak,
            analysis.metrics,
            potential,
            analysis.candidate_peaks,
            analysis.behavior,
        )
        self.assertTrue(any(APP.SLOPED_BASELINE_WARNING in message for message in messages))
        if analysis.behavior.behavior in {"irreversible_oxidation_only", "irreversible_reduction_only"}:
            self.assertTrue(any("baseline drift may affect return-peak classification" in message for message in messages))

        baseline = APP.calculate_linear_baseline(potential, current, 0, int(np.nanargmax(potential)))
        self.assertIsNotNone(baseline)
        corrected_current = current - baseline
        corrected_params = APP.auto_tune_cv_peak_parameters(potential, corrected_current)
        corrected_analysis = APP.analyze_cv_default_peaks(
            potential,
            corrected_current,
            corrected_params["min_prominence"],
            int(corrected_params["min_distance"]),
            corrected_params["min_abs_current"],
            "uA",
            1.0,
            "Current",
        )
        self.assertIsNotNone(corrected_analysis.oxidation_peak)

    def test_reverse_scan_first_branch_assignment_keeps_reversible_pair(self) -> None:
        potential, current = reverse_scan_first_trace()
        params = APP.auto_tune_cv_peak_parameters(potential, current)
        analysis = APP.analyze_cv_default_peaks(
            potential,
            current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
            "uA",
            1.0,
            "Current",
        )

        self.assertIn(analysis.behavior.behavior, {"reversible_like", "quasi_reversible"})
        self.assertIsNotNone(analysis.oxidation_peak)
        self.assertIsNotNone(analysis.reduction_peak)
        self.assertAlmostEqual(analysis.metrics["epa_V"], 0.3852, delta=0.02)
        self.assertAlmostEqual(analysis.metrics["epc_V"], 0.295, delta=0.02)
        self.assertIsNotNone(analysis.metrics["delta_ep_V"])
        self.assertFalse(any("oxidation-only" in message for message in analysis.behavior.messages))

    def test_inferred_multicycle_selected_summary_matches_per_cycle_result(self) -> None:
        potential, current, cycle_values = multicycle_no_cycle_column_trace()
        cycles = APP.available_cycle_values(cycle_values)
        self.assertEqual(cycles, [1.0, 2.0, 3.0])

        for cycle_value in cycles:
            cycle_mask = APP.cycle_selection_mask(cycle_values, cycle_value, len(potential))
            cycle_potential = potential[cycle_mask]
            cycle_current = current[cycle_mask]
            params = APP.auto_tune_cv_peak_parameters(cycle_potential, cycle_current)
            selected_analysis = APP.analyze_cv_default_peaks(
                cycle_potential,
                cycle_current,
                params["min_prominence"],
                int(params["min_distance"]),
                params["min_abs_current"],
                "uA",
                1.0,
                "Current",
            )

            per_cycle_df, _trend_messages = APP.build_per_cycle_cv_metrics(
                potential,
                current,
                cycle_values,
                cycles,
                cycle_value,
                params["min_prominence"],
                int(params["min_distance"]),
                params["min_abs_current"],
                "uA",
                1.0,
                "Current",
                selected_analysis.metrics,
                selected_analysis.oxidation_peak,
                selected_analysis.reduction_peak,
                selected_analysis.behavior,
            )
            selected_row = per_cycle_df.loc[per_cycle_df["analyzed"] == "Yes"].iloc[0]
            self.assertEqual(selected_row["cycle"], APP.cycle_display_label(cycle_value))

            if selected_analysis.metrics["epa_V"] is not None:
                self.assertAlmostEqual(
                    float(selected_row["Epa (V)"]),
                    selected_analysis.metrics["epa_V"],
                    delta=1e-4,
                )
            if selected_analysis.metrics["epc_V"] is not None:
                self.assertAlmostEqual(
                    float(selected_row["Epc (V)"]),
                    selected_analysis.metrics["epc_V"],
                    delta=1e-4,
                )
            self.assertEqual(
                selected_row["analysis quality"],
                APP.cv_analysis_quality_label(
                    selected_analysis.oxidation_peak,
                    selected_analysis.reduction_peak,
                    selected_analysis.metrics,
                    cycle_potential,
                    cv_behavior=selected_analysis.behavior,
                ),
            )

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
        self.assertEqual(details["metadata_rows_before_header"], 0)
        self.assertEqual(details["header_row"], "Line 1")
        self.assertEqual(details["rows_skipped"], 0)
        self.assertIn("100", details["scan_rate_estimate"])
        self.assertAlmostEqual(
            APP.representative_scan_rate_v_s(time_s, potential),
            0.1,
            delta=0.002,
        )

    def test_parser_details_separate_metadata_header_and_numeric_data_rows(self) -> None:
        dataset = APP.ParsedDataset(
            filename="03_cv_tab_metadata_header_uA.txt",
            dataframe=pd.DataFrame(
                {
                    "Potential_V": [0.0, 0.1],
                    "Current_uA": [1.0, 2.0],
                }
            ),
            headers=["Potential_V", "Current_uA"],
            metadata={
                "Instrument": "Synthetic Potentiostat 3000",
                "Technique": "Cyclic Voltammetry",
                "Scan rate": "200 mV/s",
                "Electrode": "Glassy carbon",
            },
            delimiter="\t",
            has_header=True,
            header_row=6,
            data_start_row=7,
            rows_skipped=6,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_uA",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V", "current:Current_uA": "uA"},
            warnings=[],
        )

        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)

        self.assertEqual(details["metadata_rows_before_header"], 5)
        self.assertEqual(details["header_row"], "Line 6")
        self.assertEqual(details["rows_skipped"], 6)
        self.assertEqual(details["rows_imported"], 2)

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
        self.assertEqual(len(filtered), 7)
        self.assertEqual([peak.id for peak in filtered[2:]], [f"peak_{idx}" for idx in range(14, 9, -1)])

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

    def test_metadata_rows_for_display_separates_structured_and_unparsed_lines(self) -> None:
        parsed_rows, unparsed_rows = APP.metadata_rows_for_display(
            {
                "Instrument": "Synthetic Potentiostat 3000",
                "Technique": "Cyclic Voltammetry",
                "Scan rate": "200 mV/s",
                "Electrode": "Glassy carbon",
                "metadata_line_5": "Operator note without delimiter",
            }
        )

        self.assertEqual(
            parsed_rows,
            [
                {"Field": "Instrument", "Value": "Synthetic Potentiostat 3000"},
                {"Field": "Technique", "Value": "Cyclic Voltammetry"},
                {"Field": "Scan rate", "Value": "200 mV/s"},
                {"Field": "Electrode", "Value": "Glassy carbon"},
            ],
        )
        self.assertEqual(unparsed_rows, [{"Line": "5", "Text": "Operator note without delimiter"}])

    def test_delimiter_reporting_distinguishes_common_cv_file_formats(self) -> None:
        self.assertEqual(
            APP.detect_delimiter(
                [
                    "Instrument: Synthetic Potentiostat 3000",
                    "Potential/V\tCurrent/uA",
                    "0.000\t1.25",
                    "0.010\t1.42",
                ]
            ),
            "\t",
        )
        self.assertEqual(APP.display_delimiter(","), "comma")
        self.assertEqual(APP.display_delimiter(";"), "semicolon")
        self.assertEqual(APP.display_delimiter("\t"), "tab")

        old_saved_tab_file = APP.ParsedDataset(
            filename="03_cv_tab_metadata_header_uA.txt",
            dataframe=pd.DataFrame({"Potential_V": [0.0], "Current_uA": [1.0]}),
            headers=["Potential_V", "Current_uA"],
            metadata={},
            delimiter="whitespace",
            has_header=True,
            header_row=6,
            data_start_row=7,
            rows_skipped=6,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_uA",
            detected_time_col=None,
            detected_units={},
            warnings=[],
        )
        self.assertEqual(APP.display_delimiter_for_dataset(old_saved_tab_file), "tab / whitespace")

    def test_experiment_header_uses_detected_type_when_available(self) -> None:
        self.assertEqual(APP.effective_experiment_type({"experiment_type": "auto", "files": {}}), "auto")
        self.assertEqual(APP.experiment_header_type_label("auto"), "Auto")
        self.assertEqual(
            APP.effective_experiment_type(
                {
                    "experiment_type": "auto",
                    "analysis_outputs": {"experiment_type": "cv"},
                    "files": {},
                }
            ),
            "cv",
        )
        self.assertEqual(
            APP.effective_experiment_type(
                {
                    "experiment_type": "auto",
                    "files": {
                        "sample.csv": {
                            "summary": {
                                "experiment_type": "lsv",
                            }
                        }
                    },
                }
            ),
            "lsv",
        )
        self.assertEqual(APP.experiment_header_type_label("cv"), "CV")
        self.assertEqual(APP.experiment_header_type_label("cv", ["Mixed CV/LSV data"]), "Auto / needs review")

    def test_cycle_selection_defaults_and_masks_expected_cycle(self) -> None:
        cycle_values = np.array([1, 1, 2, 2, 3, 3], dtype=float)

        self.assertEqual(APP.available_cycle_values(cycle_values), [1.0, 2.0, 3.0])
        self.assertEqual(APP.cycle_display_label(APP.available_cycle_values(cycle_values)[-1]), "Cycle 3")
        np.testing.assert_array_equal(
            APP.cycle_selection_mask(cycle_values, 3.0, len(cycle_values)),
            np.array([False, False, False, False, True, True]),
        )

    def test_normalized_cv_dataset_preserves_cycle_values(self) -> None:
        dataset = APP.ParsedDataset(
            filename="04_multicycle_cv_comma.csv",
            dataframe=pd.DataFrame(
                {
                    "Potential_V": [0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
                    "Current_uA": [1.0, 2.0, 1.5, 2.5, 2.0, 3.0],
                    "Cycle": [1, 1, 2, 2, 3, 3],
                }
            ),
            headers=["Potential_V", "Current_uA", "Cycle"],
            metadata={},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_uA",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V", "current:Current_uA": "uA"},
            warnings=[],
        )

        analyzed = APP.normalize_dataset(dataset, "Potential_V", "Current_uA", "Auto", "Auto", "Current", "None", 11)

        self.assertEqual(APP.available_cycle_values(analyzed.cycle_values), [1.0, 2.0, 3.0])
        self.assertEqual(analyzed.cycle_source, "column")
        self.assertEqual(APP.analyzed_cycle_label(3.0, 3), "Cycle 3 of 3")
        self.assertEqual(APP.selected_cycle_trace_name(3.0, True), "Cycle 3 — selected")
        self.assertEqual(APP.selected_cycle_trace_name(2.0, False), "Cycle 2")
        self.assertEqual(
            APP.cycle_assignment_messages(analyzed, 3, "Cycle 3 of 3"),
            [
                (
                    "info",
                    "Multi-cycle CV detected: 3 cycles from cycle column. Currently analyzing: Cycle 3 of 3. "
                    "Metrics and peak markers use the selected cycle; plot may show all cycles.",
                )
            ],
        )
        np.testing.assert_array_equal(
            APP.cycle_selection_mask(analyzed.cycle_values, 2.0, len(analyzed.potential_v)),
            np.array([False, False, True, True, False, False]),
        )

    def test_normalized_cv_dataset_can_infer_cycle_values_from_turning_points(self) -> None:
        single_cycle_potential, single_cycle_current = clean_cv_standard_columns_trace()
        potential = np.concatenate([single_cycle_potential, single_cycle_potential, single_cycle_potential])
        current = np.concatenate([single_cycle_current, single_cycle_current * 1.1, single_cycle_current * 1.25])
        dataset = APP.ParsedDataset(
            filename="inferred_multicycle_cv.csv",
            dataframe=pd.DataFrame(
                {
                    "Potential_V": potential,
                    "Current_A": current,
                }
            ),
            headers=["Potential_V", "Current_A"],
            metadata={},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current_A",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V", "current:Current_A": "A"},
            warnings=[],
        )

        analyzed = APP.normalize_dataset(dataset, "Potential_V", "Current_A", "Auto", "Auto", "Current", "None", 11)

        self.assertEqual(APP.available_cycle_values(analyzed.cycle_values), [1.0, 2.0, 3.0])
        self.assertEqual(analyzed.cycle_source, "inferred")
        self.assertEqual(
            APP.cycle_assignment_messages(analyzed, 3, "Cycle 3 of 3"),
            [
                (
                    "warning",
                    "Multi-cycle CV detected: 3 cycles inferred from potential turning points. "
                    "Currently analyzing: Cycle 3 of 3. Metrics and peak markers use the selected cycle; "
                    "plot may show all cycles. Cycle splitting may be approximate.",
                )
            ],
        )
        first_cycle_mask = APP.cycle_selection_mask(analyzed.cycle_values, 1.0, len(analyzed.potential_v))
        first_cycle_potential = analyzed.potential_v[first_cycle_mask]
        first_cycle_current = analyzed.raw_current_a[first_cycle_mask]
        params = APP.auto_tune_cv_peak_parameters(first_cycle_potential, first_cycle_current)
        peaks = APP.detect_peaks(
            first_cycle_potential,
            first_cycle_current,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
        )
        oxidation_peak, reduction_peak = APP.select_default_primary_peaks(peaks)
        _rows, metrics = APP.build_peak_metrics_rows(
            oxidation_peak,
            reduction_peak,
            None,
            first_cycle_current,
            "uA",
            1.0,
            "Current",
        )
        self.assertEqual(
            APP.cv_analysis_status(
                oxidation_peak,
                reduction_peak,
                metrics,
                first_cycle_potential,
                cycle_assignment_uncertain=True,
            ),
            "Review recommended",
        )

    def test_per_cycle_cv_metrics_has_one_row_per_cycle_and_marks_selected_cycle(self) -> None:
        single_cycle_potential, single_cycle_current = clean_cv_standard_columns_trace()
        potential = np.concatenate([single_cycle_potential, single_cycle_potential, single_cycle_potential])
        current = np.concatenate(
            [
                single_cycle_current,
                single_cycle_current * 1.1,
                single_cycle_current * 1.25,
            ]
        )
        cycle_values = np.concatenate(
            [
                np.ones_like(single_cycle_potential),
                np.full_like(single_cycle_potential, 2),
                np.full_like(single_cycle_potential, 3),
            ]
        )
        params = APP.auto_tune_cv_peak_parameters(single_cycle_potential, single_cycle_current)

        per_cycle_df, trend_messages = APP.build_per_cycle_cv_metrics(
            potential,
            current,
            cycle_values,
            [1.0, 2.0, 3.0],
            3.0,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
            "uA",
            1.0,
            "Current",
        )

        self.assertEqual(len(per_cycle_df), 3)
        self.assertEqual(per_cycle_df["cycle"].tolist(), ["Cycle 1", "Cycle 2", "Cycle 3"])
        selected_row = per_cycle_df.loc[per_cycle_df["analyzed"] == "Yes"].iloc[0]
        self.assertEqual(selected_row["cycle"], "Cycle 3")
        self.assertAlmostEqual(float(selected_row["Epa (V)"]), 0.3852, delta=0.01)
        self.assertAlmostEqual(float(selected_row["Epc (V)"]), 0.295, delta=0.01)
        self.assertEqual(selected_row["analysis quality"], "High")
        self.assertEqual(len(trend_messages), 3)

    def test_ambiguous_current_unit_confirmation_recalculates_cv_scale(self) -> None:
        potential, current_a = clean_cv_standard_columns_trace()
        current_microamp_values = current_a * 1e6
        dataset = APP.ParsedDataset(
            filename="08_messy_excel_export_cv.csv",
            dataframe=pd.DataFrame(
                {
                    "Potential_V": potential,
                    "Current": current_microamp_values,
                }
            ),
            headers=["Potential_V", "Current"],
            metadata={},
            delimiter=",",
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=1,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Potential_V",
            detected_current_col="Current",
            detected_time_col=None,
            detected_units={"potential:Potential_V": "V"},
            warnings=[],
        )

        self.assertIsNone(APP.detect_unit_from_header("Current", "current"))
        self.assertIsNone(APP.detect_unit_from_header("I", "current"))
        self.assertIsNone(APP.detect_unit_from_header("signal", "current"))
        self.assertEqual(APP.detect_unit_from_header("Current_A", "current"), "A")
        self.assertEqual(APP.detect_unit_from_header("j_mA_cm2", "current"), "mA")
        self.assertTrue(APP.current_unit_is_ambiguous(dataset, "Current", "Auto"))
        self.assertEqual(APP.suggested_current_unit_for_values(current_microamp_values), "uA")

        unconfirmed_details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)
        self.assertEqual(unconfirmed_details["current_unit_confidence"], "Low")
        self.assertEqual(unconfirmed_details["current_unit_confidence_detail"], "Column label is ambiguous")
        self.assertEqual(unconfirmed_details["suggested_current_unit"], "µA based on value magnitude")
        self.assertEqual(
            APP.current_unit_review_explanation(
                "Current",
                "uA",
                unconfirmed_details["current_unit_warning"],
            ),
            "The current column has no unit in the header. If interpreted as A, "
            "the current values are unusually large for a typical CV/LSV.",
        )
        self.assertEqual(
            APP.current_unit_review_reason("uA", unconfirmed_details["current_unit_warning"]),
            "Suggested based on value magnitude; interpreting as A gives unusually large currents.",
        )
        self.assertEqual(
            unconfirmed_details["unit_header_warning"],
            "Column units missing from header. Please confirm potential/current units.",
        )
        self.assertIn("Current unit may be wrong", unconfirmed_details["current_unit_warning"])
        self.assertEqual(
            unconfirmed_details["current_magnitude_warning"],
            "Current magnitude is unusually large. Check whether the current unit was inferred correctly.",
        )
        self.assertIn("raw values", unconfirmed_details["original_current_values"])
        self.assertEqual(APP.parser_confidence_from_details(unconfirmed_details), "High")
        self.assertEqual(
            APP.overall_analysis_status(
                "Passed",
                unconfirmed_details["current_unit_confidence"],
                APP.parser_confidence_from_details(unconfirmed_details),
                unconfirmed_details["current_unit_warning"] or unconfirmed_details["current_magnitude_warning"],
            ),
            "Review needed",
        )

        confirmed_details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0, "uA")
        self.assertEqual(confirmed_details["current_unit_confidence"], "Confirmed by user")
        self.assertEqual(confirmed_details["current_unit_confidence_detail"], "Confirmed by user")
        self.assertIn("µA", confirmed_details["current_range"])
        self.assertIn("µA", confirmed_details["raw_current_range"])
        self.assertIn("A", confirmed_details["analysis_current_range"])
        self.assertEqual(confirmed_details["unit_header_warning"], "")
        self.assertEqual(
            confirmed_details["unit_header_audit"],
            "Column units were missing from header. Current unit was confirmed by user as µA.",
        )
        self.assertEqual(confirmed_details["current_magnitude_warning"], "")
        self.assertEqual(
            APP.overall_analysis_status(
                "Passed",
                confirmed_details["current_unit_confidence"],
                APP.parser_confidence_from_details(confirmed_details),
                confirmed_details["current_unit_warning"],
            ),
            "Passed",
        )

        confirmed_amp_details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0, "A")
        self.assertEqual(confirmed_amp_details["current_unit_confidence"], "Confirmed by user")
        self.assertEqual(
            confirmed_amp_details["unit_header_audit"],
            "Column units were missing from header. Current unit was confirmed by user as A.",
        )
        self.assertEqual(
            confirmed_amp_details["current_magnitude_warning"],
            "Current magnitude is unusually large. Check whether the current unit was inferred correctly.",
        )
        self.assertEqual(
            APP.overall_analysis_status(
                "Passed",
                confirmed_amp_details["current_unit_confidence"],
                APP.parser_confidence_from_details(confirmed_amp_details),
                confirmed_amp_details["current_magnitude_warning"],
            ),
            "Review needed",
        )
        self.assertEqual(
            APP.overall_analysis_status(
                "Passed",
                confirmed_amp_details["current_unit_confidence"],
                APP.parser_confidence_from_details(confirmed_amp_details),
                "",
            ),
            "Passed",
        )

        analyzed = APP.normalize_dataset(dataset, "Potential_V", "Current", "Auto", "uA", "Current", "None", 11)
        params = APP.auto_tune_cv_peak_parameters(analyzed.potential_v, analyzed.raw_current_a)
        peaks = APP.detect_peaks(
            analyzed.potential_v,
            analyzed.raw_current_a,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
        )
        oxidation_peak, reduction_peak = APP.select_default_primary_peaks(peaks)
        _rows, metrics = APP.build_peak_metrics_rows(
            oxidation_peak,
            reduction_peak,
            None,
            analyzed.raw_current_a,
            "uA",
            1.0,
            "Current",
        )

        ipa_uA = APP.current_to_display(np.array([metrics["ipa_A"]]), "uA", "Current", 1.0)[0]
        ipc_uA = APP.current_to_display(np.array([metrics["ipc_A"]]), "uA", "Current", 1.0)[0]
        self.assertAlmostEqual(float(ipa_uA), 26.8, delta=0.5)
        self.assertAlmostEqual(float(ipc_uA), -23.28, delta=0.5)

        compact_metrics = APP.metrics_table(metrics, "uA", 1.0, "Current")
        current_dependent_metrics = compact_metrics["metric"].isin(["Ipa", "Ipc", "|Ipa/Ipc|"])
        compact_metrics.loc[current_dependent_metrics, "method"] = (
            compact_metrics.loc[current_dependent_metrics, "method"]
            + "; current unit needs confirmation"
        )
        flagged_methods = compact_metrics.loc[current_dependent_metrics, "method"].tolist()
        self.assertTrue(all("current unit needs confirmation" in method for method in flagged_methods))

    def test_messy_excel_export_preserves_metadata_and_flags_missing_units(self) -> None:
        dataset = APP.ParsedDataset(
            filename="08_messy_excel_export_cv.csv",
            dataframe=pd.DataFrame(
                {
                    "Voltage": [-0.2, 0.0, 0.4, 0.8],
                    "Current": [-29.88, -1.0, 33.83, 2.0],
                }
            ),
            headers=["Voltage", "Current"],
            metadata={"User": "student_a", "metadata_line_2": "Exported from Lab PC"},
            delimiter=",",
            has_header=True,
            header_row=3,
            data_start_row=4,
            rows_skipped=3,
            rows_dropped=0,
            missing_values=0,
            detected_potential_col="Voltage",
            detected_current_col="Current",
            detected_time_col=None,
            detected_units={"potential:Voltage": "V"},
            warnings=[],
        )

        detected = APP.detect_experiment_columns(dataset, "cv")
        parsed_metadata_rows, unparsed_metadata_rows = APP.metadata_rows_for_display(dataset.metadata)
        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)
        confirmed_details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0, "uA")

        self.assertEqual(detected["potential"], "Voltage")
        self.assertEqual(detected["current"], "Current")
        self.assertIn({"Field": "User", "Value": "student_a"}, parsed_metadata_rows)
        self.assertEqual(unparsed_metadata_rows, [{"Line": "2", "Text": "Exported from Lab PC"}])
        self.assertEqual(details["potential_range"], "-0.200 to 0.800 V")
        self.assertEqual(details["current_unit_confidence"], "Low")
        self.assertEqual(details["suggested_current_unit"], "µA based on value magnitude")
        self.assertEqual(
            details["unit_header_warning"],
            "Column units missing from header. Please confirm potential/current units.",
        )
        self.assertIn("Current unit may be wrong", details["current_unit_warning"])
        self.assertEqual(
            details["current_magnitude_warning"],
            "Current magnitude is unusually large. Check whether the current unit was inferred correctly.",
        )
        self.assertEqual(confirmed_details["current_unit_confidence"], "Confirmed by user")
        self.assertEqual(confirmed_details["unit_header_warning"], "")
        self.assertEqual(
            confirmed_details["unit_header_audit"],
            "Column units were missing from header. Current unit was confirmed by user as µA.",
        )
        self.assertEqual(confirmed_details["current_range"], "-29.88 to 33.83 µA")

    def test_missing_values_no_header_uses_plausible_amp_review_copy_and_inferred_labels(self) -> None:
        dataset = APP.ParsedDataset(
            filename="09_cv_missing_values.csv",
            dataframe=pd.DataFrame(
                {
                    "-0.2": [-0.2, 0.0, 0.4, 0.8],
                    "Column 2": [-22.28e-6, -2.0e-6, 25.57e-6, 1.0e-6],
                }
            ),
            headers=["-0.2", "Column 2"],
            metadata={"metadata_line_1": "Malformed header export"},
            delimiter=",",
            has_header=True,
            header_row=2,
            data_start_row=3,
            rows_skipped=2,
            rows_dropped=25,
            missing_values=0,
            detected_potential_col="-0.2",
            detected_current_col="Column 2",
            detected_time_col=None,
            detected_units={},
            warnings=[],
        )

        self.assertEqual(APP.user_facing_column_label(dataset, "-0.2"), "Inferred column 1")
        self.assertEqual(APP.user_facing_column_label(dataset, "Column 2"), "Inferred column 2")
        self.assertEqual(
            APP.suggested_current_unit_for_values(dataset.dataframe["Column 2"].to_numpy(dtype=float)),
            "A",
        )

        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)
        self.assertEqual(details["potential_column"], "Inferred column 1")
        self.assertEqual(details["current_column"], "Inferred column 2")
        self.assertEqual(details["raw_potential_column"], "-0.2")
        self.assertEqual(details["raw_current_column"], "Column 2")
        self.assertEqual(details["current_unit_confidence"], "Low")
        self.assertEqual(details["suggested_current_unit"], "A based on value magnitude")
        self.assertEqual(details["current_unit_warning"], "")
        self.assertEqual(details["current_magnitude_warning"], "")
        self.assertEqual(
            APP.current_unit_review_explanation(
                details["current_column"],
                "A",
                details["current_unit_warning"],
            ),
            "The current column has no unit in the header. Based on value magnitude, "
            "A is the most likely input unit. Please confirm before using current-dependent metrics.",
        )
        self.assertEqual(
            APP.current_unit_review_reason("A", details["current_unit_warning"]),
            "Suggested based on value magnitude; A gives microamp-scale currents.",
        )

        confirmed_details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0, "A")
        self.assertEqual(confirmed_details["current_unit_confidence"], "Confirmed by user")
        self.assertEqual(confirmed_details["unit_header_warning"], "")
        self.assertEqual(confirmed_details["current_range"], "-22.28 to 25.57 µA")

    def test_clean_cv_parser_summary_has_no_cleanup_warning(self) -> None:
        potential, current = clean_cv_standard_columns_trace()
        lines = ["Potential_V,Current_A"]
        lines.extend(f"{potential_value:.8f},{current_value:.12e}" for potential_value, current_value in zip(potential, current))

        dataset = APP.parse_electrochem_file(FakeUploadedFile("01_clean_cv_standard_columns.csv", "\n".join(lines)))
        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)

        self.assertEqual(dataset.rows_dropped, 0)
        self.assertEqual(details["rows_dropped"], 0)
        self.assertEqual(details["rows_dropped_reason"], "None")
        self.assertEqual(details["parser_cleanup_severity"], "none")
        self.assertEqual(details["warnings"], "None")
        self.assertEqual(APP.parser_confidence_from_details(details), "High")

    def test_bad_numeric_rows_are_visible_cleanup_warning_without_failing_parse(self) -> None:
        potential, current = clean_cv_standard_columns_trace()
        lines = ["Potential_V,Current_A"]
        corrupt_after_indices = {120, 520, 880}
        for idx, (potential_value, current_value) in enumerate(zip(potential, current)):
            if idx in corrupt_after_indices:
                lines.append("corrupted row inside numeric data,not a number")
            lines.append(f"{potential_value:.8f},{current_value:.12e}")

        dataset = APP.parse_electrochem_file(
            FakeUploadedFile("23_bad_file_text_inside_numeric_data.csv", "\n".join(lines))
        )
        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)
        analyzed = APP.normalize_dataset(dataset, "Potential_V", "Current_A", "Auto", "Auto", "Current", "None", 11)
        params = APP.auto_tune_cv_peak_parameters(analyzed.potential_v, analyzed.raw_current_a)
        analysis = APP.analyze_cv_default_peaks(
            analyzed.potential_v,
            analyzed.raw_current_a,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
            "uA",
            1.0,
            "Current",
        )

        self.assertEqual(len(dataset.dataframe), 1000)
        self.assertEqual(dataset.rows_dropped, 3)
        self.assertEqual(details["rows_dropped"], 3)
        self.assertEqual(details["parser_cleanup_severity"], "cleanup")
        self.assertEqual(details["rows_dropped_fraction"], "0.30%")
        self.assertEqual(details["rows_dropped_reason"], "non-numeric or corrupted values inside numeric data region.")
        self.assertIn("3 non-numeric data rows were removed during cleanup.", details["warnings"])
        self.assertNotEqual(details["warnings"], "None")
        self.assertIn("3 non-numeric data rows were removed during cleanup.", dataset.warnings)
        self.assertEqual(APP.parser_confidence_from_details(details), "High")
        self.assertEqual(analyzed.potential_v.size, 1000)
        self.assertIsNotNone(analysis.oxidation_peak)
        self.assertIsNotNone(analysis.reduction_peak)

    def test_many_bad_numeric_rows_downgrades_parser_status_to_review_needed(self) -> None:
        potential = np.linspace(-0.2, 0.8, 100)
        current = 2.0e-6 * np.sin(np.linspace(0.0, 4.0 * np.pi, 100))
        lines = ["Potential_V,Current_A"]
        for idx, (potential_value, current_value) in enumerate(zip(potential, current)):
            lines.append(f"{potential_value:.8f},{current_value:.12e}")
            if idx % 10 == 9:
                lines.append("bad current row,not numeric")

        dataset = APP.parse_electrochem_file(FakeUploadedFile("too_many_bad_rows.csv", "\n".join(lines)))
        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)
        parser_confidence = APP.parser_confidence_from_details(details)

        self.assertEqual(len(dataset.dataframe), 100)
        self.assertEqual(dataset.rows_dropped, 10)
        self.assertEqual(details["parser_cleanup_severity"], "strong")
        self.assertEqual(details["rows_dropped_fraction"], "9.09%")
        self.assertIn("Strong cleanup warning", details["warnings"])
        self.assertEqual(parser_confidence, "Medium")
        self.assertEqual(
            APP.overall_analysis_status("Passed", "High", parser_confidence, ""),
            "Review needed",
        )

    def test_no_numeric_table_fails_with_clear_parse_error(self) -> None:
        bad_file = FakeUploadedFile(
            "22_no_numeric_table.csv",
            "Instrument: broken export\nThis file has no usable numeric table\nnotes only",
        )

        with self.assertRaisesRegex(ValueError, "Could not find a numeric data table"):
            APP.parse_electrochem_file(bad_file)

    def test_single_numeric_column_fails_with_clear_parse_error(self) -> None:
        bad_file = FakeUploadedFile(
            "single_numeric_column.csv",
            "Potential_V\n-0.200\n0.000\n0.200\n0.400",
        )

        with self.assertRaisesRegex(ValueError, "at least two usable numeric columns"):
            APP.parse_electrochem_file(bad_file)

    def test_cv_millivolt_nanoamp_units_convert_for_summary_details_and_analysis(self) -> None:
        forward = np.linspace(-0.2, 0.8, 500)
        reverse = np.linspace(0.8, -0.2, 500)
        potential_v = np.concatenate([forward, reverse])
        current_uA = np.concatenate(
            [
                0.05
                + 0.08 * forward
                + 0.78225 * np.exp(-0.5 * ((forward - 0.3772) / 0.045) ** 2),
                -0.4
                - 2.41613 * np.exp(-0.5 * ((reverse - 0.15) / 0.18) ** 2)
                + 0.05 * reverse,
            ]
        )
        lines = ["Potential_mV,Current_nA"]
        lines.extend(
            f"{potential_value * 1000.0:.6f},{current_value * 1000.0:.6f}"
            for potential_value, current_value in zip(potential_v, current_uA)
        )

        dataset = APP.parse_electrochem_file(FakeUploadedFile("24_units_mV_nA.csv", "\n".join(lines)))
        detected_type, detection_reasons, detection_warnings = APP.detect_uploaded_experiment_type([dataset])
        details = APP.parser_summary_details(dataset, "cv", "uA", "Current", 1.0)
        analyzed = APP.normalize_dataset(dataset, "Potential_mV", "Current_nA", "Auto", "Auto", "Current", "None", 11)
        params = APP.auto_tune_cv_peak_parameters(analyzed.potential_v, analyzed.raw_current_a)
        analysis = APP.analyze_cv_default_peaks(
            analyzed.potential_v,
            analyzed.raw_current_a,
            params["min_prominence"],
            int(params["min_distance"]),
            params["min_abs_current"],
            "uA",
            1.0,
            "Current",
        )

        self.assertEqual(detected_type, "cv")
        self.assertIn("scan-direction change", detection_reasons["24_units_mV_nA.csv"])
        self.assertEqual(detection_warnings, [])
        self.assertEqual(len(dataset.dataframe), 1000)
        self.assertEqual(dataset.rows_dropped, 0)
        self.assertEqual(details["current_unit_confidence"], "High")
        self.assertEqual(APP.parser_confidence_from_details(details), "High")
        self.assertEqual(details["warnings"], "None")
        self.assertEqual(details["potential_range"], "-0.200 to 0.800 V")
        self.assertEqual(details["current_range"], "-2.81 to 0.86 µA")
        self.assertEqual(details["raw_potential_range"], "-200 to 800 mV")
        self.assertEqual(details["raw_current_range"], "-2808.63 to 862.422 nA")
        self.assertEqual(details["analysis_current_range"], "-2.80863e-06 to 8.62422e-07 A")
        self.assertEqual(details["raw_current_unit"], "nA")
        self.assertEqual(details["analysis_current_unit"], "A")
        self.assertEqual(details["display_current_unit"], "µA")
        self.assertAlmostEqual(float(analyzed.potential_v.min()), -0.2, places=6)
        self.assertAlmostEqual(float(analyzed.potential_v.max()), 0.8, places=6)
        self.assertAlmostEqual(float(analyzed.raw_current_a.min()), -2.808631865e-6, delta=1e-12)
        self.assertAlmostEqual(float(analyzed.raw_current_a.max()), 8.62421941e-7, delta=1e-12)
        self.assertEqual(analysis.behavior.behavior, "irreversible_oxidation_only")
        self.assertIsNotNone(analysis.oxidation_peak)
        self.assertIsNone(analysis.reduction_peak)
        self.assertAlmostEqual(float(analysis.metrics["epa_V"]), 0.3772, delta=0.002)
        ipa_uA = APP.current_to_display(np.array([analysis.metrics["ipa_A"]]), "uA", "Current", 1.0)[0]
        self.assertAlmostEqual(float(ipa_uA), 0.8624, delta=0.005)


if __name__ == "__main__":
    unittest.main()
