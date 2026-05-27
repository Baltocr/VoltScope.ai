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
        self.assertEqual(metric_values["Max current density"], "148.2 µA/cm²")
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
        self.assertEqual(detailed_methods["Threshold"], "Auto-selected as 10% of maximum cathodic current magnitude.")
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


if __name__ == "__main__":
    unittest.main()
