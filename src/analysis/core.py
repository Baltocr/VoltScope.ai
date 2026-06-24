"""Stable import surface for the current VoltScope analysis engine."""

from __future__ import annotations

from src.utils.legacy_app import load_legacy_app


_APP = load_legacy_app()

parse_electrochem_file = _APP.parse_electrochem_file
detect_uploaded_experiment_type = _APP.detect_uploaded_experiment_type
normalize_dataset = _APP.normalize_dataset
analyze_cv_default_peaks = _APP.analyze_cv_default_peaks
build_lsv_analysis_summary = _APP.build_lsv_analysis_summary
build_lsv_detailed_metrics = _APP.build_lsv_detailed_metrics
analyze_lsv_onset = _APP.analyze_lsv_onset
parser_summary_details = _APP.parser_summary_details

__all__ = [
    "parse_electrochem_file",
    "detect_uploaded_experiment_type",
    "normalize_dataset",
    "analyze_cv_default_peaks",
    "build_lsv_analysis_summary",
    "build_lsv_detailed_metrics",
    "analyze_lsv_onset",
    "parser_summary_details",
]
