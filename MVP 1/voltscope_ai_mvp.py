"""
VoltScope AI
------------
A Streamlit app for electrochemistry data analysis.

Current features:
- Upload messy CSV/TXT/TSV/DAT cyclic voltammetry files
- Detect delimiters, metadata rows, table headers, columns, and units
- Manually correct potential/current columns and units
- Normalize potential to V and current to A internally
- Plot interactive current vs potential voltammograms
- Export Plotly figures and analysis tables
- Smooth current data, detect oxidation/reduction peaks, and calculate peak metrics
- Apply editable linear baseline correction using anchor point controls

Run locally:
    pip install streamlit pandas numpy plotly
    streamlit run voltscope_ai_mvp.py
"""

import io
import html
import hashlib
import importlib.util
import json
import logging
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


POTENTIAL_UNITS_TO_V = {"V": 1.0, "mV": 1e-3}
CURRENT_UNITS_TO_A = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9, "pA": 1e-12}
TIME_UNITS_TO_S = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}
FREQUENCY_UNITS_TO_HZ = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}
IMPEDANCE_UNITS_TO_OHM = {"ohm": 1.0, "kohm": 1e3, "Mohm": 1e6}
PHASE_UNITS_TO_DEG = {"deg": 1.0, "rad": 180.0 / np.pi}
CURRENT_MAGNITUDE_WARNING_A = 0.1
CURRENT_MAGNITUDE_WARNING_UA = 1_000_000.0
PARSER_DROPPED_ROWS_REVIEW_RATIO = 0.01
PARSER_DROPPED_ROWS_STRONG_RATIO = 0.05
WORKSPACE_STORE_PATH = Path(__file__).with_name("voltscope_project_workspaces.json")


EXPERIMENT_TYPES = {
    "auto": {
        "label": "Auto-detect from uploaded data",
        "short_label": "Auto",
        "description": "VoltScope detects whether the uploaded data is CV or LSV.",
    },
    "cv": {
        "label": "Cyclic voltammetry (CV)",
        "short_label": "CV",
        "description": "Potential-current scans with CV peak detection, ESW, and baseline correction.",
    },
    "lsv": {
        "label": "Linear sweep voltammetry (LSV)",
        "short_label": "LSV",
        "description": "Linear sweep voltammetry data with threshold-based ESW analysis.",
    },
    "ca": {
        "label": "Chronoamperometry (CA)",
        "short_label": "CA",
        "description": "Current vs time data with steady-state current and charge estimates.",
    },
    "cp": {
        "label": "Chronopotentiometry (CP)",
        "short_label": "CP",
        "description": "Potential vs time data with drift/slope summaries.",
    },
    "eis": {
        "label": "Electrochemical impedance spectroscopy (EIS)",
        "short_label": "EIS",
        "description": "Impedance data with Nyquist and optional Bode plots.",
    },
}
@dataclass
class ESWResult:
    cathodic_limit: Optional[float]
    anodic_limit: Optional[float]
    esw: Optional[float]


@dataclass
class ParsedDataset:
    filename: str
    dataframe: pd.DataFrame
    headers: list[str]
    metadata: dict[str, str]
    delimiter: str
    has_header: bool
    header_row: Optional[int]
    data_start_row: int
    rows_skipped: int
    rows_dropped: int
    missing_values: int
    detected_potential_col: Optional[str]
    detected_current_col: Optional[str]
    detected_time_col: Optional[str]
    detected_units: dict[str, str]
    warnings: list[str]


@dataclass
class Peak:
    id: str
    peak_type: str
    index: int
    potential: float
    raw_current: float
    prominence: float
    segment_index: int
    segment_direction: str
    confidence: str


@dataclass
class RedoxCouple:
    id: str
    oxidation_peak: Peak
    reduction_peak: Peak
    delta_ep_v: float
    formal_potential_v: float
    current_ratio: float
    score: float
    confidence: str
    reason: str


@dataclass
class CVBehaviorResult:
    behavior: str
    label: str
    oxidation_peak: Optional[Peak]
    reduction_peak: Optional[Peak]
    rejected_oxidation_peak: Optional[Peak]
    rejected_reduction_peak: Optional[Peak]
    pair_confidence: str
    messages: list[str]
    redox_couples: Optional[list[RedoxCouple]] = None
    selected_couple_index: Optional[int] = None


@dataclass
class CVPeakAnalysis:
    candidate_peaks: list[Peak]
    oxidation_peak: Optional[Peak]
    reduction_peak: Optional[Peak]
    peak_rows: list[dict[str, Any]]
    metrics: dict[str, Optional[float]]
    behavior: CVBehaviorResult


@dataclass
class LSVOnsetResult:
    onset_potential: Optional[float]
    onset_direction: str
    status: str
    confidence: str
    method: str
    method_note: str
    warning: str
    threshold_base: Optional[float]
    signed_threshold_base: Optional[float]
    sustained_points_required: int
    sustained_window_v: float
    smoothing_method: str
    smoothing_window: int
    rejected_potential: Optional[float] = None
    rejected_reason: str = ""


@dataclass
class AIInterpretationResponse:
    content: str
    error: str
    configured: bool
    model: str = "Not available"


@dataclass
class AnalyzedDataset:
    filename: str
    sample_name: str
    potential_col: str
    current_col: str
    potential_unit: str
    current_unit: str
    current_source: str
    potential_v: np.ndarray
    raw_current_a: np.ndarray
    smoothed_current_a: Optional[np.ndarray]
    cycle_values: Optional[np.ndarray]
    cycle_source: str
    metadata: dict[str, str]
    warnings: list[str]


# ---------- Parsing helpers ----------


def clean_label(label: Any) -> str:
    return str(label).strip().strip('"').strip("'").strip().strip(",;")


def normalize_label(label: Any) -> str:
    text = clean_label(label).lower()
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")
    text = re.sub(r"\s+", "", text)
    return text


_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_number(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    text = text.replace("\u2212", "-").replace("\u00b5", "u").replace("\u03bc", "u")
    match = _NUMBER_RE.search(text)
    if match is None:
        return np.nan

    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return [token.strip() for token in re.split(r"\s+", line.strip()) if token.strip()]
    return [token.strip() for token in line.strip().split(delimiter)]


def numeric_count(tokens: list[str]) -> int:
    return sum(np.isfinite(parse_number(token)) for token in tokens)


def is_numeric_row(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    count = numeric_count(tokens)
    return count >= 2 and count >= max(2, len(tokens) // 2)


def has_tabular_numeric_rows(lines: list[str]) -> bool:
    for line in lines[:120]:
        if "\t" not in line:
            continue
        tokens = split_line(line, "\t")
        if len(tokens) >= 2 and numeric_count(tokens) >= 2:
            return True
    return False


def detect_delimiter(lines: list[str]) -> str:
    if has_tabular_numeric_rows(lines):
        return "\t"

    candidates = [",", "\t", ";", "whitespace"]
    best_delimiter = ","
    best_score = -1

    for delimiter in candidates:
        score = 0
        column_counts = []
        for line in lines[:120]:
            if not line.strip():
                continue
            tokens = split_line(line, delimiter)
            if len(tokens) < 2:
                continue
            count = numeric_count(tokens)
            has_letters = any(re.search(r"[A-Za-z]", token) for token in tokens)
            if count >= 2:
                score += 4
                column_counts.append(len(tokens))
            elif has_letters:
                score += 1
                column_counts.append(len(tokens))
        if column_counts:
            common_count = max(set(column_counts), key=column_counts.count)
            score += column_counts.count(common_count)
        if score > best_score:
            best_score = score
            best_delimiter = delimiter

    return best_delimiter


def deduplicate_headers(headers: list[str]) -> list[str]:
    cleaned = [clean_label(header) or f"Column {idx + 1}" for idx, header in enumerate(headers)]
    seen: dict[str, int] = {}
    unique = []
    for header in cleaned:
        count = seen.get(header, 0)
        if count:
            unique.append(f"{header}_{count + 1}")
        else:
            unique.append(header)
        seen[header] = count + 1
    return unique


def parse_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for idx, line in enumerate(lines):
        cleaned = line.strip().strip("#").strip()
        if not cleaned:
            continue
        separator = ":" if ":" in cleaned else "=" if "=" in cleaned else None
        if separator is None:
            metadata[f"metadata_line_{idx + 1}"] = cleaned
            continue
        key, value = cleaned.split(separator, 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            metadata[key] = value
    return metadata


def parse_manual_metadata_text(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for idx, line in enumerate(str(text or "").splitlines()):
        cleaned = line.strip().strip("#").strip()
        if not cleaned:
            continue
        separator = ":" if ":" in cleaned else "=" if "=" in cleaned else None
        if separator is not None:
            key, value = cleaned.split(separator, 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                metadata[key] = value
            continue
        if parse_scan_rate_value(cleaned) is not None and not any(
            "scan" in normalize_label(key) and "rate" in normalize_label(key)
            for key in metadata
        ):
            metadata["Manual scan rate"] = cleaned
        else:
            metadata[f"manual_metadata_line_{idx + 1}"] = cleaned
    return metadata


def metadata_rows_for_display(metadata: dict[str, str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    parsed_rows: list[dict[str, str]] = []
    unparsed_rows: list[dict[str, str]] = []
    for key, value in metadata.items():
        if key.startswith("metadata_line_") or key.startswith("manual_metadata_line_"):
            line_number = key.rsplit("_", 1)[-1]
            unparsed_rows.append({"Line": line_number, "Text": value})
        else:
            parsed_rows.append({"Field": key, "Value": value})
    return parsed_rows, unparsed_rows


MANUAL_METADATA_FIELDS: list[tuple[str, str, str]] = [
    ("sample_name", "Sample name", "Sample or file label"),
    ("reference_electrode", "Reference electrode", "SCE, Ag/AgCl, RHE..."),
    ("working_electrode", "Working electrode", "Glassy carbon, Pt, Au..."),
    ("counter_electrode", "Counter electrode", "Pt wire, graphite rod..."),
    ("electrode_area", "Electrode area", "0.196 cm²"),
    ("electrolyte", "Electrolyte", "0.6 M H2SO4"),
    ("concentration", "Concentration", "10 mM analyte"),
    ("scan_rate", "Scan rate", "100 mV/s"),
    ("instrument", "Instrument", "Potentiostat model"),
    ("technique", "Technique", "Cyclic Voltammetry, LSV..."),
]


def metadata_semantic_key(key: Any) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", normalize_label(str(key or "")))
    if compact in {"samplename", "sample", "sampleid", "filename"}:
        return "sample_name"
    if compact in {"reference", "ref", "referenceelectrode", "refelectrode"} or (
        "reference" in compact and "electrode" in compact
    ):
        return "reference_electrode"
    if compact in {"workingelectrode", "working", "we"} or (
        "working" in compact and "electrode" in compact
    ):
        return "working_electrode"
    if compact in {"counterelectrode", "counter", "ce", "auxiliaryelectrode"} or (
        ("counter" in compact or "auxiliary" in compact) and "electrode" in compact
    ):
        return "counter_electrode"
    if compact in {"electrodearea", "surfacearea", "geometricarea", "electrodeareacm2"} or (
        "area" in compact and "electrode" in compact
    ):
        return "electrode_area"
    if compact in {"electrolyte", "solution"}:
        return "electrolyte"
    if compact in {"concentration", "molarity", "molality"}:
        return "concentration"
    if "scanrate" in compact or ("scan" in compact and "rate" in compact):
        return "scan_rate"
    if compact in {"instrument", "potentiostat"}:
        return "instrument"
    if compact in {"technique", "method"}:
        return "technique"
    return ""


def metadata_contains_semantic(metadata: dict[str, str], semantic_key: str) -> bool:
    return any(metadata_semantic_key(key) == semantic_key for key in metadata)


def remove_metadata_semantic(metadata: dict[str, str], semantic_key: str) -> None:
    for key in list(metadata.keys()):
        if metadata_semantic_key(key) == semantic_key:
            metadata.pop(key, None)


def merge_manual_metadata(
    detected_metadata: dict[str, str],
    structured_metadata: dict[str, str],
    freeform_metadata: dict[str, str],
    mode: str,
) -> dict[str, str]:
    merged = dict(detected_metadata)
    override_mode = "override" in mode.lower()

    for label, value in structured_metadata.items():
        semantic = metadata_semantic_key(label)
        if override_mode:
            remove_metadata_semantic(merged, semantic)
            merged[label] = value
        elif semantic and not metadata_contains_semantic(merged, semantic):
            merged[label] = value
        elif not semantic and label not in merged:
            merged[label] = value

    for label, value in freeform_metadata.items():
        semantic = metadata_semantic_key(label)
        if override_mode and semantic:
            remove_metadata_semantic(merged, semantic)
            merged[label] = value
        elif override_mode:
            merged[label] = value
        elif semantic and not metadata_contains_semantic(merged, semantic):
            merged[label] = value
        elif not semantic and label not in merged:
            merged[label] = value

    return merged


def normalize_electrode_area_text(value: Any) -> Optional[str]:
    text = str(value).strip()
    match = _NUMBER_RE.search(text)
    if not match:
        return text or None
    area = float(match.group(0))
    unit = "cm^2" if "cm" in normalize_label(text) else "cm^2"
    return f"{format_summary_number(area)} {display_unit_label(unit)}"


def electrode_area_metadata_summary(metadata: dict[str, str]) -> Optional[str]:
    for key, value in metadata.items():
        compact_key = re.sub(r"[^a-z0-9]+", "", normalize_label(key))
        if compact_key in {"electrodearea", "surfacearea", "geometricarea"} or (
            "area" in compact_key and "electrode" in compact_key
        ):
            return normalize_electrode_area_text(value)
    return None


def electrode_area_metadata_value(metadata: dict[str, str]) -> Optional[float]:
    area_summary = electrode_area_metadata_summary(metadata)
    if not area_summary:
        return None
    match = _NUMBER_RE.search(area_summary)
    if not match:
        return None
    value = float(match.group(0))
    return value if np.isfinite(value) and value > 0 else None


def normalize_reference_electrode(value: Any) -> Optional[str]:
    text = str(value).strip()
    if not text:
        return None
    normalized = normalize_label(text)
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    reference_aliases = {
        "agagcl": "Ag/AgCl",
        "agcl": "Ag/AgCl",
        "sce": "SCE",
        "saturatedcalomelelectrode": "SCE",
        "shennhe": "SHE/NHE",
        "she": "SHE/NHE",
        "nhe": "SHE/NHE",
        "rhe": "RHE",
        "hghgo": "Hg/HgO",
        "hghg2so4": "Hg/Hg2SO4",
        "fcfc": "Fc/Fc+",
        "ptquasireference": "Pt quasi-reference",
    }
    for alias, reference in reference_aliases.items():
        if alias in compact:
            return reference
    return text


def reference_electrode_from_metadata(metadata: dict[str, str]) -> Optional[str]:
    for key, value in metadata.items():
        compact_key = re.sub(r"[^a-z0-9]+", "", normalize_label(key))
        if compact_key in {"reference", "ref", "referenceelectrode", "refelectrode"} or (
            "reference" in compact_key and "electrode" in compact_key
        ):
            return normalize_reference_electrode(value)
    return None


def reference_electrode_source_label(reference_electrode: Optional[str], source: str) -> str:
    if not reference_electrode:
        return "Not selected"
    source_labels = {
        "metadata": "from metadata",
        "default": "default",
        "user": "confirmed",
    }
    source_text = source_labels.get(source, source)
    return f"{reference_electrode} ({source_text})"


def reference_electrode_default_note(reference_electrode: Optional[str], source: str) -> str:
    if reference_electrode and source == "default":
        return f"Reference electrode was not found in the file; using default {reference_electrode}."
    return ""


def metadata_completeness_for_lsv(
    *,
    scan_rate: str,
    reference_electrode_source: str,
    current_source: str,
    display_units: dict[str, Any],
    electrode_area_metadata: str,
) -> tuple[str, str]:
    missing: list[str] = []
    if not scan_rate or scan_rate == "Not available":
        missing.append("scan rate")
    if reference_electrode_source == "default":
        missing.append("confirmed reference electrode")

    uses_density = current_source == "Current density" or "Current density" in {
        display_units.get("x_axis"),
        display_units.get("y_axis"),
    }
    area_value = float(display_units.get("electrode_area_cm2", 0.0) or 0.0)
    area_available = electrode_area_metadata not in {None, "", "Not detected"} or (
        current_source == "Current" and area_value > 0
    )
    if uses_density and not area_available:
        missing.append("electrode area")

    if not missing:
        return "Complete", "None"
    return "Incomplete", ", ".join(missing)


def metadata_completeness_status_from_fields(
    missing_metadata: Optional[list[str]] = None,
    metadata_status: Optional[dict[str, dict[str, str]]] = None,
) -> str:
    missing = [str(field).strip() for field in (missing_metadata or []) if str(field).strip()]
    metadata_status = metadata_status or {}
    reference_status = metadata_status.get("reference_electrode", {})
    if missing or reference_status.get("status") == "defaulted":
        return "Incomplete"
    review_statuses = {"low_confidence", "ambiguous", "review_recommended"}
    if any(record.get("status") in review_statuses for record in metadata_status.values()):
        return "Review recommended"
    return "Complete"


def unit_status_label(unit_confidence: str, unit_confirmation_required: bool = False) -> str:
    if unit_confirmation_required:
        return "Unconfirmed"
    normalized = str(unit_confidence or "").strip().lower()
    if normalized in {"confirmed", "confirmed by user", "user-confirmed"}:
        return "Confirmed"
    if normalized in {"low", "ambiguous", "review needed"}:
        return "Review recommended"
    if normalized in {"medium"}:
        return "Medium confidence"
    if normalized in {"high"}:
        return "High confidence"
    return str(unit_confidence or "Not available")


def lsv_normalization_status_line(
    current_source: str,
    display_quantity: str,
    electrode_area_cm2: float,
    electrode_area_metadata: Optional[str],
) -> str:
    metadata_area = str(electrode_area_metadata or "").strip()
    metadata_available = metadata_area not in {"", "None", "Not detected"}
    area_text = metadata_area if metadata_available else (
        f"{electrode_area_cm2:.1f} cm²"
        if electrode_area_cm2 == round(electrode_area_cm2)
        else f"{electrode_area_cm2:g} cm²"
    )
    if current_source == "Current density":
        if metadata_available:
            return (
                f"Uploaded signal: current density. Electrode area detected: {metadata_area}. "
                "Electrode-area normalization was skipped."
            )
        return "Uploaded signal: current density. Electrode-area normalization was skipped."
    if display_quantity == "Current density":
        return (
            f"Uploaded signal: current. Current was normalized by VoltScope using electrode area = {area_text} "
            "and displayed as current density."
        )
    if metadata_available:
        return (
            f"Uploaded signal: current. Electrode area detected: {metadata_area}. "
            "Current-density display is available."
        )
    return "Uploaded signal: current."


def parse_scan_rate_value(text: Any) -> Optional[float]:
    normalized = str(text or "").strip().replace("µ", "u")
    if not normalized:
        return None
    patterns = [
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(m?v)\s*/\s*s",
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(m?v)\s*(?:per\s+)?s(?:ec(?:ond)?)?\b",
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(m?v)\s*s\^-?1",
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(m?v)\s*s-1",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "mv":
            value *= 1e-3
        return value if np.isfinite(value) and value > 0 else None
    return None


def scan_rate_from_metadata(metadata: dict[str, str]) -> Optional[float]:
    for key, value in metadata.items():
        key_label = normalize_label(key)
        key_compact = re.sub(r"[^a-z0-9]+", "", key_label)
        if "scanrate" in key_compact or ("scan" in key_label and "rate" in key_label):
            value_match = _NUMBER_RE.search(str(value))
            if value_match:
                numeric_value = float(value_match.group(0))
                key_tokens = [token for token in re.split(r"[^a-z0-9]+", key_label) if token]
                key_unit = None
                for idx, token in enumerate(key_tokens):
                    next_token = key_tokens[idx + 1] if idx + 1 < len(key_tokens) else ""
                    if token in {"mv", "millivolt", "millivolts"} and next_token in {"s", "sec", "second", "seconds"}:
                        key_unit = "mV/s"
                        break
                    if token in {"v", "volt", "volts"} and next_token in {"s", "sec", "second", "seconds"}:
                        key_unit = "V/s"
                        break
                if key_unit is None:
                    if "mvpersecond" in key_compact or key_compact.endswith("mvs") or "mvs" in key_compact:
                        key_unit = "mV/s"
                    elif "vpersecond" in key_compact or key_compact.endswith("vs") or "vs" in key_compact:
                        key_unit = "V/s"
                if key_unit == "mV/s":
                    numeric_value *= 1e-3
                if key_unit in {"mV/s", "V/s"} and np.isfinite(numeric_value) and numeric_value > 0:
                    return numeric_value
                if key_unit is None and np.isfinite(numeric_value) and numeric_value > 0:
                    # Potentiostat exports often store "Scan rate: 10" with the unit in surrounding notes
                    # or omitted entirely. Values above 1 are almost always mV/s in CV/LSV exports.
                    return numeric_value * 1e-3 if numeric_value > 1 else numeric_value
        candidates = [str(value), f"{key} {value}"]
        if "scanrate" in key_compact or ("scan" in key_label and "rate" in key_label):
            candidates.insert(0, str(value))
        for candidate in candidates:
            scan_rate = parse_scan_rate_value(candidate)
            if scan_rate is not None:
                return scan_rate
    return None


def scan_rate_summary_from_value(scan_rate_v_s: float) -> str:
    return f"{format_summary_number(scan_rate_v_s)} V/s ({format_summary_number(scan_rate_v_s * 1000)} mV/s)"


def find_table_bounds(lines: list[str], delimiter: str) -> tuple[int, int, list[str], bool]:
    saw_single_numeric_column = False
    for idx, line in enumerate(lines):
        tokens = split_line(line, delimiter)
        if numeric_count(tokens) == 1:
            saw_single_numeric_column = True
        if len(tokens) < 2:
            continue

        if is_numeric_row(tokens):
            previous_tokens = split_line(lines[idx - 1], delimiter) if idx > 0 else []
            previous_has_header = (
                len(previous_tokens) >= 2
                and not is_numeric_row(previous_tokens)
                and any(re.search(r"[A-Za-z]", token) for token in previous_tokens)
            )
            if previous_has_header:
                return idx - 1, idx, deduplicate_headers(previous_tokens), True
            headers = [f"Column {column_idx + 1}" for column_idx in range(len(tokens))]
            return idx, idx, headers, False

        if idx + 1 < len(lines):
            next_tokens = split_line(lines[idx + 1], delimiter)
            if is_numeric_row(next_tokens):
                return idx, idx + 1, deduplicate_headers(tokens), True

    if saw_single_numeric_column:
        raise ValueError("The uploaded file must contain at least two usable numeric columns.")
    raise ValueError("Could not find a numeric data table in the uploaded file.")


def dropped_rows_total_candidate_count(rows_imported: int, rows_dropped: int) -> int:
    return max(int(rows_imported) + int(rows_dropped), 0)


def dropped_rows_fraction(rows_imported: int, rows_dropped: int) -> float:
    total = dropped_rows_total_candidate_count(rows_imported, rows_dropped)
    if total <= 0:
        return 0.0
    return float(rows_dropped) / float(total)


def dropped_rows_severity(rows_imported: int, rows_dropped: int) -> str:
    if rows_dropped <= 0:
        return "none"
    fraction = dropped_rows_fraction(rows_imported, rows_dropped)
    if fraction >= PARSER_DROPPED_ROWS_STRONG_RATIO:
        return "strong"
    if fraction >= PARSER_DROPPED_ROWS_REVIEW_RATIO:
        return "review"
    return "cleanup"


def dropped_rows_cleanup_notice(rows_imported: int, rows_dropped: int) -> str:
    if rows_dropped <= 0:
        return ""
    row_word = "row" if rows_dropped == 1 else "rows"
    base = f"{rows_dropped} non-numeric data {row_word} were removed during cleanup."
    fraction = dropped_rows_fraction(rows_imported, rows_dropped)
    if fraction >= PARSER_DROPPED_ROWS_STRONG_RATIO:
        return (
            f"Strong cleanup warning: {base} "
            f"That is {fraction:.1%} of candidate data rows; review the file before interpreting results."
        )
    if fraction >= PARSER_DROPPED_ROWS_REVIEW_RATIO:
        return (
            f"Data cleanup review recommended: {base} "
            f"That is {fraction:.1%} of candidate data rows."
        )
    return base


def dropped_rows_reason(rows_dropped: int) -> str:
    if rows_dropped <= 0:
        return "None"
    return "non-numeric or corrupted values inside numeric data region."


def dropped_rows_fraction_label(rows_imported: int, rows_dropped: int) -> str:
    if rows_dropped <= 0:
        return "0%"
    return f"{dropped_rows_fraction(rows_imported, rows_dropped):.2%}"


def current_unit_from_header_token(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    label = normalize_label(header)
    compact = re.sub(r"[^a-z0-9]+", "", label)
    tokens = [token for token in re.split(r"[^a-z0-9]+", label) if token]

    unit_tokens = [
        ("pA", {"pa", "picoamp", "picoamps", "picoampere", "picoamperes"}),
        ("nA", {"na", "nanoamp", "nanoamps", "nanoampere", "nanoamperes"}),
        ("uA", {"ua", "microamp", "microamps", "microampere", "microamperes"}),
        ("mA", {"ma", "milliamp", "milliamps", "milliampere", "milliamperes"}),
        ("A", {"a", "amp", "amps", "ampere", "amperes"}),
    ]
    for unit, aliases in unit_tokens:
        if any(token in aliases for token in tokens):
            return unit

    compact_patterns = [
        ("pA", ("pa", "picoamp", "picoamps", "picoampere", "picoamperes")),
        ("nA", ("na", "nanoamp", "nanoamps", "nanoampere", "nanoamperes")),
        ("uA", ("ua", "microamp", "microamps", "microampere", "microamperes")),
        ("mA", ("ma", "milliamp", "milliamps", "milliampere", "milliamperes")),
        ("A", ("a", "amp", "amps", "ampere", "amperes")),
    ]
    for unit, suffixes in compact_patterns:
        if any(
            compact in {suffix, f"current{suffix}", f"i{suffix}", f"j{suffix}", f"currentdensity{suffix}"}
            or compact.endswith(f"per{suffix}")
            for suffix in suffixes
        ):
            return unit
    return None


def potential_header_has_explicit_unit(header: Optional[str]) -> bool:
    if not header:
        return False
    label = normalize_label(header)
    compact = re.sub(r"[^a-z0-9]+", "", label)
    tokens = [token for token in re.split(r"[^a-z0-9]+", label) if token]
    if any(token in {"v", "volt", "volts", "mv", "millivolt", "millivolts"} for token in tokens):
        return True
    return compact in {
        "v",
        "ev",
        "emv",
        "ewev",
        "ewemv",
        "potentialv",
        "potentialmv",
        "voltagev",
        "voltagemv",
    }


def detect_unit_from_header(header: str, role: str) -> Optional[str]:
    label = normalize_label(header)
    compact = re.sub(r"[^a-z0-9]+", "", label)

    if role == "potential":
        if "mv" in label or "millivolt" in label:
            return "mV"
        if (
            "/v" in label
            or "(v)" in label
            or label in {"v", "e", "voltage"}
            or compact in {"v", "e", "potentialv", "voltagev", "ewev", "ev"}
            or compact.endswith("volts")
        ):
            return "V"
        return None

    if role == "current":
        return current_unit_from_header_token(header)

    if role == "time":
        if "ms" in label or compact.endswith("ms") or "millisecond" in label:
            return "ms"
        if "min" in label or compact.endswith("min") or "minute" in label:
            return "min"
        if compact in {"times", "ts", "seconds", "secs"} or compact.endswith("seconds"):
            return "s"
        if "ms" in label:
            return "ms"
        if "min" in label:
            return "min"
        if label.endswith("/h") or label.endswith("(h)") or label in {"h", "hour", "hours"}:
            return "h"
        if "/s" in label or "(s)" in label or label in {"s", "t", "time"}:
            return "s"
        return None

    if role == "frequency":
        if "mhz" in label:
            return "MHz"
        if "khz" in label:
            return "kHz"
        if "hz" in label or "freq" in label:
            return "Hz"
        return None

    if role in {"zreal", "zimag", "zmod", "impedance"}:
        if "mohm" in label or "megaohm" in label:
            return "Mohm"
        if "kohm" in label or "komega" in label:
            return "kohm"
        if "ohm" in label or "omega" in label or "\u03a9" in header:
            return "ohm"
        return None

    if role == "phase":
        if "rad" in label:
            return "rad"
        if "deg" in label or "degree" in label or "phase" in label:
            return "deg"
        return None

    return None


def header_indicates_current_density(header: str) -> bool:
    label = normalize_label(header)
    compact = re.sub(r"[^a-z0-9]+", "", label)
    density_tokens = [
        "currentdensity",
        "currentdens",
        "currentperarea",
        "ma/cm2",
        "ua/cm2",
        "na/cm2",
        "pa/cm2",
        "a/cm2",
        "macm2",
        "uacm2",
        "nacm2",
        "pacm2",
        "acm2",
        "macm-2",
        "uacm-2",
        "nacm-2",
        "pacm-2",
        "acm-2",
    ]
    if any(token in label for token in density_tokens):
        return True
    if any(token in compact for token in ["currentdensity", "currentdens", "macm2", "uacm2", "nacm2", "pacm2", "acm2"]):
        return True
    return compact in {"j", "ja", "jma", "jua", "jna", "jpa"}


def text_indicates_current_density(text: str) -> bool:
    label = normalize_label(text)
    compact = re.sub(r"[^a-z0-9]+", "", label)
    density_patterns = [
        "current density",
        "current_density",
        "current-density",
        "do not normalize",
        "do_not_normalize",
        "already normalized",
        "normalized current",
        "ma/cm2",
        "ma/cm^2",
        "ma/cm²",
        "ma cm2",
        "ma_cm2",
        "ua/cm2",
        "ua/cm^2",
        "µa/cm2",
        "µa/cm^2",
        "ua_cm2",
        "a/cm2",
        "a/cm^2",
    ]
    if any(pattern in label for pattern in density_patterns):
        return True
    if any(
        token in compact
        for token in [
            "currentdensity",
            "currentdens",
            "currentperarea",
            "donotnormalize",
            "alreadynormalized",
            "normalizedcurrent",
            "jmacm2",
            "juacm2",
            "macm2",
            "uacm2",
            "acm2",
        ]
    ):
        return True
    return bool(re.search(r"(^|[^a-z0-9])j([^a-z0-9]|$)", label))


def dataset_suggests_current_density_signal(dataset: ParsedDataset, current_col: Optional[str] = None) -> bool:
    text_parts = [dataset.filename, *dataset.headers]
    text_parts.extend(dataset.metadata.keys())
    text_parts.extend(dataset.metadata.values())
    if current_col:
        text_parts.append(current_col)
    return any(text_indicates_current_density(str(part)) for part in text_parts if part is not None)


def score_potential_column(header: str, values: np.ndarray) -> float:
    label = normalize_label(header)
    compact = re.sub(r"[^a-z0-9]+", "", label)
    finite = values[np.isfinite(values)]
    score = 0.0
    potential_unit = detect_unit_from_header(header, "potential")
    current_unit = detect_unit_from_header(header, "current")

    if any(keyword in label for keyword in ["potential", "voltage", "ewe", "e/v", "potential/v"]):
        score += 7
    if label in {"e", "v", "x", "potentialv", "ewe/v"}:
        score += 4
    if potential_unit and compact in {"emv", "ev", "ewev", "ewemv", "potentialv", "potentialmv", "voltagev", "voltagemv"}:
        score += 6
    elif potential_unit:
        score += 4
    if current_unit and not potential_unit:
        score -= 5
    if "time" in label or "current" in label or "density" in label or label in {"i", "a", "j", "cycle"}:
        score -= 5

    if finite.size:
        value_range = float(np.nanmax(finite) - np.nanmin(finite))
        median_abs = float(np.nanmedian(np.abs(finite)))
        if 0.01 <= value_range <= 10:
            score += 3
        if median_abs <= 10:
            score += 1
        if np.unique(np.round(finite, 9)).size > 3:
            score += 1

    return score


def score_current_column(header: str, values: np.ndarray) -> float:
    label = normalize_label(header)
    finite = values[np.isfinite(values)]
    score = 0.0
    current_unit = detect_unit_from_header(header, "current")
    potential_unit = detect_unit_from_header(header, "potential")

    if any(keyword in label for keyword in ["current", "i/a", "j/a", "current/a", "amp", "amps"]):
        score += 7
    if header_indicates_current_density(header):
        score += 4
    if label in {"i", "j", "y", "a", "ma", "ua", "na"}:
        score += 4
    if "potential" in label or "voltage" in label or label in {"e", "v"}:
        score -= 5
    if "cycle" in label or label in {"n", "index"}:
        score -= 4
    if current_unit:
        score += 2
    if potential_unit and not current_unit:
        score -= 5

    if finite.size:
        value_range = float(np.nanmax(finite) - np.nanmin(finite))
        median_abs = float(np.nanmedian(np.abs(finite)))
        if value_range > 0:
            score += 1
        if median_abs <= 1:
            score += 1
        if np.unique(np.round(finite, 12)).size > 3:
            score += 1

    return score


def detect_column_roles(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    potential_scores = {
        column: score_potential_column(column, df[column].to_numpy(dtype=float))
        for column in df.columns
    }
    current_scores = {
        column: score_current_column(column, df[column].to_numpy(dtype=float))
        for column in df.columns
    }

    potential_col = max(potential_scores, key=potential_scores.get, default=None)
    if potential_col is not None and potential_scores[potential_col] < 2:
        potential_col = None

    current_candidates = sorted(current_scores, key=current_scores.get, reverse=True)
    current_col = None
    for candidate in current_candidates:
        if candidate != potential_col and current_scores[candidate] >= 2:
            current_col = candidate
            break

    time_col = None
    for column in df.columns:
        label = normalize_label(column)
        if label in {"time", "times", "t", "time/s", "times"} or "time" in label:
            time_col = column
            break

    return potential_col, current_col, time_col




def score_time_column(header: str, values: np.ndarray) -> float:
    label = normalize_label(header)
    finite = values[np.isfinite(values)]
    score = 0.0

    if "time" in label or label in {"t", "s", "time/s", "seconds"}:
        score += 8
    if detect_unit_from_header(header, "time"):
        score += 2
    if any(term in label for term in ["potential", "voltage", "current", "freq", "zreal", "zimag"]):
        score -= 4

    if finite.size:
        differences = np.diff(finite)
        nonzero = differences[np.abs(differences) > 1e-15]
        if nonzero.size and np.mean(nonzero > 0) > 0.85:
            score += 4
        if np.nanmin(finite) >= 0:
            score += 1

    return score


def score_frequency_column(header: str, values: np.ndarray) -> float:
    label = normalize_label(header)
    finite = values[np.isfinite(values)]
    score = 0.0

    if any(term in label for term in ["frequency", "freq", "hz"]):
        score += 9
    if label in {"f", "freq/hz"}:
        score += 4
    if any(term in label for term in ["time", "current", "potential", "zreal", "zimag"]):
        score -= 4
    if detect_unit_from_header(header, "frequency"):
        score += 2

    if finite.size:
        positive_fraction = float(np.mean(finite > 0))
        if positive_fraction > 0.95:
            score += 2
        if np.nanmax(finite) / max(np.nanmin(finite[finite > 0]), 1e-30) > 10 if np.any(finite > 0) else False:
            score += 1

    return score


def score_impedance_column(header: str, values: np.ndarray, role: str) -> float:
    label = normalize_label(header)
    finite = values[np.isfinite(values)]
    score = 0.0

    if role == "zreal":
        keywords = ["zreal", "zre", "z'", "real", "rez", "z_real"]
        wrong_keywords = ["imag", "zimag", "phase", "freq"]
    elif role == "zimag":
        keywords = ["zimag", "zim", "z''", "imag", "imz", "z_imag", "-z"]
        wrong_keywords = ["real", "zreal", "phase", "freq"]
    elif role == "zmod":
        keywords = ["zmod", "|z|", "modz", "magnitude", "impedance", "z/ohm"]
        wrong_keywords = ["real", "imag", "phase", "freq"]
    else:
        keywords = ["phase", "theta", "deg"]
        wrong_keywords = ["real", "imag", "zmod", "freq"]

    if any(keyword in label for keyword in keywords):
        score += 9
    if any(keyword in label for keyword in wrong_keywords):
        score -= 5
    if detect_unit_from_header(header, role):
        score += 2

    if finite.size:
        if role in {"zreal", "zmod"} and np.mean(finite >= 0) > 0.75:
            score += 1
        if role == "phase" and np.nanmax(np.abs(finite)) <= 360:
            score += 2
        if np.unique(np.round(finite, 12)).size > 3:
            score += 1

    return score


def score_column_for_role(header: str, values: np.ndarray, role: str) -> float:
    if role == "potential":
        return score_potential_column(header, values)
    if role == "current":
        return score_current_column(header, values)
    if role == "time":
        return score_time_column(header, values)
    if role == "frequency":
        return score_frequency_column(header, values)
    if role in {"zreal", "zimag", "zmod", "phase"}:
        return score_impedance_column(header, values, role)
    return 0.0


def detect_column_by_role(
    df: pd.DataFrame,
    role: str,
    excluded: Optional[set[str]] = None,
    min_score: float = 2.0,
) -> Optional[str]:
    excluded = excluded or set()
    scores = {}
    for column in df.columns:
        if column in excluded:
            continue
        scores[column] = score_column_for_role(column, df[column].to_numpy(dtype=float), role)

    if not scores:
        return None
    best_column = max(scores, key=scores.get)
    return best_column if scores[best_column] >= min_score else None


def detect_experiment_columns(dataset: ParsedDataset, experiment_type: str) -> dict[str, Optional[str]]:
    df = dataset.dataframe
    columns: dict[str, Optional[str]] = {}
    excluded: set[str] = set()

    if experiment_type in {"cv", "lsv"}:
        columns["potential"] = dataset.detected_potential_col or detect_column_by_role(df, "potential", excluded)
        if columns["potential"]:
            excluded.add(columns["potential"])
        columns["current"] = dataset.detected_current_col or detect_column_by_role(df, "current", excluded)
        return columns

    if experiment_type == "ca":
        columns["time"] = dataset.detected_time_col or detect_column_by_role(df, "time", excluded)
        if columns["time"]:
            excluded.add(columns["time"])
        columns["current"] = dataset.detected_current_col or detect_column_by_role(df, "current", excluded)
        return columns

    if experiment_type == "cp":
        columns["time"] = dataset.detected_time_col or detect_column_by_role(df, "time", excluded)
        if columns["time"]:
            excluded.add(columns["time"])
        columns["potential"] = dataset.detected_potential_col or detect_column_by_role(df, "potential", excluded)
        return columns

    if experiment_type == "eis":
        columns["frequency"] = detect_column_by_role(df, "frequency", excluded, min_score=3.0)
        if columns["frequency"]:
            excluded.add(columns["frequency"])
        columns["zreal"] = detect_column_by_role(df, "zreal", excluded, min_score=3.0)
        if columns["zreal"]:
            excluded.add(columns["zreal"])
        columns["zimag"] = detect_column_by_role(df, "zimag", excluded, min_score=3.0)
        if columns["zimag"]:
            excluded.add(columns["zimag"])
        columns["zmod"] = detect_column_by_role(df, "zmod", excluded, min_score=3.0)
        if columns["zmod"]:
            excluded.add(columns["zmod"])
        columns["phase"] = detect_column_by_role(df, "phase", excluded, min_score=3.0)
        return columns

    return columns

def read_uploaded_text(uploaded_file: Any) -> str:
    raw = uploaded_file.getvalue()
    for encoding in ["utf-8-sig", "utf-16", "latin-1"]:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_electrochem_file(uploaded_file: Any) -> ParsedDataset:
    text = read_uploaded_text(uploaded_file)
    lines = [line.rstrip("\n\r") for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("The uploaded file is empty.")

    delimiter = detect_delimiter(lines)
    header_idx, data_start_idx, headers, has_header = find_table_bounds(lines, delimiter)
    metadata = parse_metadata(lines[:header_idx])

    rows = []
    rows_dropped = 0
    for line in lines[data_start_idx:]:
        tokens = split_line(line, delimiter)
        if len(tokens) < 2:
            rows_dropped += 1
            continue

        if len(tokens) > len(headers):
            extra_headers = [f"Column {idx + 1}" for idx in range(len(headers), len(tokens))]
            headers = deduplicate_headers(headers + extra_headers)
        elif len(tokens) < len(headers):
            tokens = tokens + [""] * (len(headers) - len(tokens))

        values = [parse_number(token) for token in tokens[: len(headers)]]
        if np.isfinite(values).sum() >= 2:
            rows.append(values)
        else:
            rows_dropped += 1

    if not rows:
        raise ValueError("Could not parse numeric rows from the data table.")

    df = pd.DataFrame(rows, columns=headers)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("The uploaded file must contain at least two usable numeric columns.")
    headers = list(df.columns)
    missing_values = int(df.isna().sum().sum())

    potential_col, current_col, time_col = detect_column_roles(df)
    detected_units: dict[str, str] = {}
    for column in headers:
        for role in [
            "potential",
            "current",
            "time",
            "frequency",
            "zreal",
            "zimag",
            "zmod",
            "phase",
        ]:
            unit = detect_unit_from_header(column, role)
            if unit:
                detected_units[f"{role}:{column}"] = unit

    warnings = []
    if not has_header:
        warnings.append("No header row was detected; generated generic column names.")
    cleanup_notice = dropped_rows_cleanup_notice(len(df), rows_dropped)
    if cleanup_notice:
        warnings.append(cleanup_notice)

    return ParsedDataset(
        filename=uploaded_file.name,
        dataframe=df,
        headers=headers,
        metadata=metadata,
        delimiter=delimiter,
        has_header=has_header,
        header_row=header_idx + 1 if has_header else None,
        data_start_row=data_start_idx + 1,
        rows_skipped=data_start_idx,
        rows_dropped=rows_dropped,
        missing_values=missing_values,
        detected_potential_col=potential_col,
        detected_current_col=current_col,
        detected_time_col=time_col,
        detected_units=detected_units,
        warnings=warnings,
    )


# ---------- Unit conversion and analysis helpers ----------


def resolve_unit(
    selected_unit: str,
    dataset: ParsedDataset,
    role: str,
    column: str,
    default: str,
) -> str:
    if selected_unit != "Auto":
        return selected_unit
    return dataset.detected_units.get(f"{role}:{column}", default)


def infer_potential_unit_from_values(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return "V"
    p95_abs = float(np.nanpercentile(np.abs(finite), 95))
    value_range = float(np.nanmax(finite) - np.nanmin(finite))
    if 20 < p95_abs <= 5000 or 20 < value_range <= 10000:
        return "mV"
    return "V"


def convert_potential_to_volts(values: np.ndarray, unit: str) -> np.ndarray:
    return values * POTENTIAL_UNITS_TO_V.get(unit, 1.0)


def convert_current_to_amps(values: np.ndarray, unit: str) -> np.ndarray:
    return values * CURRENT_UNITS_TO_A.get(unit, 1.0)


CURRENT_DENSITY_UNITS_TO_A_PER_CM2 = {
    "A/cm^2": 1.0,
    "mA/cm^2": 1e-3,
    "uA/cm^2": 1e-6,
    "nA/cm^2": 1e-9,
    "pA/cm^2": 1e-12,
}
CURRENT_DENSITY_UNIT = "mA/cm^2"
DEFAULT_REFERENCE_ELECTRODE = "Ag/AgCl"
REFERENCE_ELECTRODE_OPTIONS = [
    DEFAULT_REFERENCE_ELECTRODE,
    "SCE",
    "SHE/NHE",
    "RHE",
    "Hg/HgO",
    "Hg/Hg2SO4",
    "Fc/Fc+",
    "Pt quasi-reference",
    "Other",
]


def is_current_density_unit(display_unit: str) -> bool:
    return display_unit in CURRENT_DENSITY_UNITS_TO_A_PER_CM2


def current_density_unit_from_current_unit(current_unit: str) -> str:
    if current_unit in CURRENT_UNITS_TO_A:
        return f"{current_unit}/cm^2"
    return CURRENT_DENSITY_UNIT


def current_unit_from_density_unit(current_density_unit: str) -> str:
    mapping = {
        "A/cm^2": "A",
        "mA/cm^2": "mA",
        "uA/cm^2": "uA",
        "nA/cm^2": "nA",
        "pA/cm^2": "pA",
    }
    return mapping.get(current_density_unit, current_density_unit)


def confirmation_unit_options(current_source: str) -> list[str]:
    if current_source == "Current density":
        return ["A/cm^2", "mA/cm^2", "uA/cm^2", "nA/cm^2", "pA/cm^2"]
    return ["A", "mA", "uA", "nA", "pA"]


def confirmation_unit_type(current_source: str) -> str:
    return "current_density" if current_source == "Current density" else "raw_current"


def lsv_current_source_state_key(
    project_name: str,
    experiment_id: str,
    filename: str,
    current_col: Optional[str],
) -> str:
    return (
        f"lsv_current_source::{project_name}::{experiment_id}::"
        f"{filename}::{current_col or 'current'}"
    )


def confirmed_unit_source_for_lsv(
    dataset: ParsedDataset,
    current_col: Optional[str],
    current_source_state_key: str,
    session_state: Optional[dict[str, Any]] = None,
) -> str:
    state = st.session_state if session_state is None else session_state
    session_source = state.get(current_source_state_key)
    if session_source in {"Current", "Current density"}:
        return str(session_source)
    return current_source_guess_for_dataset(dataset, current_col)


def unit_input_from_confirmed_unit(confirmed_unit: Any, current_source: str) -> Optional[str]:
    if confirmed_unit is None:
        return None
    unit = str(confirmed_unit)
    if current_source == "Current density":
        unit = current_unit_from_density_unit(unit)
    return unit if unit in CURRENT_UNITS_TO_A else None


def current_density_from_base(
    values: np.ndarray,
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float = 1.0,
) -> np.ndarray:
    if current_source == "Current density":
        return values / CURRENT_DENSITY_UNITS_TO_A_PER_CM2.get(display_unit, 1e-3)
    return values / electrode_area_cm2 / CURRENT_DENSITY_UNITS_TO_A_PER_CM2.get(display_unit, 1e-3)


def current_density_value_to_base(
    value: float,
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float = 1.0,
) -> float:
    base_density = value * CURRENT_DENSITY_UNITS_TO_A_PER_CM2.get(display_unit, 1e-3)
    if current_source == "Current density":
        return base_density
    return base_density * electrode_area_cm2


def current_to_display(
    values: np.ndarray,
    display_unit: str,
    current_source: str = "Current",
    electrode_area_cm2: float = 1.0,
) -> np.ndarray:
    if current_source == "Current density":
        if is_current_density_unit(display_unit):
            return values / CURRENT_DENSITY_UNITS_TO_A_PER_CM2.get(display_unit, 1e-3)
        return values * electrode_area_cm2 / CURRENT_UNITS_TO_A.get(display_unit, 1.0)
    if is_current_density_unit(display_unit):
        return current_density_from_base(values, display_unit, "Current", electrode_area_cm2)
    return values / CURRENT_UNITS_TO_A.get(display_unit, 1.0)


def current_display_value_to_base(
    value: float,
    display_unit: str,
    current_source: str = "Current",
    electrode_area_cm2: float = 1.0,
) -> float:
    if current_source == "Current density":
        if is_current_density_unit(display_unit):
            return current_density_value_to_base(value, display_unit, "Current density", electrode_area_cm2)
        current_a = value * CURRENT_UNITS_TO_A.get(display_unit, 1.0)
        return current_a / electrode_area_cm2
    if is_current_density_unit(display_unit):
        return current_density_value_to_base(value, display_unit, "Current", electrode_area_cm2)
    return value * CURRENT_UNITS_TO_A.get(display_unit, 1.0)


def current_from_amps(
    values: np.ndarray,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
) -> np.ndarray:
    return current_to_display(values, display_unit, "Current", electrode_area_cm2)


def current_value_to_amps(
    value: float,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
) -> float:
    return current_display_value_to_base(value, display_unit, "Current", electrode_area_cm2)


def current_axis_label(display_unit: str) -> str:
    unit_label = display_unit_label(display_unit)
    if is_current_density_unit(display_unit):
        return f"Current density ({unit_label})"
    return f"Current ({unit_label})"


def current_hover_label(display_unit: str) -> str:
    return "Current density" if is_current_density_unit(display_unit) else "Current"


def lsv_axis_title(axis_quantity: str, display_units: dict[str, Any], reference_electrode: Optional[str]) -> str:
    if axis_quantity == "Potential":
        title = f"Potential ({display_units['potential']})"
        if reference_electrode:
            title += f" vs {reference_electrode}"
        return title
    if display_units.get("unit_review_required"):
        return lsv_unconfirmed_unit_label(display_units.get("current_source", "Current"))
    return current_axis_label(display_units["current"])


def dataframe_preview_for_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    hidden_columns = [
        column
        for column in dataframe.columns
        if re.sub(r"[^a-z0-9]+", "", str(column).lower()) == "cycle"
    ]
    return dataframe.drop(columns=hidden_columns, errors="ignore")


def unit_options_for_role(role: str) -> list[str]:
    if role == "potential":
        return ["Auto", "V", "mV"]
    if role == "current":
        return ["Auto", "A", "mA", "uA", "nA", "pA"]
    if role == "time":
        return ["Auto", "s", "ms", "min", "h"]
    if role == "frequency":
        return ["Auto", "Hz", "kHz", "MHz"]
    if role in {"zreal", "zimag", "zmod", "impedance"}:
        return ["Auto", "ohm", "kohm", "Mohm"]
    if role == "phase":
        return ["Auto", "deg", "rad"]
    return ["Auto"]


def default_base_unit_for_role(role: str) -> str:
    defaults = {
        "potential": "V",
        "current": "A",
        "time": "s",
        "frequency": "Hz",
        "zreal": "ohm",
        "zimag": "ohm",
        "zmod": "ohm",
        "impedance": "ohm",
        "phase": "deg",
    }
    return defaults.get(role, "")


def display_unit_options_for_role(role: str) -> list[str]:
    if role == "potential":
        return ["V", "mV"]
    if role == "current":
        return ["uA", "mA", "A", "nA", "pA"]
    if role == "time":
        return ["s", "ms", "min", "h"]
    if role == "frequency":
        return ["Hz", "kHz", "MHz"]
    if role in {"zreal", "zimag", "zmod", "impedance"}:
        return ["ohm", "kohm", "Mohm"]
    if role == "phase":
        return ["deg", "rad"]
    return [""]


def unit_factor_for_role(role: str, unit: str) -> float:
    if role == "potential":
        return POTENTIAL_UNITS_TO_V.get(unit, 1.0)
    if role == "current":
        return CURRENT_UNITS_TO_A.get(unit, 1.0)
    if role == "time":
        return TIME_UNITS_TO_S.get(unit, 1.0)
    if role == "frequency":
        return FREQUENCY_UNITS_TO_HZ.get(unit, 1.0)
    if role in {"zreal", "zimag", "zmod", "impedance"}:
        return IMPEDANCE_UNITS_TO_OHM.get(unit, 1.0)
    if role == "phase":
        return PHASE_UNITS_TO_DEG.get(unit, 1.0)
    return 1.0


def convert_role_to_base(values: np.ndarray, role: str, unit: str) -> np.ndarray:
    return values * unit_factor_for_role(role, unit)


def convert_role_from_base(values: np.ndarray, role: str, display_unit: str) -> np.ndarray:
    return values / unit_factor_for_role(role, display_unit)


def resolve_role_unit(selected_unit: str, dataset: ParsedDataset, role: str, column: Optional[str]) -> str:
    if selected_unit != "Auto":
        return selected_unit
    if column:
        detected = dataset.detected_units.get(f"{role}:{column}")
        if detected:
            return detected
    return default_base_unit_for_role(role)


def resolve_role_unit_from_values(
    selected_unit: str,
    dataset: ParsedDataset,
    role: str,
    column: Optional[str],
    values: np.ndarray,
) -> str:
    unit = resolve_role_unit(selected_unit, dataset, role, column)
    if selected_unit == "Auto" and role == "potential" and column:
        detected = dataset.detected_units.get(f"{role}:{column}")
        if not detected:
            return infer_potential_unit_from_values(values)
    return unit


def current_header_has_explicit_unit(header: Optional[str]) -> bool:
    return current_unit_from_header_token(header) is not None


def current_unit_confidence_detail(
    dataset: ParsedDataset,
    current_col: Optional[str],
    selected_current_unit: str = "Auto",
) -> str:
    if selected_current_unit != "Auto":
        return "Confirmed by user"
    if current_col and dataset.detected_units.get(f"current:{current_col}") and current_header_has_explicit_unit(current_col):
        return "Detected from column header"
    return "Column label is ambiguous"


def current_unit_confidence(
    dataset: ParsedDataset,
    current_col: Optional[str],
    selected_current_unit: str = "Auto",
) -> str:
    if selected_current_unit != "Auto":
        return "Confirmed by user"
    if current_col and dataset.detected_units.get(f"current:{current_col}") and current_header_has_explicit_unit(current_col):
        return "High"
    return "Low"


def current_unit_is_ambiguous(
    dataset: ParsedDataset,
    current_col: Optional[str],
    selected_current_unit: str = "Auto",
) -> bool:
    return current_unit_confidence(dataset, current_col, selected_current_unit) == "Low"


def suggested_current_unit_for_values(values: np.ndarray) -> str:
    finite_abs = np.abs(np.asarray(values, dtype=float))
    finite_abs = finite_abs[np.isfinite(finite_abs)]
    finite_abs = finite_abs[finite_abs > 0]
    if not finite_abs.size:
        return "uA"
    p95 = float(np.nanpercentile(finite_abs, 95))
    if p95 < 1e-3:
        return "A"
    if p95 < 1:
        return "mA"
    if p95 < 1e3:
        return "uA"
    if p95 < 1e6:
        return "nA"
    return "pA"


def suggested_current_unit_for_context(values: np.ndarray, current_source: str) -> str:
    if current_source != "Current density":
        return suggested_current_unit_for_values(values)
    finite_abs = np.abs(np.asarray(values, dtype=float))
    finite_abs = finite_abs[np.isfinite(finite_abs)]
    finite_abs = finite_abs[finite_abs > 0]
    if not finite_abs.size:
        return "mA"
    p95 = float(np.nanpercentile(finite_abs, 95))
    if p95 < 1e-3:
        return "A"
    if p95 < 0.1:
        return "uA"
    if p95 < 1e3:
        return "mA"
    return "uA"


def current_unit_magnitude_warning(
    dataset: ParsedDataset,
    current_col: Optional[str],
    selected_current_unit: str = "Auto",
    interpreted_current_unit: str = "A",
    current_source: str = "Current",
    electrode_area_cm2: float = 1.0,
) -> str:
    if not current_col or current_col not in dataset.dataframe.columns:
        return ""
    if not current_unit_is_ambiguous(dataset, current_col, selected_current_unit):
        return ""

    current_values = dataset.dataframe[current_col].to_numpy(dtype=float)
    finite_abs_raw = np.abs(current_values[np.isfinite(current_values)])
    finite_abs_raw = finite_abs_raw[finite_abs_raw > 0]
    if not finite_abs_raw.size:
        return ""

    suggested_unit = suggested_current_unit_for_context(current_values, current_source)
    interpreted_current_a = convert_current_to_amps(current_values, interpreted_current_unit)
    interpreted_display = current_to_display(
        interpreted_current_a,
        "uA/cm^2" if current_source == "Current density" else "uA",
        current_source,
        electrode_area_cm2,
    )
    finite_interpreted = np.abs(interpreted_display[np.isfinite(interpreted_display)])
    p95_interpreted = float(np.nanpercentile(finite_interpreted, 95)) if finite_interpreted.size else 0.0

    raw_p95 = float(np.nanpercentile(finite_abs_raw, 95))
    raw_values_look_scaled = 1 <= raw_p95 <= 1000 and suggested_unit != interpreted_current_unit
    interpreted_is_large = p95_interpreted >= 1e4
    if raw_values_look_scaled or interpreted_is_large:
        interpreted_label = display_unit_label(
            current_unit_label_for_summary(interpreted_current_unit, current_source)
        )
        return (
            "Current unit may be wrong. The current column has no unit label and values are "
            f"unusually large if interpreted as {interpreted_label}."
        )
    return ""


def current_unit_review_explanation(current_col_label: str, suggested_unit: str, current_unit_warning: str) -> str:
    suggested_label = display_unit_label(suggested_unit)
    is_density_suggestion = "/cm" in suggested_label
    if current_unit_warning:
        if is_density_suggestion:
            return (
                "The current-density column has no unit in the header. The uploaded signal appears "
                "to already be current density, but the unit must be confirmed before using current-dependent metrics."
            )
        return (
            "The current column has no unit in the header. If interpreted as A, "
            "the current values are unusually large for a typical CV/LSV."
        )
    if suggested_unit == "A":
        return (
            "The current column has no unit in the header. Based on value magnitude, "
            "A is the most likely input unit. Please confirm before using current-dependent metrics."
        )
    if is_density_suggestion:
        return (
            "The current-density signal has no unit in the header. Based on filename/metadata "
            f"and value magnitude, {suggested_label} is the most likely input unit. "
            "Please confirm before using current-density metrics."
        )
    return (
        f"The current column '{current_col_label}' has no unit in the header. Based on value magnitude, "
        f"{suggested_label} is the most likely input unit. "
        "Please confirm before using current-dependent metrics."
    )


def current_unit_review_reason(suggested_unit: str, current_unit_warning: str) -> str:
    suggested_label = display_unit_label(suggested_unit)
    if current_unit_warning:
        if "/cm" in suggested_label:
            return "Suggested from filename/metadata and value magnitude; uploaded signal appears to be current density."
        return "Suggested based on value magnitude; interpreting as A gives unusually large currents."
    if "/cm" in suggested_label:
        return f"Suggested from filename/metadata and value magnitude; {suggested_label} is plausible."
    if suggested_unit == "A":
        return "Suggested based on value magnitude; A gives microamp-scale currents."
    return f"Suggested based on value magnitude; {suggested_label} gives a plausible current scale."


def current_magnitude_sanity_warning(
    current_base_values: np.ndarray,
    current_source: str = "Current",
    max_abs_current_a_threshold: float = CURRENT_MAGNITUDE_WARNING_A,
    max_abs_current_uA_threshold: float = CURRENT_MAGNITUDE_WARNING_UA,
) -> str:
    if current_source != "Current":
        return ""
    finite_abs_a = np.abs(np.asarray(current_base_values, dtype=float))
    finite_abs_a = finite_abs_a[np.isfinite(finite_abs_a)]
    if not finite_abs_a.size:
        return ""
    max_abs_current_a = float(np.nanmax(finite_abs_a))
    max_abs_current_uA = max_abs_current_a * 1e6
    if max_abs_current_a > max_abs_current_a_threshold or max_abs_current_uA > max_abs_current_uA_threshold:
        return "Current magnitude is unusually large. Check whether the current unit was inferred correctly."
    return ""


def current_magnitude_ack_key(key_prefix: str, filename: str) -> str:
    return f"{key_prefix}::ack_current_magnitude::{filename}"


def missing_unit_header_warning(
    dataset: ParsedDataset,
    potential_col: Optional[str],
    current_col: Optional[str],
    selected_current_unit: str = "Auto",
) -> str:
    missing_current_unit = current_unit_is_ambiguous(dataset, current_col, selected_current_unit)
    if missing_current_unit:
        return "Column units missing from header. Please confirm potential/current units."
    return ""


def unit_header_audit_note(
    current_col: Optional[str],
    selected_current_unit: str,
    current_source: str = "Current",
) -> str:
    if selected_current_unit == "Auto":
        return ""
    if current_header_has_explicit_unit(current_col):
        return ""
    confirmed_unit = current_unit_label_for_summary(selected_current_unit, current_source)
    return (
        "Column units were missing from header. Current unit was confirmed by user as "
        f"{display_unit_label(confirmed_unit)}."
    )


def finite_pair_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=bool)
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def validate_cv_data(filename: str, potential_v: np.ndarray, current_a: np.ndarray) -> list[str]:
    warnings = []
    finite_mask = np.isfinite(potential_v) & np.isfinite(current_a)
    valid_rows = int(np.sum(finite_mask))

    if valid_rows < 5:
        warnings.append(f"{filename}: too few numeric rows for reliable CV/LSV analysis.")
        return warnings

    potential_range = float(np.nanmax(potential_v[finite_mask]) - np.nanmin(potential_v[finite_mask]))
    current_range = float(np.nanmax(current_a[finite_mask]) - np.nanmin(current_a[finite_mask]))
    if potential_range == 0:
        warnings.append(f"{filename}: potential values are constant.")
    elif potential_range > 20:
        warnings.append(f"{filename}: potential range is unusually large after unit conversion; check potential units.")
    if current_range == 0:
        warnings.append(f"{filename}: current values are constant.")

    return warnings


def normalize_dataset(
    dataset: ParsedDataset,
    potential_col: str,
    current_col: str,
    selected_potential_unit: str,
    selected_current_unit: str,
    current_source: str,
    smoothing_method: str,
    smoothing_window: int,
) -> AnalyzedDataset:
    cycle_col = find_cycle_column(dataset)
    working_cols = [potential_col, current_col]
    if cycle_col and cycle_col not in working_cols:
        working_cols.append(cycle_col)
    working = dataset.dataframe[working_cols].dropna(subset=[potential_col, current_col]).copy()
    potential_values = working[potential_col].to_numpy(dtype=float)
    current_values = working[current_col].to_numpy(dtype=float)
    cycle_values = working[cycle_col].to_numpy(dtype=float) if cycle_col and cycle_col in working else None
    cycle_source = "column" if cycle_values is not None else "none"

    potential_unit = resolve_unit(selected_potential_unit, dataset, "potential", potential_col, "V")
    if selected_potential_unit == "Auto" and not dataset.detected_units.get(f"potential:{potential_col}"):
        potential_unit = infer_potential_unit_from_values(potential_values)
    current_unit = resolve_unit(selected_current_unit, dataset, "current", current_col, "A")

    potential_v = convert_potential_to_volts(potential_values, potential_unit)
    raw_current_a = convert_current_to_amps(current_values, current_unit)

    finite_mask = np.isfinite(potential_v) & np.isfinite(raw_current_a)
    potential_v = potential_v[finite_mask]
    raw_current_a = raw_current_a[finite_mask]
    if cycle_values is not None:
        cycle_values = cycle_values[finite_mask]
    else:
        inferred_cycle_values = infer_cycle_values_from_potential(potential_v)
        if inferred_cycle_values is not None:
            cycle_values = inferred_cycle_values
            cycle_source = "inferred"

    smoothed_current_a = None
    if smoothing_method != "None" and len(raw_current_a) >= 5:
        smoothed_current_a = smooth_current(raw_current_a, smoothing_method, smoothing_window)

    warnings = validate_cv_data(dataset.filename, potential_v, raw_current_a)
    source_warnings = [
        warning
        for warning in dataset.warnings
        if LEGACY_APPROXIMATE_CYCLE_WARNING not in warning
    ]

    return AnalyzedDataset(
        filename=dataset.filename,
        sample_name=dataset.filename.rsplit(".", 1)[0],
        potential_col=potential_col,
        current_col=current_col,
        potential_unit=potential_unit,
        current_unit=current_unit,
        current_source=current_source,
        potential_v=potential_v,
        raw_current_a=raw_current_a,
        smoothed_current_a=smoothed_current_a,
        cycle_values=cycle_values,
        cycle_source=cycle_source,
        metadata=dataset.metadata,
        warnings=source_warnings + warnings,
    )


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if len(values) < window_size:
        return values.copy()
    kernel = np.ones(window_size) / window_size
    padded = np.pad(values, (window_size // 2, window_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def savitzky_golay_smooth(values: np.ndarray, window_size: int, poly_order: int = 2) -> np.ndarray:
    window_size = max(poly_order + 3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if len(values) < window_size:
        return values.copy()

    half_window = window_size // 2
    smoothed = np.empty_like(values, dtype=float)

    for idx in range(len(values)):
        start = max(0, idx - half_window)
        end = min(len(values), idx + half_window + 1)
        local_y = values[start:end]
        local_x = np.arange(start, end) - idx
        degree = min(poly_order, len(local_y) - 1)
        if degree < 1:
            smoothed[idx] = values[idx]
            continue
        coefficients = np.polyfit(local_x, local_y, degree)
        smoothed[idx] = np.polyval(coefficients, 0)

    return smoothed


def smooth_current(values: np.ndarray, method: str, window_size: int) -> np.ndarray:
    if method == "Moving average":
        return moving_average(values, window_size)
    if method == "Savitzky-Golay":
        return savitzky_golay_smooth(values, window_size)
    return values.copy()


def find_threshold_crossing(
    potential: np.ndarray,
    current: np.ndarray,
    threshold: float,
    direction: str,
) -> Optional[float]:
    if direction not in {"anodic", "cathodic"}:
        raise ValueError("direction must be 'anodic' or 'cathodic'")

    target = threshold if direction == "anodic" else -threshold
    order = np.argsort(potential)
    x = potential[order]
    y = current[order]
    crossings = []

    for idx in range(1, len(x)):
        x1, x2 = x[idx - 1], x[idx]
        y1, y2 = y[idx - 1], y[idx]

        if y1 == target:
            crossings.append(float(x1))
            continue
        if y2 == target:
            crossings.append(float(x2))
            continue
        if (y1 - target) * (y2 - target) >= 0:
            continue

        crossing = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        crossings.append(float(crossing))

    if crossings:
        return min(crossings) if direction == "anodic" else max(crossings)

    mask = y >= target if direction == "anodic" else y <= target
    if not np.any(mask):
        return None

    threshold_points = x[mask]
    return float(np.min(threshold_points)) if direction == "anodic" else float(np.max(threshold_points))


def calculate_esw(potential: np.ndarray, current: np.ndarray, threshold: float) -> ESWResult:
    cathodic = find_threshold_crossing(potential, current, threshold, direction="cathodic")
    anodic = find_threshold_crossing(potential, current, threshold, direction="anodic")

    if cathodic is None or anodic is None:
        return ESWResult(cathodic, anodic, None)
    return ESWResult(cathodic, anodic, anodic - cathodic)


# ---------- Peak and baseline analysis ----------


LEGACY_APPROXIMATE_CYCLE_WARNING = "multiple potential turning points detected; cycle splitting is approximate"


def detect_potential_turning_points(potential: np.ndarray) -> list[int]:
    if len(potential) < 3:
        return []

    differences = np.diff(potential)
    signs = np.sign(differences)
    turning_points = []

    previous_sign = 0
    for idx, sign in enumerate(signs, start=1):
        if sign == 0:
            continue
        if previous_sign != 0 and sign != previous_sign:
            turning_points.append(idx)
        previous_sign = sign

    return turning_points


def split_scan_segments(potential: np.ndarray) -> list[tuple[int, int, str]]:
    turning_points = detect_potential_turning_points(potential)
    breakpoints = [0] + turning_points + [len(potential) - 1]
    segments = []

    for segment_idx in range(len(breakpoints) - 1):
        start = breakpoints[segment_idx]
        end = breakpoints[segment_idx + 1] + 1
        if end - start < 3:
            continue
        direction = "forward" if potential[end - 1] >= potential[start] else "reverse"
        segments.append((start, end, direction))

    return segments


def infer_cycle_values_from_potential(potential: np.ndarray) -> Optional[np.ndarray]:
    segments = split_scan_segments(potential)
    if len(segments) < 3:
        return None

    cycle_values = np.full(len(potential), np.nan, dtype=float)
    for segment_index, (start, end, _direction) in enumerate(segments):
        cycle_values[start:end] = float(segment_index // 2 + 1)

    if np.any(~np.isfinite(cycle_values)):
        finite_cycles = cycle_values[np.isfinite(cycle_values)]
        fill_value = float(finite_cycles[0]) if finite_cycles.size else 1.0
        cycle_values[~np.isfinite(cycle_values)] = fill_value

    if len(np.unique(cycle_values[np.isfinite(cycle_values)])) < 2:
        return None
    return cycle_values


def estimate_peak_prominence(current: np.ndarray, idx: int, peak_type: str, window_size: int) -> float:
    start = max(0, idx - window_size)
    end = min(len(current), idx + window_size + 1)
    left = current[start:idx]
    right = current[idx + 1 : end]
    if len(left) == 0 or len(right) == 0:
        return 0.0

    if peak_type == "oxidation":
        local_baseline = max(float(np.nanmin(left)), float(np.nanmin(right)))
        return max(0.0, float(current[idx] - local_baseline))

    local_baseline = min(float(np.nanmax(left)), float(np.nanmax(right)))
    return max(0.0, float(local_baseline - current[idx]))


def confidence_from_prominence(prominence: float, min_prominence: float) -> str:
    if min_prominence <= 0:
        return "medium"
    if prominence >= 5 * min_prominence:
        return "high"
    if prominence >= 2 * min_prominence:
        return "medium"
    return "low"


def enforce_min_distance(peaks: list[Peak], min_distance: int) -> list[Peak]:
    selected: list[Peak] = []
    for peak in sorted(peaks, key=lambda item: item.prominence, reverse=True):
        if all(abs(peak.index - existing.index) >= min_distance for existing in selected):
            selected.append(peak)
    return sorted(selected, key=lambda item: item.index)


def segment_inner_bounds(start: int, end: int, edge_fraction: float = 0.05) -> tuple[int, int]:
    segment_length = end - start
    if segment_length < 8:
        return start + 1, end - 1
    edge_points = max(2, int(segment_length * edge_fraction))
    inner_start = min(end - 2, start + edge_points)
    inner_end = max(inner_start + 1, end - edge_points)
    return inner_start, inner_end


def expected_peak_direction(peak_type: str) -> Optional[str]:
    if peak_type == "oxidation":
        return "forward"
    if peak_type == "reduction":
        return "reverse"
    return None


def peak_has_expected_current_sign(peak_type: str, current_value: float) -> bool:
    if peak_type == "oxidation":
        return current_value > 0
    if peak_type == "reduction":
        return current_value < 0
    return False


def peak_matches_scan_or_current_sign(peak_type: str, segment_direction: str, current_value: float) -> bool:
    expected_direction = expected_peak_direction(peak_type)
    return segment_direction == expected_direction or peak_has_expected_current_sign(peak_type, current_value)


def confidence_rank(confidence: str) -> int:
    ranks = {
        "low": 0,
        "medium": 1,
        "medium-high": 2,
        "high": 3,
        "manual": 4,
    }
    return ranks.get(confidence, 0)


def cap_confidence(confidence: str, max_confidence: str) -> str:
    return confidence if confidence_rank(confidence) <= confidence_rank(max_confidence) else max_confidence


CANDIDATE_PEAK_FILTER_OPTIONS = [
    "Selected Epa/Epc only",
    "Top candidates",
    "High-confidence candidates",
    "All candidate extrema",
]


def candidate_peak_rank_key(peak: Peak, potential: Optional[np.ndarray] = None) -> tuple[float, float, float, int]:
    not_near_edge_bonus = 0.0
    if potential is not None and not peak_is_near_segment_endpoint(potential, peak):
        not_near_edge_bonus = max(abs(float(peak.raw_current)), float(peak.prominence), 1e-30)
    combined_score = float(peak.prominence) + abs(float(peak.raw_current)) + not_near_edge_bonus
    return (combined_score, abs(float(peak.raw_current)), float(peak.prominence), confidence_rank(peak.confidence))


def unique_peaks_by_id(peaks: list[Peak]) -> list[Peak]:
    seen: set[str] = set()
    unique: list[Peak] = []
    for peak in peaks:
        if peak.id in seen:
            continue
        seen.add(peak.id)
        unique.append(peak)
    return unique


def filter_candidate_peaks_for_display(
    peaks: list[Peak],
    selected_oxidation_peak: Optional[Peak],
    selected_reduction_peak: Optional[Peak],
    display_filter: str,
    top_n: int = 5,
    potential: Optional[np.ndarray] = None,
) -> list[Peak]:
    selected = [peak for peak in [selected_oxidation_peak, selected_reduction_peak] if peak is not None]
    selected_ids = {peak.id for peak in selected}

    if display_filter == "Selected Epa/Epc only":
        return unique_peaks_by_id(selected)

    if display_filter == "High-confidence candidates":
        return [peak for peak in peaks if peak.confidence == "high"]

    if display_filter == "All candidate extrema":
        return peaks

    ranked_non_selected = sorted(
        [peak for peak in peaks if peak.id not in selected_ids],
        key=lambda peak: candidate_peak_rank_key(peak, potential),
        reverse=True,
    )
    return unique_peaks_by_id(selected + ranked_non_selected[:top_n])


def is_index_near_segment_endpoint(start: int, end: int, idx: int, edge_fraction: float = 0.05) -> bool:
    inner_start, inner_end = segment_inner_bounds(start, end, edge_fraction)
    return idx <= inner_start or idx >= inner_end - 1


def peak_is_near_segment_endpoint(potential: np.ndarray, peak: Peak) -> bool:
    if peak.segment_direction == "manual":
        segments = split_scan_segments(potential)
    else:
        segments = [
            (start, end, direction)
            for index, (start, end, direction) in enumerate(split_scan_segments(potential), start=1)
            if index == peak.segment_index
        ]

    for start, end, _direction in segments:
        if start <= peak.index < end:
            return is_index_near_segment_endpoint(start, end, peak.index)
    return True


def confidence_from_peak_context(
    current: np.ndarray,
    idx: int,
    peak_type: str,
    segment_direction: str,
    start: int,
    end: int,
    prominence: float,
    min_prominence: float,
    min_abs_current: float,
) -> str:
    correct_segment = peak_matches_scan_or_current_sign(peak_type, segment_direction, float(current[idx]))
    near_edge = is_index_near_segment_endpoint(start, end, idx)

    segment_current = current[start:end]
    finite = segment_current[np.isfinite(segment_current)]
    if finite.size < 5 or not np.isfinite(current[idx]):
        return "low"

    segment_min = float(np.nanmin(finite))
    segment_max = float(np.nanmax(finite))
    segment_range = max(segment_max - segment_min, 1e-15)
    noise_floor = max(robust_current_noise(finite), segment_range * 0.005, 1e-15)
    peak_current = float(current[idx])

    if peak_type == "oxidation":
        dominance = (peak_current - segment_min) / segment_range
        magnitude = max(abs(peak_current), max(0.0, peak_current - float(np.nanmedian(finite))))
    else:
        dominance = (segment_max - peak_current) / segment_range
        magnitude = max(abs(peak_current), max(0.0, float(np.nanmedian(finite)) - peak_current))

    prominence_to_noise = prominence / noise_floor
    magnitude_to_noise = magnitude / noise_floor

    score = 0
    score += 2 if correct_segment else -3
    score += 1 if not near_edge else -3

    if prominence_to_noise >= 5 or magnitude_to_noise >= 15:
        score += 3
    elif prominence_to_noise >= 3 or magnitude_to_noise >= 8:
        score += 2
    elif prominence_to_noise >= 1.5 or magnitude_to_noise >= 4:
        score += 1
    else:
        score -= 2

    if dominance >= 0.75:
        score += 3
    elif dominance >= 0.5:
        score += 1
    else:
        score -= 2

    if min_prominence > 0:
        if prominence >= 1.25 * min_prominence:
            score += 1
        elif prominence < min_prominence:
            score -= 2

    if min_abs_current > 0 and abs(peak_current) < 1.5 * min_abs_current:
        score -= 1

    if near_edge:
        return "low"
    if not correct_segment:
        return "medium" if score >= 4 else "low"
    if dominance < 0.4 and score >= 4:
        return "medium"
    if score >= 7:
        confidence = "high"
    elif score >= 5:
        confidence = "medium-high"
    elif score >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    if prominence_to_noise < 3:
        return cap_confidence(confidence, "low")
    if prominence_to_noise < 5:
        return cap_confidence(confidence, "medium")
    if prominence_to_noise < 8:
        return cap_confidence(confidence, "medium-high")
    return confidence


def peak_selection_strength(peak: Peak) -> float:
    directional_current = float(peak.raw_current) if peak.peak_type == "oxidation" else float(-peak.raw_current)
    return max(0.0, directional_current) + float(peak.prominence)


def apply_candidate_ambiguity_confidence(peaks: list[Peak], potential: np.ndarray) -> None:
    for peak_type in ["oxidation", "reduction"]:
        candidates = [
            peak
            for peak in peaks
            if peak.peak_type == peak_type
            and peak_has_expected_current_sign(peak_type, peak.raw_current)
            and not peak_is_near_segment_endpoint(potential, peak)
            and peak.confidence != "manual"
        ]
        if len(candidates) < 3:
            continue

        selected = max(candidates, key=peak_selection_strength)
        selected_strength = max(peak_selection_strength(selected), 1e-30)
        selected_prominence = max(float(selected.prominence), 1e-30)
        similar_candidates = [
            peak
            for peak in candidates
            if peak.id != selected.id
            and (
                float(peak.prominence) >= 0.75 * selected_prominence
                or peak_selection_strength(peak) >= 0.75 * selected_strength
            )
        ]
        if len(similar_candidates) >= 2:
            selected.confidence = cap_confidence(selected.confidence, "medium")
            for peak in similar_candidates:
                peak.confidence = cap_confidence(peak.confidence, "medium")
        elif similar_candidates:
            selected.confidence = cap_confidence(selected.confidence, "medium-high")


def segment_extreme_peak(
    potential: np.ndarray,
    current: np.ndarray,
    start: int,
    end: int,
    direction: str,
    segment_index: int,
    min_prominence: float,
    min_abs_current: float,
    prominence_window: int,
) -> Optional[Peak]:
    if direction not in {"forward", "reverse"}:
        return None

    inner_start, inner_end = segment_inner_bounds(start, end)
    if inner_end <= inner_start:
        return None

    segment_current = current[inner_start:inner_end]
    finite_mask = np.isfinite(segment_current)
    if not np.any(finite_mask):
        return None

    if direction == "forward":
        local_positions = np.where(finite_mask)[0]
        local_idx = int(local_positions[np.nanargmax(segment_current[finite_mask])])
        peak_type = "oxidation"
    else:
        local_positions = np.where(finite_mask)[0]
        local_idx = int(local_positions[np.nanargmin(segment_current[finite_mask])])
        peak_type = "reduction"

    idx = inner_start + local_idx
    current_value = float(current[idx])
    if abs(current_value) < min_abs_current:
        return None

    prominence = estimate_peak_prominence(current, idx, peak_type, prominence_window)
    if prominence < min_prominence:
        return None

    return Peak(
        id="",
        peak_type=peak_type,
        index=idx,
        potential=float(potential[idx]),
        raw_current=current_value,
        prominence=prominence,
        segment_index=segment_index,
        segment_direction=direction,
        confidence=confidence_from_peak_context(
            current,
            idx,
            peak_type,
            direction,
            start,
            end,
            prominence,
            min_prominence,
            min_abs_current,
        ),
    )


def robust_current_noise(current: np.ndarray) -> float:
    finite = current[np.isfinite(current)]
    if finite.size < 4:
        return 0.0
    differences = np.diff(finite)
    differences = differences[np.isfinite(differences)]
    if differences.size < 3:
        return 0.0
    median = float(np.nanmedian(differences))
    mad = float(np.nanmedian(np.abs(differences - median)))
    return 1.4826 * mad / np.sqrt(2)


def auto_tune_cv_peak_parameters(potential: np.ndarray, current: np.ndarray) -> dict[str, float]:
    finite_current = current[np.isfinite(current)]
    current_range = float(np.nanmax(finite_current) - np.nanmin(finite_current)) if finite_current.size else 0.0
    tiny_current = max(current_range * 1e-6, 1e-15)

    segments = split_scan_segments(potential)
    if not segments:
        segments = [(0, len(potential), "scan")]

    segment_lengths = [end - start for start, end, _direction in segments if end > start]
    min_segment_length = min(segment_lengths) if segment_lengths else len(potential)
    min_distance = int(max(5, min(35, round(min_segment_length * 0.03))))
    prominence_window = max(3, min_distance)

    segment_prominences = []
    for segment_index, (start, end, direction) in enumerate(segments, start=1):
        peak = segment_extreme_peak(
            potential,
            current,
            start,
            end,
            direction,
            segment_index,
            min_prominence=0.0,
            min_abs_current=0.0,
            prominence_window=prominence_window,
        )
        if peak is not None and np.isfinite(peak.prominence) and peak.prominence > 0:
            segment_prominences.append(float(peak.prominence))

    noise_floor = robust_current_noise(current)
    range_floor = current_range * 0.002
    if segment_prominences:
        weakest_segment_peak = min(segment_prominences)
        min_prominence = min(max(noise_floor * 4, range_floor, tiny_current), weakest_segment_peak * 0.6)
    else:
        min_prominence = max(noise_floor * 4, current_range * 0.02, tiny_current)

    return {
        "min_prominence": float(max(min_prominence, 0.0)),
        "min_distance": float(max(min_distance, 1)),
        "min_abs_current": 0.0,
    }


def detect_peaks(
    potential: np.ndarray,
    current: np.ndarray,
    min_prominence: float,
    min_distance: int,
    min_abs_current: float,
) -> list[Peak]:
    segments = split_scan_segments(potential)
    if not segments:
        segments = [(0, len(potential), "scan")]

    candidate_peaks: list[Peak] = []
    prominence_window = max(3, min_distance)

    for segment_index, (start, end, direction) in enumerate(segments, start=1):
        segment_peak = segment_extreme_peak(
            potential,
            current,
            start,
            end,
            direction,
            segment_index,
            min_prominence,
            min_abs_current,
            prominence_window,
        )
        if segment_peak is not None:
            candidate_peaks.append(segment_peak)

        inner_start, inner_end = segment_inner_bounds(start, end)
        for idx in range(inner_start, inner_end):
            current_value = current[idx]
            if not np.isfinite(current_value) or abs(current_value) < min_abs_current:
                continue

            if current[idx] > current[idx - 1] and current[idx] > current[idx + 1]:
                prominence = estimate_peak_prominence(current, idx, "oxidation", prominence_window)
                if prominence >= min_prominence:
                    candidate_peaks.append(
                        Peak(
                            id="",
                            peak_type="oxidation",
                            index=idx,
                            potential=float(potential[idx]),
                            raw_current=float(current_value),
                            prominence=prominence,
                            segment_index=segment_index,
                            segment_direction=direction,
                            confidence=confidence_from_peak_context(
                                current,
                                idx,
                                "oxidation",
                                direction,
                                start,
                                end,
                                prominence,
                                min_prominence,
                                min_abs_current,
                            ),
                        )
                    )

            if current[idx] < current[idx - 1] and current[idx] < current[idx + 1]:
                prominence = estimate_peak_prominence(current, idx, "reduction", prominence_window)
                if prominence >= min_prominence:
                    candidate_peaks.append(
                        Peak(
                            id="",
                            peak_type="reduction",
                            index=idx,
                            potential=float(potential[idx]),
                            raw_current=float(current_value),
                            prominence=prominence,
                            segment_index=segment_index,
                            segment_direction=direction,
                            confidence=confidence_from_peak_context(
                                current,
                                idx,
                                "reduction",
                                direction,
                                start,
                                end,
                                prominence,
                                min_prominence,
                                min_abs_current,
                            ),
                        )
                    )

    filtered = enforce_min_distance(candidate_peaks, min_distance)
    type_counts = {"oxidation": 0, "reduction": 0}
    for peak in filtered:
        type_counts[peak.peak_type] += 1
        prefix = "ox" if peak.peak_type == "oxidation" else "red"
        peak.id = f"{prefix}_{type_counts[peak.peak_type]}"
    apply_candidate_ambiguity_confidence(filtered, potential)
    return filtered


def default_primary_peak_index(peaks: list[Peak], peak_type: str) -> int:
    expected_direction = expected_peak_direction(peak_type)

    def sort_key(index: int) -> tuple[bool, float, bool, float, int]:
        peak = peaks[index]
        peak_height = peak_selection_strength(peak)
        return (
            peak_has_expected_current_sign(peak_type, peak.raw_current),
            peak_height,
            peak.segment_direction == expected_direction,
            float(peak.prominence),
            confidence_rank(peak.confidence),
        )

    return max(range(len(peaks)), key=sort_key)


def calculate_linear_baseline(
    potential: np.ndarray,
    current: np.ndarray,
    anchor_a_idx: int,
    anchor_b_idx: int,
) -> Optional[np.ndarray]:
    e1 = float(potential[anchor_a_idx])
    e2 = float(potential[anchor_b_idx])
    i1 = float(current[anchor_a_idx])
    i2 = float(current[anchor_b_idx])

    if e1 == e2:
        return None

    slope = (i2 - i1) / (e2 - e1)
    intercept = i1 - slope * e1
    return slope * potential + intercept


def peak_label(
    peak: Peak,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
    current_source: str = "Current",
) -> str:
    current_display = current_to_display(np.array([peak.raw_current]), display_unit, current_source, electrode_area_cm2)[0]
    prominence_display = current_to_display(np.array([peak.prominence]), display_unit, current_source, electrode_area_cm2)[0]
    display_label = display_unit_label(display_unit)
    return (
        f"{peak.id}: {peak.peak_type}, E={peak.potential:.4g} V, "
        f"I={current_display:.4g} {display_label}, prom={prominence_display:.4g} {display_label}"
    )


def make_manual_peak(peak_type: str, idx: int, potential: np.ndarray, current: np.ndarray) -> Peak:
    prefix = "manual_ox" if peak_type == "oxidation" else "manual_red"
    return Peak(
        id=prefix,
        peak_type=peak_type,
        index=idx,
        potential=float(potential[idx]),
        raw_current=float(current[idx]),
        prominence=0.0,
        segment_index=0,
        segment_direction="manual",
        confidence="manual",
    )


def scan_direction_label(direction: str) -> str:
    if direction == "forward":
        return "Forward"
    if direction == "reverse":
        return "Reverse"
    if direction == "manual":
        return "Manual"
    return "Scan"


def local_extremum_label(peak: Peak) -> str:
    if peak.peak_type == "oxidation":
        return "local maximum"
    if peak.peak_type == "reduction":
        return "local minimum"
    return "local extremum"


def peak_selection_method_label(metric_name: str, peak: Peak) -> str:
    if metric_name == "oxidation":
        if peak.segment_direction == "forward":
            return "Maximum current on forward scan"
        if peak.segment_direction == "manual":
            return "Manually selected oxidation peak"
        return "Maximum current candidate selected as oxidation peak"
    if peak.segment_direction == "reverse":
        return "Minimum current on reverse scan"
    if peak.segment_direction == "manual":
        return "Manually selected reduction peak"
    return "Minimum current candidate selected as reduction peak"


def select_default_primary_peaks(peaks: list[Peak]) -> tuple[Optional[Peak], Optional[Peak]]:
    oxidation_peaks = [peak for peak in peaks if peak.peak_type == "oxidation"]
    reduction_peaks = [peak for peak in peaks if peak.peak_type == "reduction"]
    oxidation_peak = None
    reduction_peak = None
    if oxidation_peaks:
        oxidation_peak = oxidation_peaks[default_primary_peak_index(oxidation_peaks, "oxidation")]
    if reduction_peaks:
        reduction_peak = reduction_peaks[default_primary_peak_index(reduction_peaks, "reduction")]
    return oxidation_peak, reduction_peak


def build_peak_metrics_rows(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    baseline_current: Optional[np.ndarray],
    analysis_current: np.ndarray,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
    current_source: str = "Current",
) -> tuple[list[dict[str, Any]], dict[str, Optional[float]]]:
    rows = []
    metrics: dict[str, Optional[float]] = {
        "epa_V": None,
        "ipa_A": None,
        "epc_V": None,
        "ipc_A": None,
        "delta_ep_V": None,
        "ipa_ipc_ratio": None,
    }

    for metric_name, peak in [("oxidation", oxidation_peak), ("reduction", reduction_peak)]:
        if peak is None:
            continue
        display_label = display_unit_label(display_unit)
        baseline_at_peak = 0.0 if baseline_current is None else float(baseline_current[peak.index])
        raw_current = float(analysis_current[peak.index])
        corrected_current = raw_current - baseline_at_peak
        raw_display = current_to_display(np.array([raw_current]), display_unit, current_source, electrode_area_cm2)[0]
        baseline_display = current_to_display(np.array([baseline_at_peak]), display_unit, current_source, electrode_area_cm2)[0]
        corrected_display = current_to_display(
            np.array([corrected_current]), display_unit, current_source, electrode_area_cm2
        )[0]

        rows.append(
            {
                "peak": metric_name,
                "potential_V": peak.potential,
                f"current_{display_label}": raw_display,
                f"baseline_current_{display_label}": baseline_display,
                f"corrected_current_{display_label}": corrected_display,
                "scan": scan_direction_label(peak.segment_direction),
                "confidence": peak.confidence,
                "method": peak_selection_method_label(metric_name, peak),
            }
        )

        if metric_name == "oxidation":
            metrics["epa_V"] = peak.potential
            metrics["ipa_A"] = corrected_current
        else:
            metrics["epc_V"] = peak.potential
            metrics["ipc_A"] = corrected_current

    if metrics["epa_V"] is not None and metrics["epc_V"] is not None:
        metrics["delta_ep_V"] = metrics["epa_V"] - metrics["epc_V"]
    if metrics["ipa_A"] is not None and metrics["ipc_A"] not in {None, 0}:
        metrics["ipa_ipc_ratio"] = abs(metrics["ipa_A"] / metrics["ipc_A"])

    return rows, metrics


def cv_pair_metrics_review_reason(
    metrics: dict[str, Optional[float]],
    pair_confidence: Optional[str],
) -> Optional[str]:
    epa = metrics.get("epa_V")
    epc = metrics.get("epc_V")
    if epa is None or epc is None:
        return None
    if epa < epc:
        return (
            "Invalid: Reversible-pair metrics are invalid because the selected oxidation peak occurs "
            "at a lower potential than the selected reduction peak. Review candidate peaks before "
            "using \u0394Ep, E\u00b0\u2032, or |Ipa/Ipc|."
        )
    confidence = str(pair_confidence or "").strip()
    if confidence.lower() not in {"high", "manual"}:
        return (
            f"Tentative: pair confidence is {confidence or 'not high'}; verify peak assignment before "
            "interpreting reversible-pair metrics."
        )
    return None


def format_reviewed_pair_metric(value: str, review_reason: Optional[str]) -> str:
    if not review_reason:
        return value
    if review_reason.startswith("Invalid"):
        return "Invalid"
    return f"Tentative: {value}"


def metrics_table(
    metrics: dict[str, Optional[float]],
    display_unit: str,
    electrode_area_cm2: float = 1.0,
    current_source: str = "Current",
    cv_behavior: Optional[str] = None,
    pair_confidence: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    display_label = display_unit_label(display_unit)
    pair_metric_review_reason = cv_pair_metrics_review_reason(metrics, pair_confidence)
    if metrics["epa_V"] is not None:
        epa_method = (
            "Selected oxidation peak candidate"
            if pair_metric_review_reason and pair_metric_review_reason.startswith("Invalid")
            else "Maximum current on forward scan"
        )
        rows.append({"metric": "Epa", "value": f"{metrics['epa_V']:.4g} V", "method": epa_method})
    if metrics["ipa_A"] is not None:
        ipa_display = current_to_display(np.array([metrics["ipa_A"]]), display_unit, current_source, electrode_area_cm2)[0]
        rows.append({"metric": "Ipa", "value": f"{ipa_display:.4g} {display_label}", "method": "Current at Epa"})
    if metrics["epc_V"] is not None:
        epc_method = (
            "Selected reduction peak candidate"
            if pair_metric_review_reason and pair_metric_review_reason.startswith("Invalid")
            else "Minimum current on reverse scan"
        )
        rows.append({"metric": "Epc", "value": f"{metrics['epc_V']:.4g} V", "method": epc_method})
    if metrics["ipc_A"] is not None:
        ipc_display = current_to_display(np.array([metrics["ipc_A"]]), display_unit, current_source, electrode_area_cm2)[0]
        rows.append({"metric": "Ipc", "value": f"{ipc_display:.4g} {display_label}", "method": "Current at Epc"})
    if metrics["delta_ep_V"] is not None:
        rows.append(
            {
                "metric": "\u0394Ep",
                "value": format_reviewed_pair_metric(f"{metrics['delta_ep_V'] * 1000:.4g} mV", pair_metric_review_reason),
                "method": "Epa - Epc" if not pair_metric_review_reason else f"Epa - Epc; {pair_metric_review_reason}",
            }
        )
    if metrics["epa_V"] is not None and metrics["epc_V"] is not None:
        formal_potential = (metrics["epa_V"] + metrics["epc_V"]) / 2
        rows.append(
            {
                "metric": "E\u00b0\u2032",
                "value": format_reviewed_pair_metric(f"{formal_potential:.4g} V", pair_metric_review_reason),
                "method": "(Epa + Epc) / 2" if not pair_metric_review_reason else f"(Epa + Epc) / 2; {pair_metric_review_reason}",
            }
        )
    if metrics["ipa_ipc_ratio"] is not None:
        rows.append(
            {
                "metric": "|Ipa/Ipc|",
                "value": format_reviewed_pair_metric(f"{metrics['ipa_ipc_ratio']:.4g}", pair_metric_review_reason),
                "method": "abs(Ipa/Ipc)" if not pair_metric_review_reason else f"abs(Ipa/Ipc); {pair_metric_review_reason}",
            }
        )
    if cv_behavior == "irreversible_oxidation_only":
        rows.extend(
            [
                {
                    "metric": "Epc",
                    "value": "No reliable cathodic peak",
                    "method": "No reliable return peak detected on reverse scan",
                },
                {"metric": "Ipc", "value": "Not detected", "method": "No reliable cathodic peak"},
                {
                    "metric": "\u0394Ep",
                    "value": "Not applicable",
                    "method": "Requires a reliable Epa/Epc pair",
                },
                {
                    "metric": "E\u00b0\u2032",
                    "value": "Not applicable",
                    "method": "Requires a reliable Epa/Epc pair",
                },
                {
                    "metric": "|Ipa/Ipc|",
                    "value": "Not meaningful",
                    "method": "Requires a reliable cathodic return peak",
                },
            ]
        )
    elif cv_behavior == "irreversible_reduction_only":
        rows.extend(
            [
                {
                    "metric": "Epa",
                    "value": "No reliable anodic peak",
                    "method": "No reliable oxidation peak detected on forward scan",
                },
                {"metric": "Ipa", "value": "Not detected", "method": "No reliable anodic peak"},
                {
                    "metric": "\u0394Ep",
                    "value": "Not applicable",
                    "method": "Requires a reliable Epa/Epc pair",
                },
                {
                    "metric": "E\u00b0\u2032",
                    "value": "Not applicable",
                    "method": "Requires a reliable Epa/Epc pair",
                },
                {
                    "metric": "|Ipa/Ipc|",
                    "value": "Not meaningful",
                    "method": "Requires a reliable anodic return peak",
                },
            ]
        )
    return pd.DataFrame(rows)


def format_voltage_table_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.4f}"


def format_current_table_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.2f}"


def selected_peak_metrics_display_table(
    peak_metric_df: pd.DataFrame,
    display_unit: str,
    baseline_active: bool,
    review_needed: bool = False,
) -> pd.DataFrame:
    display_label = display_unit_label(display_unit)
    current_col = f"current_{display_label}"
    baseline_col = f"baseline_current_{display_label}"
    corrected_col = f"corrected_current_{display_label}"

    if peak_metric_df.empty:
        return peak_metric_df

    if baseline_active:
        display_df = peak_metric_df.copy()
        if current_col in display_df.columns:
            display_df = display_df.rename(columns={current_col: f"raw_current_{display_label}"})
        columns = [
            "peak",
            "potential_V",
            f"raw_current_{display_label}",
            baseline_col,
            corrected_col,
            "scan",
            "confidence",
            "method",
        ]
    else:
        display_df = peak_metric_df.copy()
        columns = [
            "peak",
            "potential_V",
            current_col,
            "scan",
            "confidence",
            "method",
        ]

    display_df = display_df[[column for column in columns if column in display_df.columns]].copy()
    if "peak" in display_df.columns:
        peak_label_map = (
            {
                "oxidation": "Selected candidate oxidation peak",
                "reduction": "Selected candidate reduction extremum",
            }
            if review_needed
            else {"oxidation": "Oxidation", "reduction": "Reduction"}
        )
        display_df["peak"] = display_df["peak"].map(peak_label_map).fillna(
            display_df["peak"].astype(str).str.title()
        )
    if "potential_V" in display_df.columns:
        display_df["potential_V"] = display_df["potential_V"].map(format_voltage_table_value)
    current_columns = [
        column
        for column in display_df.columns
        if str(column).startswith(("current_", "raw_current_", "baseline_current_", "corrected_current_"))
    ]
    for column in current_columns:
        display_df[column] = display_df[column].map(format_current_table_value)
    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].map(format_confidence_label)

    rename_map = {
        "peak": "Peak",
        "potential_V": "E / V",
        current_col: f"I / {display_label}",
        f"raw_current_{display_label}": f"Raw I / {display_label}",
        baseline_col: f"Baseline I / {display_label}",
        corrected_col: f"Corrected I / {display_label}",
        "scan": "Scan",
        "confidence": "Confidence",
        "method": "Method",
    }
    return display_df.rename(columns={key: value for key, value in rename_map.items() if key in display_df.columns})


def format_current_metric(
    value: Optional[float],
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float,
    needs_unit_review: bool = False,
) -> str:
    if value is None or not np.isfinite(value):
        return "Not detected"
    display_value = current_to_display(np.array([value]), display_unit, current_source, electrode_area_cm2)[0]
    value_text = f"{display_value:.4g} {display_unit_label(display_unit)}"
    if needs_unit_review:
        value_text += " (unit review)"
    return value_text


def pending_current_metric(unit_confidence: str) -> bool:
    return unit_confidence == "Low"


def format_potential_metric(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "Not detected"
    return f"{value:.4g} V"


def candidate_peak_ambiguity_messages(
    candidate_peaks: Optional[list[Peak]],
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
) -> list[str]:
    if not candidate_peaks:
        return []
    messages = []
    for selected_peak in [oxidation_peak, reduction_peak]:
        if selected_peak is None:
            continue
        same_role = [
            peak
            for peak in candidate_peaks
            if peak.id != selected_peak.id
            and peak.peak_type == selected_peak.peak_type
            and peak.segment_direction == selected_peak.segment_direction
        ]
        selected_prominence = max(float(selected_peak.prominence), 1e-30)
        selected_strength = max(peak_selection_strength(selected_peak), 1e-30)
        similar = [
            peak
            for peak in same_role
            if float(peak.prominence) >= 0.75 * selected_prominence
            or peak_selection_strength(peak) >= 0.75 * selected_strength
        ]
        if len(similar) >= 2:
            messages.append(
                "Several candidate extrema have similar prominence. Primary peak selection may be ambiguous."
            )
            break
    return messages


def peak_is_reliable_for_pair(peak: Optional[Peak], peak_type: str, potential: np.ndarray) -> bool:
    if peak is None:
        return False
    if peak.confidence == "manual":
        return True
    if peak.peak_type != peak_type:
        return False
    if not peak_has_expected_current_sign(peak_type, peak.raw_current):
        return False
    if peak_is_near_segment_endpoint(potential, peak):
        return False
    return confidence_rank(peak.confidence) >= confidence_rank("medium-high")


def peak_is_meaningful_for_couple(
    peak: Peak,
    peak_type: str,
    potential: np.ndarray,
    max_abs_current: float,
) -> bool:
    if peak.peak_type != peak_type:
        return False
    if peak_is_near_segment_endpoint(potential, peak):
        return False
    if confidence_rank(peak.confidence) < confidence_rank("medium"):
        return False
    if not peak_has_expected_current_sign(peak_type, peak.raw_current):
        return False
    if max_abs_current > 0 and abs(float(peak.raw_current)) < 0.25 * max_abs_current:
        return False
    return True


def redox_pair_confidence(
    oxidation_peak: Peak,
    reduction_peak: Peak,
    delta_ep_v: float,
    current_ratio: float,
) -> str:
    lower_peak_confidence = min(
        confidence_rank(oxidation_peak.confidence),
        confidence_rank(reduction_peak.confidence),
    )
    if (
        lower_peak_confidence >= confidence_rank("medium-high")
        and 0.6 <= current_ratio <= 1.7
        and 0.02 <= delta_ep_v <= 0.18
    ):
        return "High"
    if (
        lower_peak_confidence >= confidence_rank("medium")
        and 0.45 <= current_ratio <= 2.2
        and 0.015 <= delta_ep_v <= 0.25
    ):
        return "Medium-high"
    return "Medium"


def redox_pair_score(
    oxidation_peak: Peak,
    reduction_peak: Peak,
    delta_ep_v: float,
    current_ratio: float,
) -> float:
    strength = peak_selection_strength(oxidation_peak) + peak_selection_strength(reduction_peak)
    ratio_penalty = min(abs(np.log(max(current_ratio, 1e-12))), np.log(4.0)) / np.log(4.0)
    ratio_score = 1.0 - ratio_penalty
    delta_score = max(0.0, 1.0 - abs(delta_ep_v - 0.08) / 0.30)
    confidence_bonus = (
        confidence_rank(oxidation_peak.confidence) + confidence_rank(reduction_peak.confidence)
    ) * max(strength, 1e-30) * 0.06
    return strength * (0.55 + 0.35 * ratio_score + 0.10 * delta_score) + confidence_bonus


SLOPED_BASELINE_WARNING = (
    "Sloped baseline/background suspected. Try linear baseline correction before interpreting peak current."
)


def detect_sloped_baseline_background(potential: np.ndarray, current: np.ndarray) -> bool:
    finite = np.isfinite(potential) & np.isfinite(current)
    potential = potential[finite]
    current = current[finite]
    if potential.size < 20:
        return False

    order = np.argsort(potential)
    x = potential[order]
    y = current[order]
    span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-15)
    current_range = max(float(np.nanmax(y) - np.nanmin(y)), 1e-15)
    edge_count = max(5, int(0.12 * len(y)))
    low_edge = y[:edge_count]
    high_edge = y[-edge_count:]
    if low_edge.size < 3 or high_edge.size < 3:
        return False

    edge_drift = abs(float(np.nanmedian(high_edge) - np.nanmedian(low_edge)))
    slope = edge_drift / span
    residual = y - np.interp(x, [x[0], x[-1]], [float(np.nanmedian(low_edge)), float(np.nanmedian(high_edge))])
    residual_noise = max(robust_current_noise(residual), 1e-15)
    drift_to_noise = edge_drift / residual_noise

    return edge_drift >= 0.18 * current_range and drift_to_noise >= 6 and slope > 0


def detect_redox_couples(candidate_peaks: Optional[list[Peak]], potential: np.ndarray) -> list[RedoxCouple]:
    if not candidate_peaks:
        return []

    max_abs_current = max((abs(float(peak.raw_current)) for peak in candidate_peaks), default=0.0)
    oxidation_candidates = [
        peak
        for peak in candidate_peaks
        if peak_is_meaningful_for_couple(peak, "oxidation", potential, max_abs_current)
    ]
    reduction_candidates = [
        peak
        for peak in candidate_peaks
        if peak_is_meaningful_for_couple(peak, "reduction", potential, max_abs_current)
    ]

    plausible_pairs: list[RedoxCouple] = []
    for oxidation_peak in oxidation_candidates:
        for reduction_peak in reduction_candidates:
            if max(confidence_rank(oxidation_peak.confidence), confidence_rank(reduction_peak.confidence)) < confidence_rank("medium-high"):
                continue
            delta_ep_v = float(oxidation_peak.potential - reduction_peak.potential)
            if not 0.015 <= delta_ep_v <= 0.35:
                continue
            if reduction_peak.raw_current == 0:
                continue
            current_ratio = abs(float(oxidation_peak.raw_current / reduction_peak.raw_current))
            if not 0.35 <= current_ratio <= 2.8:
                continue

            confidence = redox_pair_confidence(oxidation_peak, reduction_peak, delta_ep_v, current_ratio)
            score = redox_pair_score(oxidation_peak, reduction_peak, delta_ep_v, current_ratio)
            plausible_pairs.append(
                RedoxCouple(
                    id="",
                    oxidation_peak=oxidation_peak,
                    reduction_peak=reduction_peak,
                    delta_ep_v=delta_ep_v,
                    formal_potential_v=(float(oxidation_peak.potential) + float(reduction_peak.potential)) / 2,
                    current_ratio=current_ratio,
                    score=score,
                    confidence=confidence,
                    reason=(
                        f"Paired by scan direction, \u0394Ep {delta_ep_v * 1000:.4g} mV, "
                        f"and |Ipa/Ipc| {current_ratio:.4g}."
                    ),
                )
            )

    paired_oxidation_ids: set[str] = set()
    paired_reduction_ids: set[str] = set()
    selected_pairs: list[RedoxCouple] = []
    for pair in sorted(plausible_pairs, key=lambda item: item.score, reverse=True):
        if pair.oxidation_peak.id in paired_oxidation_ids or pair.reduction_peak.id in paired_reduction_ids:
            continue
        pair.id = f"Couple {len(selected_pairs) + 1}"
        selected_pairs.append(pair)
        paired_oxidation_ids.add(pair.oxidation_peak.id)
        paired_reduction_ids.add(pair.reduction_peak.id)

    return selected_pairs


def behavior_from_redox_couple(
    behavior: CVBehaviorResult,
    selected_index: int,
) -> CVBehaviorResult:
    couples = behavior.redox_couples or []
    if not couples:
        return behavior
    selected_index = max(0, min(selected_index, len(couples) - 1))
    selected_couple = couples[selected_index]
    return CVBehaviorResult(
        behavior.behavior,
        behavior.label,
        selected_couple.oxidation_peak,
        selected_couple.reduction_peak,
        None,
        None,
        selected_couple.confidence,
        behavior.messages,
        couples,
        selected_index,
    )


def classify_cv_behavior(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    metrics: dict[str, Optional[float]],
    potential: np.ndarray,
    candidate_peaks: Optional[list[Peak]] = None,
    baseline_background_suspected: bool = False,
) -> CVBehaviorResult:
    oxidation_reliable = peak_is_reliable_for_pair(oxidation_peak, "oxidation", potential)
    reduction_reliable = peak_is_reliable_for_pair(reduction_peak, "reduction", potential)
    current_ratio = metrics.get("ipa_ipc_ratio")
    high_oxidation_to_reduction_ratio = current_ratio is not None and current_ratio > 2.0
    high_reduction_to_oxidation_ratio = current_ratio is not None and current_ratio < 0.5
    ambiguity_messages = candidate_peak_ambiguity_messages(candidate_peaks, oxidation_peak, reduction_peak)
    redox_couples = detect_redox_couples(candidate_peaks, potential)
    baseline_messages = [SLOPED_BASELINE_WARNING] if baseline_background_suspected else []
    irreversible_baseline_message = (
        "Baseline/background shape may affect return-peak classification; review after baseline correction before "
        "interpreting irreversible behavior."
    )

    if len(redox_couples) >= 2:
        selected_couple = redox_couples[0]
        return CVBehaviorResult(
            "multiple_redox_couples",
            "Multiple redox couples",
            selected_couple.oxidation_peak,
            selected_couple.reduction_peak,
            None,
            None,
            selected_couple.confidence,
            [
                "Multiple redox couples detected. Metrics are calculated for the selected couple; "
                "review pair assignment before interpreting results."
            ]
            + baseline_messages,
            redox_couples,
            0,
        )

    if oxidation_reliable and reduction_reliable:
        if current_ratio is None or 0.5 <= current_ratio <= 2.0:
            return CVBehaviorResult(
                "reversible_like",
                "Reversible-like",
                oxidation_peak,
                reduction_peak,
                None,
                None,
                "High",
                baseline_messages,
            )
        return CVBehaviorResult(
            "quasi_reversible",
            "Quasi-reversible / review needed",
            oxidation_peak,
            reduction_peak,
            None,
            None,
            "Medium",
            [
                f"Peak current ratio looks unusual: |Ipa/Ipc| is {current_ratio:.4g}. "
                "Confirm whether this trace is quasi-reversible, irreversible, or affected by baseline/noise."
            ]
            + baseline_messages,
        )

    if oxidation_reliable and (not reduction_reliable or high_oxidation_to_reduction_ratio):
        return CVBehaviorResult(
            "irreversible_oxidation_only",
            "Irreversible oxidation-only",
            oxidation_peak,
            None,
            None,
            reduction_peak,
            "Low",
            [
                "Irreversible or oxidation-only CV: strong oxidation peak detected, but no reliable cathodic "
                "return peak was found. \u0394Ep, E\u00b0\u2032, and |Ipa/Ipc| are not applicable because no "
                "reliable cathodic return peak was detected."
            ]
            + baseline_messages
            + ([irreversible_baseline_message] if baseline_background_suspected else []),
        )

    if reduction_reliable and (not oxidation_reliable or high_reduction_to_oxidation_ratio):
        return CVBehaviorResult(
            "irreversible_reduction_only",
            "Irreversible reduction-only",
            None,
            reduction_peak,
            oxidation_peak,
            None,
            "Low",
            [
                "Irreversible or reduction-only CV: strong reduction peak detected, but no reliable oxidation peak "
                "was found. \u0394Ep, E\u00b0\u2032, and |Ipa/Ipc| are not applicable because no reliable anodic "
                "return peak was detected."
            ]
            + baseline_messages
            + ([irreversible_baseline_message] if baseline_background_suspected else []),
        )

    if oxidation_peak is None and reduction_peak is None:
        return CVBehaviorResult(
            "no_reliable_peaks",
            "No reliable peaks",
            None,
            None,
            None,
            None,
            "Low",
            ["No reliable primary Epa/Epc peaks were detected. Review the data mapping and peak settings."]
            + baseline_messages,
        )

    messages = [
        "Weak/noisy CV: selected Epa/Epc may be noise-sensitive. Try smoothing, baseline correction, "
        "or manual peak override before interpreting peak currents."
    ]
    messages.extend(baseline_messages)
    messages.extend(ambiguity_messages)
    return CVBehaviorResult(
        "noisy_ambiguous",
        "Noisy / ambiguous",
        oxidation_peak,
        reduction_peak,
        None,
        None,
        "Medium" if oxidation_peak is not None or reduction_peak is not None else "Low",
        unique_messages(messages),
    )


def analyze_cv_default_peaks(
    potential: np.ndarray,
    current: np.ndarray,
    min_prominence: float,
    min_distance: int,
    min_abs_current: float,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
    current_source: str = "Current",
    baseline_current: Optional[np.ndarray] = None,
    baseline_background_suspected: bool = False,
) -> CVPeakAnalysis:
    peaks = detect_peaks(
        potential,
        current,
        min_prominence,
        int(min_distance),
        min_abs_current,
    )
    raw_oxidation_peak, raw_reduction_peak = select_default_primary_peaks(peaks)
    _raw_rows, raw_metrics = build_peak_metrics_rows(
        raw_oxidation_peak,
        raw_reduction_peak,
        baseline_current,
        current,
        display_unit,
        electrode_area_cm2,
        current_source,
    )
    behavior = classify_cv_behavior(
        raw_oxidation_peak,
        raw_reduction_peak,
        raw_metrics,
        potential,
        peaks,
        baseline_background_suspected,
    )
    peak_rows, metrics = build_peak_metrics_rows(
        behavior.oxidation_peak,
        behavior.reduction_peak,
        baseline_current,
        current,
        display_unit,
        electrode_area_cm2,
        current_source,
    )
    return CVPeakAnalysis(
        peaks,
        behavior.oxidation_peak,
        behavior.reduction_peak,
        peak_rows,
        metrics,
        behavior,
    )


def unique_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        unique.append(message)
    return unique


def cv_analysis_quality_messages(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    metrics: Optional[dict[str, Optional[float]]] = None,
    potential: Optional[np.ndarray] = None,
    candidate_peaks: Optional[list[Peak]] = None,
    cv_behavior: Optional[CVBehaviorResult] = None,
) -> list[str]:
    messages = []
    if cv_behavior is not None and cv_behavior.messages:
        messages.extend(cv_behavior.messages)
        if cv_behavior.behavior in {
            "irreversible_oxidation_only",
            "irreversible_reduction_only",
            "multiple_redox_couples",
            "no_reliable_peaks",
            "noisy_ambiguous",
        }:
            return unique_messages(messages)

    if oxidation_peak is None or reduction_peak is None:
        missing = []
        if oxidation_peak is None:
            missing.append("oxidation peak")
        if reduction_peak is None:
            missing.append("reduction peak")
        messages.append(
            f"CV analysis incomplete: selected primary Epa/Epc pair is missing the {', '.join(missing)}. "
            "Review peak selection settings or use manual peak correction."
        )

    if potential is not None:
        endpoint_peaks = [
            peak.peak_type
            for peak in [oxidation_peak, reduction_peak]
            if peak is not None and peak_is_near_segment_endpoint(potential, peak)
        ]
        if endpoint_peaks:
            messages.append(
                f"Selected primary peak is near a scan endpoint for: {', '.join(endpoint_peaks)}. "
                "Endpoint-adjacent peaks can be switching artifacts."
            )

    low_confidence = [
        peak.peak_type
        for peak in [oxidation_peak, reduction_peak]
        if peak is not None and peak.confidence == "low"
    ]
    if low_confidence:
        messages.append(
            f"Selected primary peak needs review: low-confidence detection for {', '.join(low_confidence)}. "
            "Confirm the selected marker before using the metrics."
        )

    noise_sensitive = [
        peak.peak_type
        for peak in [oxidation_peak, reduction_peak]
        if peak is not None and peak.confidence == "medium"
    ]
    if noise_sensitive:
        messages.append(
            "Weak/noisy CV: selected Epa/Epc may be noise-sensitive. Try smoothing, baseline correction, "
            "or manual peak override before interpreting peak currents."
        )

    messages.extend(candidate_peak_ambiguity_messages(candidate_peaks, oxidation_peak, reduction_peak))

    if metrics is not None:
        delta_ep = metrics.get("delta_ep_V")
        if delta_ep is not None and (delta_ep <= 0 or delta_ep < 0.005 or delta_ep > 1.0):
            messages.append(
                f"Peak separation looks unusual: \u0394Ep is {delta_ep * 1000:.4g} mV. "
                "Confirm that the selected Epa/Epc pair belongs to the same redox couple."
            )

        current_ratio = metrics.get("ipa_ipc_ratio")
        if current_ratio is not None and (current_ratio < 0.2 or current_ratio > 5):
            messages.append(
                f"Peak current ratio looks unusual: |Ipa/Ipc| is {current_ratio:.4g}. "
                "Confirm peak selection, baseline correction, and data quality."
            )
    return unique_messages(messages)


def cv_analysis_status(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    metrics: dict[str, Optional[float]],
    potential: np.ndarray,
    cycle_assignment_uncertain: bool = False,
    candidate_peaks: Optional[list[Peak]] = None,
    cv_behavior: Optional[CVBehaviorResult] = None,
) -> str:
    if cv_behavior is not None:
        if cv_behavior.behavior == "no_reliable_peaks":
            return "Incomplete"
        if cv_behavior.behavior in {
            "irreversible_oxidation_only",
            "irreversible_reduction_only",
            "multiple_redox_couples",
            "noisy_ambiguous",
            "quasi_reversible",
        }:
            return "Review recommended"
    if oxidation_peak is None and reduction_peak is None:
        return "Failed"
    if oxidation_peak is None or reduction_peak is None:
        return "Incomplete"
    if cycle_assignment_uncertain:
        return "Review recommended"
    if cv_analysis_quality_messages(oxidation_peak, reduction_peak, metrics, potential, candidate_peaks):
        return "Review recommended"
    return "Passed"


def parser_confidence_from_details(details: dict[str, Any]) -> str:
    if details.get("potential_column") in {None, "Not detected"}:
        return "Low"
    if details.get("current_column") in {None, "Not detected"}:
        return "Low"
    cleanup_severity = str(details.get("parser_cleanup_severity") or "none")
    if cleanup_severity in {"review", "strong"}:
        return "Medium"
    warnings = str(details.get("warnings", "None")).strip()
    cleanup_notice = str(details.get("parser_cleanup_notice") or "").strip()
    warning_parts = [
        part.strip()
        for part in warnings.split(";")
        if part.strip() and part.strip() != "None"
    ]
    non_cleanup_warnings = [part for part in warning_parts if not cleanup_notice or part != cleanup_notice]
    if non_cleanup_warnings:
        return "Medium"
    return "High"


def overall_analysis_status(
    analysis_status: str,
    unit_confidence: str = "High",
    parser_confidence: str = "High",
    current_unit_warning: str = "",
) -> str:
    if analysis_status in {"Failed", "Incomplete"}:
        return analysis_status
    if unit_confidence == "Low" or parser_confidence == "Low" or current_unit_warning:
        return "Review needed"
    if analysis_status == "Review recommended" or parser_confidence == "Medium":
        return "Review needed"
    return "Passed"


def cv_analysis_quality_label(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    metrics: dict[str, Optional[float]],
    potential: np.ndarray,
    cycle_assignment_uncertain: bool = False,
    candidate_peaks: Optional[list[Peak]] = None,
    cv_behavior: Optional[CVBehaviorResult] = None,
) -> str:
    status = cv_analysis_status(
        oxidation_peak,
        reduction_peak,
        metrics,
        potential,
        cycle_assignment_uncertain,
        candidate_peaks,
        cv_behavior,
    )
    if status != "Passed":
        return status

    confidence_values = [
        peak.confidence for peak in [oxidation_peak, reduction_peak] if peak is not None
    ]
    if len(confidence_values) == 2 and all(value == "high" for value in confidence_values):
        return "High"
    if len(confidence_values) == 2 and all(value in {"high", "medium-high"} for value in confidence_values):
        return "Medium-high"
    if confidence_values:
        return ", ".join(confidence_values).title()
    return "Incomplete"


def peak_confidence_display(peak: Optional[Peak], rejected_peak: Optional[Peak] = None) -> str:
    if peak is not None:
        return format_confidence_label(peak.confidence)
    if rejected_peak is not None:
        return "Not detected / Low"
    return "Not detected"


def selected_peak_legend_label(cv_behavior: Optional[CVBehaviorResult]) -> str:
    if cv_behavior is None:
        return "Selected Epa/Epc"
    if cv_behavior.behavior == "irreversible_oxidation_only":
        return "Selected Epa"
    if cv_behavior.behavior == "irreversible_reduction_only":
        return "Selected Epc"
    if cv_behavior.behavior == "multiple_redox_couples":
        return "Selected redox couple"
    if cv_behavior.behavior == "no_reliable_peaks":
        return "No selected peak"
    return "Selected Epa/Epc"


def candidate_peak_explanation(cv_behavior: Optional[CVBehaviorResult]) -> str:
    prefix = "Candidate peaks are local extrema retained for transparency. "
    if cv_behavior is None:
        return prefix + "Only the selected Epa/Epc pair is used for CV metrics."
    if cv_behavior.behavior == "irreversible_oxidation_only":
        return prefix + "Only the selected Epa is used; reversible-pair metrics are not calculated."
    if cv_behavior.behavior == "irreversible_reduction_only":
        return prefix + "Only the selected Epc is used; reversible-pair metrics are not calculated."
    if cv_behavior.behavior == "multiple_redox_couples":
        return prefix + "Only the selected redox couple is used for the main CV metrics."
    if cv_behavior.behavior == "no_reliable_peaks":
        return prefix + "No primary peak is reliable enough for CV metrics."
    return prefix + "Only the selected Epa/Epc pair is used for CV metrics."


def candidate_rejection_reasons(cv_behavior: CVBehaviorResult) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if cv_behavior.rejected_reduction_peak is not None:
        reasons[cv_behavior.rejected_reduction_peak.id] = (
            "Rejected: insufficient cathodic return evidence."
        )
    if cv_behavior.rejected_oxidation_peak is not None:
        reasons[cv_behavior.rejected_oxidation_peak.id] = (
            "Rejected: insufficient anodic return evidence."
        )
    return reasons


def redox_couple_option_label(couple: RedoxCouple) -> str:
    return (
        f"{couple.id} — E\u00b0 {couple.formal_potential_v:.4g} V · "
        f"\u0394Ep {couple.delta_ep_v * 1000:.4g} mV · "
        f"|Ipa/Ipc| {couple.current_ratio:.4g}"
    )


def redox_couples_display_table(
    couples: list[RedoxCouple],
    selected_index: int,
) -> pd.DataFrame:
    rows = []
    for index, couple in enumerate(couples):
        rows.append(
            {
                "Couple": couple.id,
                "Epa": f"{couple.oxidation_peak.potential:.4g} V",
                "Epc": f"{couple.reduction_peak.potential:.4g} V",
                "\u0394Ep": f"{couple.delta_ep_v * 1000:.4g} mV",
                "E\u00b0": f"{couple.formal_potential_v:.4g} V",
                "|Ipa/Ipc|": f"{couple.current_ratio:.4g}",
                "Confidence": couple.confidence,
                "Selected?": "Selected" if index == selected_index else "",
            }
        )
    return pd.DataFrame(rows)


def metric_trend_summary(label: str, values: list[Optional[float]], unit: str) -> str:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if len(finite_values) < 2:
        return f"{label}: not enough detected values."

    start_value = finite_values[0]
    end_value = finite_values[-1]
    delta = end_value - start_value
    scale = max(abs(start_value), abs(end_value), 1e-12)
    if abs(delta) / scale < 0.05:
        trend = "roughly stable"
    elif label == "Ipc":
        trend = "became more cathodic" if end_value < start_value else "became less cathodic"
    else:
        trend = "increased" if delta > 0 else "decreased"

    return f"{label}: {trend} ({start_value:.4g} to {end_value:.4g} {unit})."


def build_per_cycle_cv_metrics(
    potential_v: np.ndarray,
    analysis_current_a: np.ndarray,
    cycle_values: Optional[np.ndarray],
    cycles: list[float],
    selected_cycle_value: Optional[float],
    min_prominence_a: float,
    min_distance: int,
    min_abs_current_a: float,
    display_unit: str,
    electrode_area_cm2: float,
    current_source: str,
    selected_cycle_metrics: Optional[dict[str, Optional[float]]] = None,
    selected_oxidation_peak: Optional[Peak] = None,
    selected_reduction_peak: Optional[Peak] = None,
    selected_cv_behavior: Optional[CVBehaviorResult] = None,
    auto_tune_each_cycle: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    trend_values: dict[str, list[Optional[float]]] = {
        "ipa": [],
        "ipc": [],
        "delta_ep": [],
    }
    display_label = display_unit_label(display_unit)

    for cycle_value in cycles:
        cycle_mask = cycle_selection_mask(cycle_values, cycle_value, len(potential_v))
        if not np.any(cycle_mask):
            continue

        cycle_potential = potential_v[cycle_mask]
        cycle_current = analysis_current_a[cycle_mask]
        is_selected_cycle = selected_cycle_value is not None and np.isclose(
            cycle_value,
            selected_cycle_value,
            rtol=0.0,
            atol=1e-9,
        )

        if is_selected_cycle and selected_cycle_metrics is not None:
            metrics = selected_cycle_metrics
            oxidation_peak = selected_oxidation_peak
            reduction_peak = selected_reduction_peak
            behavior = selected_cv_behavior
        else:
            cycle_min_prominence = min_prominence_a
            cycle_min_distance = int(min_distance)
            cycle_min_abs_current = min_abs_current_a
            if auto_tune_each_cycle:
                cycle_params = auto_tune_cv_peak_parameters(cycle_potential, cycle_current)
                cycle_min_prominence = float(cycle_params["min_prominence"])
                cycle_min_distance = int(cycle_params["min_distance"])
                cycle_min_abs_current = float(cycle_params["min_abs_current"])

            cycle_analysis = analyze_cv_default_peaks(
                cycle_potential,
                cycle_current,
                cycle_min_prominence,
                cycle_min_distance,
                cycle_min_abs_current,
                display_unit,
                electrode_area_cm2,
                current_source,
            )
            oxidation_peak = cycle_analysis.oxidation_peak
            reduction_peak = cycle_analysis.reduction_peak
            metrics = cycle_analysis.metrics
            behavior = cycle_analysis.behavior

        epa = metrics.get("epa_V")
        epc = metrics.get("epc_V")
        ipa = metrics.get("ipa_A")
        ipc = metrics.get("ipc_A")
        delta_ep = metrics.get("delta_ep_V")
        formal_potential = (epa + epc) / 2 if epa is not None and epc is not None else None
        pair_metric_review_reason = cv_pair_metrics_review_reason(
            metrics,
            behavior.pair_confidence if behavior is not None else None,
        )
        ipa_display = (
            current_to_display(np.array([ipa]), display_unit, current_source, electrode_area_cm2)[0]
            if ipa is not None
            else None
        )
        ipc_display = (
            current_to_display(np.array([ipc]), display_unit, current_source, electrode_area_cm2)[0]
            if ipc is not None
            else None
        )
        trend_values["ipa"].append(ipa_display)
        trend_values["ipc"].append(ipc_display)
        trend_values["delta_ep"].append(delta_ep * 1000 if delta_ep is not None else None)

        rows.append(
            {
                "cycle": cycle_display_label(cycle_value),
                "analyzed": "Yes" if is_selected_cycle else "",
                "Epa (V)": "Not detected" if epa is None else f"{epa:.4g}",
                f"Ipa ({display_label})": "Not detected" if ipa_display is None else f"{ipa_display:.4g}",
                "Epc (V)": "Not detected" if epc is None else f"{epc:.4g}",
                f"Ipc ({display_label})": "Not detected" if ipc_display is None else f"{ipc_display:.4g}",
                "\u0394Ep (mV)": (
                    "Not detected"
                    if delta_ep is None
                    else format_reviewed_pair_metric(f"{delta_ep * 1000:.4g}", pair_metric_review_reason)
                ),
                "E\u00b0\u2032 (V)": (
                    "Not detected"
                    if formal_potential is None
                    else format_reviewed_pair_metric(f"{formal_potential:.4g}", pair_metric_review_reason)
                ),
                "|Ipa/Ipc|": (
                    "Not detected"
                    if metrics.get("ipa_ipc_ratio") is None
                    else format_reviewed_pair_metric(f"{metrics['ipa_ipc_ratio']:.4g}", pair_metric_review_reason)
                ),
                "analysis quality": cv_analysis_quality_label(
                    oxidation_peak,
                    reduction_peak,
                    metrics,
                    cycle_potential,
                    cv_behavior=behavior,
                ),
            }
        )

    trend_messages = [
        metric_trend_summary("Ipa", trend_values["ipa"], display_label),
        metric_trend_summary("Ipc", trend_values["ipc"], display_label),
        metric_trend_summary("\u0394Ep", trend_values["delta_ep"], "mV"),
    ]
    return pd.DataFrame(rows), trend_messages


def status_badge_class(status: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", status.lower()).strip("-")


def summarize_scan_rate_for_card(scan_rate_estimate: str) -> Optional[str]:
    if not scan_rate_estimate or scan_rate_estimate == "Not available":
        return None
    match = re.search(r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*mV/s\)", scan_rate_estimate)
    if match:
        value = float(match.group(1))
        if abs(value) >= 10:
            formatted = f"{value:.0f}" if np.isclose(value, round(value), rtol=0, atol=0.05) else f"{value:.1f}"
        else:
            formatted = f"{value:.3g}"
        return f"{formatted} mV/s"
    return scan_rate_estimate


def scan_rate_source_from_method(method: str) -> str:
    normalized = str(method or "").lower()
    if "metadata" in normalized:
        return "metadata"
    if "manual" in normalized or "user" in normalized:
        return "manual"
    if "dE/dt".lower() in normalized or "time and potential" in normalized:
        return "inferred"
    return "not available"


def format_scan_rate_with_source(scan_rate_estimate: str, scan_rate_source: str = "") -> str:
    label = summarize_scan_rate_for_card(scan_rate_estimate)
    if not label:
        return "Not available"
    source = str(scan_rate_source or "").strip().lower()
    if source in {"inferred", "metadata", "manual"}:
        return f"{label} ({source})"
    return label


def render_cv_analysis_summary(
    metrics: dict[str, Optional[float]],
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    display_unit: str,
    electrode_area_cm2: float,
    current_source: str,
    potential: np.ndarray,
    scan_rate_estimate: str = "Not available",
    scan_rate_source: str = "",
    cycle_count: str = "Not detected",
    title_suffix: str = "",
    cycle_assignment_uncertain: bool = False,
    analyzed_cycle_label: Optional[str] = None,
    analysis_status_override: Optional[str] = None,
    analysis_quality_override: Optional[str] = None,
    unit_confidence: str = "High",
    parser_confidence: str = "High",
    current_unit_warning: str = "",
    candidate_peaks: Optional[list[Peak]] = None,
    cv_behavior: Optional[CVBehaviorResult] = None,
) -> None:
    epa = metrics.get("epa_V")
    epc = metrics.get("epc_V")
    formal_potential = (epa + epc) / 2 if epa is not None and epc is not None else None
    delta_ep_mv = metrics.get("delta_ep_V") * 1000 if metrics.get("delta_ep_V") is not None else None

    segments = split_scan_segments(potential)
    scan_mode = "Forward/reverse" if len(segments) >= 2 else "Single scan"
    switching_potential = None
    if segments:
        switch_idx = segments[0][1] - 1
        if 0 <= switch_idx < len(potential) and np.isfinite(potential[switch_idx]):
            switching_potential = f"{potential[switch_idx]:.4g} V"
    scan_rate_display = format_scan_rate_with_source(scan_rate_estimate, scan_rate_source)
    cycle_count_label = None
    if cycle_count and not str(cycle_count).startswith("Not detected"):
        cycle_count_label = str(cycle_count)

    confidence_values = [peak.confidence for peak in [oxidation_peak, reduction_peak] if peak is not None]
    if cv_behavior is not None:
        confidence_label = cv_behavior.pair_confidence
    elif oxidation_peak is None or reduction_peak is None:
        confidence_label = "Incomplete"
    elif len(confidence_values) == 2 and all(value == "high" for value in confidence_values):
        confidence_label = "High"
    elif any(value == "low" for value in confidence_values):
        confidence_label = "Low"
    elif any(value == "medium" for value in confidence_values):
        confidence_label = "Medium"
    elif len(confidence_values) == 2 and all(value in {"high", "medium-high"} for value in confidence_values):
        confidence_label = "Medium-high"
    elif confidence_values:
        confidence_label = ", ".join(confidence_values).title()
    else:
        confidence_label = "Incomplete"

    peak_status = cv_analysis_status(
        oxidation_peak,
        reduction_peak,
        metrics,
        potential,
        cycle_assignment_uncertain,
        candidate_peaks,
        cv_behavior,
    )
    status = analysis_status_override or overall_analysis_status(
        peak_status,
        unit_confidence,
        parser_confidence,
        current_unit_warning,
    )
    status_class = status_badge_class(status)
    current_metrics_pending = pending_current_metric(unit_confidence)
    current_metrics_need_review = current_metrics_pending or bool(current_unit_warning)
    ipa_value = (
        "Pending unit confirmation"
        if current_metrics_pending
        else format_current_metric(
            metrics.get("ipa_A"),
            display_unit,
            current_source,
            electrode_area_cm2,
            current_metrics_need_review,
        )
    )
    ipc_value = (
        "Pending unit confirmation"
        if current_metrics_pending
        else format_current_metric(
            metrics.get("ipc_A"),
            display_unit,
            current_source,
            electrode_area_cm2,
            current_metrics_need_review,
        )
    )
    ratio_value = (
        "Pending unit confirmation"
        if current_metrics_pending
        else (
            "Not detected"
            if metrics.get("ipa_ipc_ratio") is None
            else f"{metrics['ipa_ipc_ratio']:.4g}{' (unit review)' if current_metrics_need_review else ''}"
        )
    )
    epa_value = format_potential_metric(epa)
    epc_value = format_potential_metric(epc)
    delta_ep_value = "Not detected" if delta_ep_mv is None else f"{delta_ep_mv:.4g} mV"
    formal_potential_value = format_potential_metric(formal_potential)
    pair_metric_review_reason = cv_pair_metrics_review_reason(metrics, confidence_label)
    if pair_metric_review_reason:
        delta_ep_value = format_reviewed_pair_metric(delta_ep_value, pair_metric_review_reason)
        formal_potential_value = format_reviewed_pair_metric(formal_potential_value, pair_metric_review_reason)
        if not current_metrics_pending and metrics.get("ipa_ipc_ratio") is not None:
            ratio_value = format_reviewed_pair_metric(ratio_value, pair_metric_review_reason)
    if cv_behavior is not None:
        if cv_behavior.behavior == "irreversible_oxidation_only":
            epc_value = "No reliable cathodic peak"
            ipc_value = "Not detected"
            delta_ep_value = "Not applicable"
            formal_potential_value = "Not applicable"
            ratio_value = "Not meaningful"
        elif cv_behavior.behavior == "irreversible_reduction_only":
            epa_value = "No reliable anodic peak"
            ipa_value = "Not detected"
            delta_ep_value = "Not applicable"
            formal_potential_value = "Not applicable"
            ratio_value = "Not meaningful"

    title = f"Analysis overview{title_suffix}"
    behavior_label = cv_behavior.label if cv_behavior is not None else "Reversible-like"
    selected_couple_label = None
    if cv_behavior is not None and cv_behavior.behavior == "multiple_redox_couples" and cv_behavior.redox_couples:
        selected_couple_label = (cv_behavior.redox_couples or [])[cv_behavior.selected_couple_index or 0].id

    diagnostic_items = [
        ("Analysis status", status),
        ("Unit confidence", unit_confidence),
        ("Parser confidence", parser_confidence),
    ]
    if cv_behavior is not None and cv_behavior.behavior == "irreversible_oxidation_only":
        diagnostic_items.extend(
            [
                ("Oxidation peak confidence", peak_confidence_display(cv_behavior.oxidation_peak, cv_behavior.rejected_oxidation_peak)),
                ("Cathodic return peak confidence", peak_confidence_display(cv_behavior.reduction_peak, cv_behavior.rejected_reduction_peak)),
                ("Reversible-pair confidence", cv_behavior.pair_confidence),
            ]
        )
    elif cv_behavior is not None and cv_behavior.behavior == "irreversible_reduction_only":
        diagnostic_items.extend(
            [
                ("Reduction peak confidence", peak_confidence_display(cv_behavior.reduction_peak, cv_behavior.rejected_reduction_peak)),
                ("Anodic return peak confidence", peak_confidence_display(cv_behavior.oxidation_peak, cv_behavior.rejected_oxidation_peak)),
                ("Reversible-pair confidence", cv_behavior.pair_confidence),
            ]
        )
    else:
        diagnostic_items.append(("Peak detection confidence", confidence_label))
        if cv_behavior is not None:
            diagnostic_items.extend(
                [
                    ("Oxidation peak confidence", peak_confidence_display(cv_behavior.oxidation_peak, cv_behavior.rejected_oxidation_peak)),
                    ("Reduction peak confidence", peak_confidence_display(cv_behavior.reduction_peak, cv_behavior.rejected_reduction_peak)),
                    ("Pair confidence", cv_behavior.pair_confidence),
                ]
            )
    diagnostic_items.extend(
        [
            ("Overall quality", analysis_quality_override or status),
            ("Scan mode", scan_mode),
        ]
    )
    if switching_potential:
        diagnostic_items.append(("Switching potential", switching_potential))
    if analyzed_cycle_label:
        diagnostic_items.append(("Analyzed cycle", analyzed_cycle_label))
    elif cycle_count_label:
        diagnostic_items.append(("Cycle count", cycle_count_label))
    diagnostic_items.append(("Scan rate", scan_rate_display))
    if selected_couple_label:
        diagnostic_items.append(("Selected redox couple", selected_couple_label))

    summary_items = [
        ("CV behavior", behavior_label),
        ("Scan rate", scan_rate_display),
        ("Epa", epa_value),
        ("Ipa", ipa_value),
        ("Epc", epc_value),
        ("Ipc", ipc_value),
        ("\u0394Ep", delta_ep_value),
        ("E\u00b0\u2032", formal_potential_value),
        ("|Ipa/Ipc|", ratio_value),
    ]

    metric_html = "\n".join(
        f"""
        <div class="analysis-metric">
            <span>{escape_html(label)}</span>
            <strong>{escape_html(value)}</strong>
        </div>
        """
        for label, value in summary_items
    )
    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-card-header">
                <h4>{escape_html(title)}</h4>
                <span class="status-badge status-{escape_html(status_class)}">{escape_html(status)}</span>
            </div>
            <div class="analysis-grid">{metric_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if pair_metric_review_reason and not (
        cv_behavior is not None
        and cv_behavior.behavior in {"irreversible_oxidation_only", "irreversible_reduction_only"}
    ):
        if pair_metric_review_reason.startswith("Invalid"):
            st.warning(pair_metric_review_reason.removeprefix("Invalid: ").strip())
        else:
            st.caption(pair_metric_review_reason)
    with st.expander("Diagnostics", expanded=False):
        render_readable_table(
            pd.DataFrame(
                {
                    "Field": [label for label, _value in diagnostic_items],
                    "Value": [value for _label, value in diagnostic_items],
                }
            )
        )
    if current_metrics_pending:
        st.caption("Current-dependent metrics will be calculated after unit confirmation.")


def lsv_scan_direction(potential_v: np.ndarray) -> str:
    finite = potential_v[np.isfinite(potential_v)]
    if finite.size < 2:
        return "anodic"
    return "anodic" if float(finite[-1]) >= float(finite[0]) else "cathodic"


def select_lsv_onset(result: ESWResult, direction: str) -> tuple[Optional[float], Optional[str]]:
    if direction == "cathodic" and result.cathodic_limit is not None:
        return result.cathodic_limit, "cathodic"
    if direction == "anodic" and result.anodic_limit is not None:
        return result.anodic_limit, "anodic"
    if result.anodic_limit is not None:
        return result.anodic_limit, "anodic"
    if result.cathodic_limit is not None:
        return result.cathodic_limit, "cathodic"
    return None, None


def default_lsv_onset_smoothing_window(point_count: int) -> int:
    if point_count < 7:
        return 0
    window = min(15, max(5, int(round(point_count * 0.015))))
    if window % 2 == 0:
        window += 1
    return min(window, point_count if point_count % 2 == 1 else point_count - 1)


def interpolate_threshold_crossing(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    threshold: float,
) -> float:
    if y2 == y1:
        return float(x2)
    return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))


def analyze_lsv_onset(
    potential_v: np.ndarray,
    current_values: np.ndarray,
    threshold_a: Optional[float],
    direction: Optional[str] = None,
    min_consecutive_points: int = 5,
    min_potential_window_v: float = 0.05,
    edge_fraction: float = 0.03,
    use_smoothing: bool = True,
    smoothing_window: Optional[int] = None,
) -> LSVOnsetResult:
    direction = direction or lsv_dominant_response_direction(current_values) or lsv_scan_direction(potential_v)
    method = "Sustained threshold crossing"
    threshold_a = float(threshold_a) if threshold_a is not None else None
    signed_threshold = None if threshold_a is None else (threshold_a if direction == "anodic" else -threshold_a)
    point_count = int(min(len(potential_v), len(current_values)))
    smoothing_window = (
        default_lsv_onset_smoothing_window(point_count)
        if smoothing_window is None
        else int(smoothing_window)
    )

    base_result = LSVOnsetResult(
        None,
        direction,
        "Review needed",
        "Not detected",
        method,
        "No reliable sustained threshold crossing was detected.",
        "No reliable sustained threshold crossing was detected.",
        threshold_a,
        signed_threshold,
        int(min_consecutive_points),
        float(min_potential_window_v),
        "Moving average" if use_smoothing and smoothing_window >= 3 else "None",
        int(max(0, smoothing_window)),
    )

    if threshold_a is None or threshold_a <= 0 or point_count < 3:
        return base_result

    potential = np.asarray(potential_v[:point_count], dtype=float)
    raw_current = np.asarray(current_values[:point_count], dtype=float)
    finite = np.isfinite(potential) & np.isfinite(raw_current)
    if np.sum(finite) < 3:
        return base_result

    if point_count < max(8, min_consecutive_points * 2):
        sparse_result = calculate_esw(potential, raw_current, threshold_a)
        sparse_onset, sparse_direction = select_lsv_onset(sparse_result, direction)
        if sparse_onset is not None and sparse_direction is not None:
            signed_sparse_threshold = threshold_a if sparse_direction == "anodic" else -threshold_a
            return LSVOnsetResult(
                sparse_onset,
                sparse_direction,
                "Passed",
                "Limited by sparse data",
                method,
                (
                    "Onset estimated by linear threshold interpolation. "
                    "Sustained-crossing checks are limited because the trace has few points."
                ),
                "",
                threshold_a,
                signed_sparse_threshold,
                int(min_consecutive_points),
                float(min_potential_window_v),
                "None",
                0,
            )
        return base_result

    detection_current = raw_current.copy()
    if use_smoothing and smoothing_window >= 3 and point_count >= smoothing_window:
        detection_current = moving_average(raw_current, smoothing_window)

    comparison = detection_current >= signed_threshold if direction == "anodic" else detection_current <= signed_threshold
    comparison &= np.isfinite(detection_current) & np.isfinite(potential)
    edge_points = max(2, int(point_count * edge_fraction))
    edge_points = min(edge_points, max(2, point_count // 4))
    noise_floor = max(robust_current_noise(raw_current), robust_current_noise(detection_current), 1e-15)
    response_magnitude = (
        max(0.0, float(np.nanmax(detection_current[np.isfinite(detection_current)])))
        if direction == "anodic"
        else max(0.0, abs(float(np.nanmin(detection_current[np.isfinite(detection_current)]))))
    )
    if response_magnitude < noise_floor * 8:
        base_result.rejected_reason = "Signal is not sufficiently above local noise for a reliable onset."
        base_result.warning = "No reliable sustained threshold crossing was detected."
        return base_result

    first_rejected_potential = None
    first_rejected_reason = ""
    indices = np.flatnonzero(comparison)
    for idx in indices:
        if idx > 0 and comparison[idx - 1]:
            continue

        if idx < edge_points:
            first_rejected_potential = float(potential[idx])
            first_rejected_reason = "Rejected crossing because it occurs too close to the scan start."
            continue
        if idx >= point_count - edge_points:
            first_rejected_potential = float(potential[idx])
            first_rejected_reason = "Rejected crossing because it occurs too close to the scan end."
            continue

        run_end = idx
        while run_end < point_count and bool(comparison[run_end]):
            run_end += 1
        run_length = run_end - idx
        run_span = abs(float(potential[run_end - 1] - potential[idx])) if run_length > 0 else 0.0
        if run_length < min_consecutive_points or run_span < min_potential_window_v:
            first_rejected_potential = float(potential[idx])
            first_rejected_reason = (
                "Rejected crossing because the signal drops back below threshold too quickly "
                "or spans too narrow a potential window."
            )
            continue

        window_end = min(run_end, idx + max(min_consecutive_points, 5))
        local_deltas = np.diff(detection_current[idx:window_end])
        local_deltas = local_deltas[np.isfinite(local_deltas)]
        if local_deltas.size:
            median_delta = float(np.nanmedian(local_deltas))
            if direction == "anodic" and median_delta < -noise_floor:
                first_rejected_potential = float(potential[idx])
                first_rejected_reason = "Rejected crossing because current does not grow after crossing."
                continue
            if direction == "cathodic" and median_delta > noise_floor:
                first_rejected_potential = float(potential[idx])
                first_rejected_reason = "Rejected crossing because cathodic current magnitude does not grow after crossing."
                continue

        if direction == "anodic":
            post_growth = float(np.nanmax(detection_current[idx:]) - signed_threshold)
        else:
            post_growth = float(signed_threshold - np.nanmin(detection_current[idx:]))
        if post_growth < max(abs(signed_threshold) * 0.35, noise_floor * 3):
            first_rejected_potential = float(potential[idx])
            first_rejected_reason = "Rejected crossing because it is not followed by sustained current growth."
            continue

        previous_idx = idx - 1
        onset_potential = (
            float(potential[idx])
            if previous_idx < 0
            else interpolate_threshold_crossing(
                float(potential[previous_idx]),
                float(detection_current[previous_idx]),
                float(potential[idx]),
                float(detection_current[idx]),
                float(signed_threshold),
            )
        )
        smoothing_note = (
            f" Onset detected from smoothed signal ({smoothing_window}-point moving average); raw trace shown."
            if use_smoothing and smoothing_window >= 3
            else ""
        )
        return LSVOnsetResult(
            onset_potential,
            direction,
            "Passed",
            "Reliable",
            method,
            (
                f"Onset = linearly interpolated potential where {direction} signal first crosses the threshold "
                f"and remains past it for at least {min_consecutive_points} points / "
                f"{min_potential_window_v * 1000:.0f} mV.{smoothing_note}"
            ),
            "",
            threshold_a,
            signed_threshold,
            int(min_consecutive_points),
            float(min_potential_window_v),
            "Moving average" if use_smoothing and smoothing_window >= 3 else "None",
            int(max(0, smoothing_window)),
            first_rejected_potential,
            first_rejected_reason,
        )

    if first_rejected_potential is not None:
        base_result.rejected_potential = first_rejected_potential
        base_result.rejected_reason = first_rejected_reason
        if "too close" in first_rejected_reason:
            base_result.warning = "No reliable sustained threshold crossing was detected."
        else:
            base_result.warning = "Possible false onset from noise spike; review smoothing/threshold settings."
        base_result.method_note = base_result.warning
    return base_result


def readable_lsv_current_unit(
    current_values: np.ndarray,
    current_source: str,
    electrode_area_cm2: float,
    prefer_density: bool = True,
) -> str:
    density_units = ["nA/cm^2", "uA/cm^2", "mA/cm^2", "A/cm^2"]
    current_units = ["nA", "uA", "mA", "A"]
    candidate_units = density_units if prefer_density else current_units

    for unit in candidate_units:
        display_values = current_to_display(current_values, unit, current_source, electrode_area_cm2)
        finite_abs = np.abs(display_values[np.isfinite(display_values)])
        if not finite_abs.size:
            continue
        p95 = float(np.nanpercentile(finite_abs, 95))
        if 0.1 <= p95 < 1000:
            return unit

    return "uA/cm^2" if prefer_density else "uA"


def lsv_display_unit_for_records(
    records: list[dict[str, Any]],
    current_source: str,
    electrode_area_cm2: float,
    prefer_density: bool = True,
) -> str:
    current_arrays = [
        np.asarray(record.get("data", {}).get("current", np.array([])), dtype=float)
        for record in records
        if record.get("data", {}).get("current") is not None
    ]
    if not current_arrays:
        return "uA/cm^2" if prefer_density else "uA"
    return readable_lsv_current_unit(
        np.concatenate(current_arrays),
        current_source,
        electrode_area_cm2,
        prefer_density=prefer_density,
    )


def format_lsv_current_value(
    current_value: Optional[float],
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float,
) -> str:
    if current_value is None:
        return "Not available"
    display_value = current_to_display(
        np.array([current_value], dtype=float),
        display_unit,
        current_source,
        electrode_area_cm2,
    )[0]
    if not np.isfinite(display_value):
        return "Not available"
    return f"{display_value:.4g} {display_unit_label(display_unit)}"


def format_lsv_threshold(
    threshold_base: Optional[float],
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float,
) -> str:
    if threshold_base is None:
        return "Not available"
    return format_lsv_current_value(threshold_base, display_unit, current_source, electrode_area_cm2)


def lsv_threshold_magnitude_and_crossing(
    threshold_base: Optional[float],
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float,
    direction: Optional[str],
) -> tuple[str, str]:
    if threshold_base is None:
        return "Not available", "Not available"
    magnitude_base = abs(float(threshold_base))
    signed_base = -magnitude_base if direction == "cathodic" else magnitude_base
    return (
        format_lsv_threshold(magnitude_base, display_unit, current_source, electrode_area_cm2),
        format_lsv_threshold(signed_base, display_unit, current_source, electrode_area_cm2),
    )


def lsv_crossing_threshold_label(direction: Optional[str]) -> str:
    if direction == "cathodic":
        return "Cathodic crossing threshold"
    if direction == "anodic":
        return "Anodic crossing threshold"
    return "Crossing threshold"


def lsv_threshold_plot_labels(
    threshold_display: float,
    display_unit: str,
    direction: Optional[str],
    unit_review_required: bool = False,
) -> tuple[str, str]:
    unit_label = "unit unconfirmed" if unit_review_required else display_unit_label(display_unit)
    value = f"{abs(float(threshold_display)):.4g} {unit_label}"
    if unit_review_required:
        if direction == "cathodic":
            return f"Magnitude: {value}", f"Threshold: -{value}"
        if direction == "anodic":
            return f"Threshold: {value}", f"Magnitude: {value}"
        return f"Threshold: {value}", f"Threshold: -{value}"
    if direction == "cathodic":
        return f"Magnitude: {value}", f"Threshold: -{value}"
    if direction == "anodic":
        return f"Threshold: +{value}", f"Magnitude: {value}"
    return f"Threshold: +{value}", f"Threshold: -{value}"


def lsv_unconfirmed_unit_label(current_source: str) -> str:
    return "Current density (unit unconfirmed)" if current_source == "Current density" else "Signal (unit unconfirmed)"


def lsv_threshold_quantity_label(display_unit: str) -> str:
    return "current density" if is_current_density_unit(display_unit) else "current"


def lsv_axis_options_for_current_source(current_source: str) -> tuple[list[str], str]:
    if current_source == "Current density":
        return ["Potential", "Current density"], "Current density"
    return ["Potential", "Current", "Current density"], "Current"


def lsv_dominant_response_direction(current_values: np.ndarray) -> Optional[str]:
    finite = np.asarray(current_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    max_current = float(np.nanmax(finite))
    min_current = float(np.nanmin(finite))
    if abs(min_current) > abs(max_current):
        return "cathodic"
    if abs(max_current) > 0:
        return "anodic"
    return None


def lsv_analysis_direction(potential_v: np.ndarray, current_values: np.ndarray) -> str:
    return lsv_dominant_response_direction(current_values) or lsv_scan_direction(potential_v)


def lsv_auto_threshold_preset_label(display_unit: str) -> str:
    return f"Auto (10% of max {lsv_threshold_quantity_label(display_unit)})"


def lsv_threshold_preset_options(display_unit: str) -> list[str]:
    if is_current_density_unit(display_unit):
        return [
            lsv_auto_threshold_preset_label(display_unit),
            "1 mA/cm\u00b2",
            "5 mA/cm\u00b2",
            "10 mA/cm\u00b2",
            "20 mA/cm\u00b2",
            "50 mA/cm\u00b2",
            "Custom",
        ]
    return [
        lsv_auto_threshold_preset_label(display_unit),
        "1 \u00b5A",
        "5 \u00b5A",
        "10 \u00b5A",
        "50 \u00b5A",
        "100 \u00b5A",
        "Custom",
    ]


def lsv_threshold_preset_base_value(
    preset: str,
    current_source: str,
    electrode_area_cm2: float,
) -> Optional[float]:
    density_presets = {
        "1 mA/cm\u00b2": 1e-3,
        "5 mA/cm\u00b2": 5e-3,
        "10 mA/cm\u00b2": 10e-3,
        "20 mA/cm\u00b2": 20e-3,
        "50 mA/cm\u00b2": 50e-3,
    }
    current_presets = {
        "1 \u00b5A": 1e-6,
        "5 \u00b5A": 5e-6,
        "10 \u00b5A": 10e-6,
        "50 \u00b5A": 50e-6,
        "100 \u00b5A": 100e-6,
    }
    if preset in density_presets:
        density_base = density_presets[preset]
        return density_base if current_source == "Current density" else density_base * electrode_area_cm2
    return current_presets.get(preset)


def lsv_threshold_source_details(
    source: str,
    quantity_label: str,
    direction: Optional[str] = None,
) -> tuple[str, str]:
    if source == "Auto-selected":
        if direction == "anodic":
            return source, f"Auto-selected as 10% of maximum anodic {quantity_label}."
        if direction == "cathodic":
            return source, f"Auto-selected as 10% of maximum cathodic {quantity_label} magnitude."
        return source, f"Auto-selected as 10% of maximum absolute {quantity_label} magnitude."
    if source == "Default":
        return source, "Default threshold."
    return "User-defined", "User-defined threshold."


def threshold_values_match(value: Optional[float], expected: Optional[float]) -> bool:
    if value is None or expected is None:
        return False
    tolerance = max(abs(float(expected)) * 1e-9, 1e-12)
    return abs(float(value) - float(expected)) <= tolerance


def lsv_threshold_recommendation(
    preset: str,
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float,
    auto_threshold_display: float,
    auto_source: str,
    direction: Optional[str] = None,
) -> tuple[float, str, str]:
    quantity_label = lsv_threshold_quantity_label(display_unit)
    auto_label = lsv_auto_threshold_preset_label(display_unit)
    if preset == auto_label:
        source, rule = lsv_threshold_source_details(auto_source, quantity_label, direction)
        return auto_threshold_display, source, rule

    preset_base = lsv_threshold_preset_base_value(preset, current_source, electrode_area_cm2)
    if preset_base is not None:
        preset_display = current_to_display(
            np.array([preset_base], dtype=float),
            display_unit,
            current_source,
            electrode_area_cm2,
        )[0]
        source, rule = lsv_threshold_source_details("User-defined", quantity_label, direction)
        return float(preset_display), source, rule

    source, rule = lsv_threshold_source_details("User-defined", quantity_label, direction)
    return auto_threshold_display, source, rule


def format_lsv_potential(
    potential_v: Optional[float],
    reference_electrode: Optional[str] = None,
    decimals: int = 3,
) -> str:
    if potential_v is None or not np.isfinite(potential_v):
        return "Not detected"
    reference = f" vs {reference_electrode}" if reference_electrode else ""
    return f"{potential_v:.{decimals}f} V{reference}"


def build_lsv_analysis_summary(
    record: dict[str, Any],
    parsed_dataset: Optional[ParsedDataset],
    threshold_a: Optional[float],
    display_units: dict[str, Any],
    reference_electrode: Optional[str] = None,
    threshold_source: str = "User-defined",
    threshold_rule: str = "User-defined threshold.",
    reference_electrode_source: str = "user",
    unit_review_needed: bool = False,
) -> dict[str, str]:
    potential_v = np.asarray(record["data"].get("potential", np.array([])), dtype=float)
    current_values = np.asarray(record["data"].get("current", np.array([])), dtype=float)
    current_source = display_units.get("current_source", "Current")
    electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
    display_axes = {
        display_units.get("x_axis", "Potential"),
        display_units.get("y_axis", "Current density"),
    }
    prefer_density = "Current density" in display_axes
    current_unit = readable_lsv_current_unit(
        current_values,
        current_source,
        electrode_area_cm2,
        prefer_density=prefer_density,
    )
    current_label = "current density" if is_current_density_unit(current_unit) else "current"

    direction = lsv_analysis_direction(potential_v, current_values)
    onset_result = analyze_lsv_onset(potential_v, current_values, threshold_a, direction)
    onset_potential = onset_result.onset_potential
    onset_direction = onset_result.onset_direction or direction

    status = onset_result.status
    review_note = onset_result.warning
    if onset_potential is None:
        method_note = review_note or "No reliable sustained threshold crossing was detected."
    else:
        threshold_note = format_lsv_threshold(
            onset_result.signed_threshold_base,
            current_unit,
            current_source,
            electrode_area_cm2,
        )
        reference_note = f" Reported vs {reference_electrode}." if reference_electrode else ""
        smoothing_note = (
            f" Onset detected from smoothed signal ({onset_result.smoothing_window}-point moving average); raw trace shown."
            if onset_result.smoothing_method != "None"
            else ""
        )
        method_note = (
            f"Onset was estimated using the selected threshold rule: linearly interpolated potential where "
            f"{onset_direction} {current_label} first crosses {threshold_note} and remains past the threshold "
            f"for at least {onset_result.sustained_points_required} points / "
            f"{onset_result.sustained_window_v * 1000:.0f} mV.{reference_note}{smoothing_note}"
        )
    if unit_review_needed:
        status = "Review needed"
        review_note = "Current-dependent metrics may be incorrectly scaled until current unit is confirmed."
        threshold_text = (
            f"{format_summary_number(abs(threshold_a))} unit unconfirmed"
            if threshold_a is not None
            else "unit unconfirmed"
        )
        reference_note = f" Reported vs {reference_electrode}." if reference_electrode else ""
        method_note = (
            f"Onset/current-dependent metrics are provisional until the {current_label} unit is confirmed. "
            f"Threshold = {threshold_text}.{reference_note}"
        )

    finite_current = current_values[np.isfinite(current_values)]
    finite_potential = potential_v[np.isfinite(potential_v)]
    max_current = None
    max_potential = None
    min_current = None
    min_potential = None
    if finite_current.size and potential_v.size == current_values.size:
        max_idx = int(np.nanargmax(current_values))
        min_idx = int(np.nanargmin(current_values))
        max_current = float(current_values[max_idx])
        max_potential = float(potential_v[max_idx]) if np.isfinite(potential_v[max_idx]) else None
        min_current = float(current_values[min_idx])
        min_potential = float(potential_v[min_idx]) if np.isfinite(potential_v[min_idx]) else None

    scan_rate = "Not available"
    scan_rate_note = ""
    electrode_area_metadata = "Not detected"
    if parsed_dataset is not None:
        potential_col = record.get("columns", {}).get("potential")
        scan_rate_estimate = scan_rate_estimate_summary(parsed_dataset, potential_col)
        scan_rate = summarize_scan_rate_for_card(scan_rate_estimate) or "Not available"
        if scan_rate == "Not available":
            scan_rate_note = scan_rate_unavailable_reason(parsed_dataset, potential_col)
        electrode_area_metadata = electrode_area_metadata_summary(parsed_dataset.metadata) or "Not detected"

    method_quantity = "Current-density" if is_current_density_unit(current_unit) else "Current"
    summary_direction = onset_direction or direction
    if summary_direction == "cathodic":
        onset_label = "Cathodic onset potential"
        primary_current_label = f"Max cathodic {current_label}"
        primary_current = min_current
        primary_potential_label = f"Potential at max cathodic {current_label}"
        primary_potential = min_potential
    elif summary_direction == "anodic":
        onset_label = "Anodic onset potential"
        primary_current_label = f"Max anodic {current_label}"
        primary_current = max_current
        primary_potential_label = f"Potential at max anodic {current_label}"
        primary_potential = max_potential
    else:
        onset_label = "Onset potential"
        primary_current_label = f"Max {current_label}"
        primary_current = max_current
        primary_potential_label = f"Potential at max {current_label}"
        primary_potential = max_potential
    threshold_magnitude, crossing_threshold = lsv_threshold_magnitude_and_crossing(
        threshold_a,
        current_unit,
        current_source,
        electrode_area_cm2,
        summary_direction,
    )
    if unit_review_needed:
        provisional_unit_label = lsv_unconfirmed_unit_label(current_source)
        threshold_magnitude = f"{format_summary_number(abs(threshold_a)) if threshold_a is not None else 'Not available'} unit unconfirmed"
        crossing_threshold = (
            f"-{format_summary_number(abs(threshold_a))} unit unconfirmed"
            if threshold_a is not None and summary_direction == "cathodic"
            else threshold_magnitude
        )
        primary_current_value = "Pending unit confirmation"
        max_value = "Pending unit confirmation"
        min_value = "Pending unit confirmation"
        onset_method = f"{provisional_unit_label} threshold crossing"
        method_quantity = "Current-density" if current_source == "Current density" else "Signal"
    else:
        primary_current_value = format_lsv_current_value(primary_current, current_unit, current_source, electrode_area_cm2)
        max_value = format_lsv_current_value(max_current, current_unit, current_source, electrode_area_cm2)
        min_value = format_lsv_current_value(min_current, current_unit, current_source, electrode_area_cm2)
        onset_method = f"{method_quantity} threshold crossing"

    normalization_note = ""
    if current_source == "Current density":
        normalization_note = "Uploaded signal was treated as current density; electrode-area normalization was skipped."
        if electrode_area_metadata != "Not detected":
            normalization_note += f" Electrode area metadata: {electrode_area_metadata}. No additional normalization applied."
    elif prefer_density:
        area_text = f"{electrode_area_cm2:.1f}" if electrode_area_cm2 == round(electrode_area_cm2) else f"{electrode_area_cm2:g}"
        if electrode_area_metadata != "Not detected":
            normalization_note = (
                f"Electrode area detected. Current was normalized by {area_text} cm² and displayed as "
                "current density."
            )
        else:
            normalization_note = f"Current density was calculated from raw current using electrode area = {area_text} cm²."
    elif electrode_area_metadata != "Not detected":
        normalization_note = (
            "Electrode area detected. Current-density display is available, but current is currently selected. "
            f"Detected area: {electrode_area_metadata}."
        )
    normalization_status = lsv_normalization_status_line(
        current_source,
        "Current density" if prefer_density else "Current",
        electrode_area_cm2,
        electrode_area_metadata,
    )
    metadata_completeness, missing_metadata = metadata_completeness_for_lsv(
        scan_rate=scan_rate,
        reference_electrode_source=reference_electrode_source,
        current_source=current_source,
        display_units=display_units,
        electrode_area_metadata=electrode_area_metadata,
    )
    if scan_rate == "Not available":
        scan_rate_source = "not available"
    elif parsed_dataset is not None and scan_rate_from_metadata(parsed_dataset.metadata) is not None:
        scan_rate_source = "metadata"
    else:
        scan_rate_source = "estimated from time/potential data"

    return {
        "status": status,
        "review_note": review_note,
        "metadata_completeness": metadata_completeness,
        "missing_metadata": missing_metadata,
        "onset_label": onset_label,
        "onset_potential": format_lsv_potential(onset_potential, reference_electrode),
        "onset_method": onset_method,
        "threshold": threshold_magnitude,
        "threshold_magnitude": threshold_magnitude,
        "crossing_threshold": crossing_threshold,
        "crossing_threshold_label": lsv_crossing_threshold_label(summary_direction),
        "threshold_source": threshold_source,
        "threshold_rule": threshold_rule,
        "unit_status": "Unconfirmed" if unit_review_needed else "Confirmed",
        "suggested_unit": display_unit_label(str(display_units.get("suggested_current_unit") or "")),
        "method_note": method_note,
        "onset_confidence": onset_result.confidence,
        "sustained_requirement": (
            f"{onset_result.sustained_points_required} points / "
            f"{onset_result.sustained_window_v * 1000:.0f} mV"
        ),
        "onset_detection_signal": (
            f"{onset_result.smoothing_method}, {onset_result.smoothing_window} points"
            if onset_result.smoothing_method != "None"
            else "Raw signal"
        ),
        "rejected_crossing_reason": onset_result.rejected_reason,
        "direction": summary_direction.title(),
        "primary_current_label": primary_current_label,
        "primary_current_value": primary_current_value,
        "primary_potential_label": primary_potential_label,
        "primary_potential": format_lsv_potential(primary_potential, reference_electrode),
        "max_label": f"Max {current_label}",
        "max_value": max_value,
        "max_potential": format_lsv_potential(max_potential, reference_electrode),
        "min_label": f"Min {current_label}",
        "min_value": min_value,
        "min_potential": format_lsv_potential(min_potential, reference_electrode),
        "potential_range": format_fixed_range(finite_potential, "V", 3),
        "scan_rate": scan_rate,
        "scan_rate_note": scan_rate_note,
        "electrode_area_metadata": electrode_area_metadata,
        "normalization_status": normalization_status,
        "normalization_note": normalization_note,
        "reference_electrode_source": reference_electrode_source,
        "scan_rate_source": scan_rate_source,
        "reference_electrode": reference_electrode_source_label(reference_electrode, reference_electrode_source),
        "reference_note": reference_electrode_default_note(reference_electrode, reference_electrode_source),
    }


def render_lsv_analysis_summary(summary: dict[str, str]) -> None:
    status = summary.get("status", "Review needed")
    status_class = status_badge_class(status)
    direction = str(summary.get("direction", "")).lower()
    threshold_items = [
        (
            "Threshold magnitude" if direction == "cathodic" else "Threshold",
            summary.get("threshold_magnitude", summary.get("threshold", "Not available")),
        )
    ]
    if direction == "cathodic":
        threshold_items.append(
            (
                summary.get("crossing_threshold_label", "Cathodic crossing threshold"),
                summary.get("crossing_threshold", "Not available"),
            )
        )
    summary_items = [
        (summary.get("onset_label", "Onset potential"), summary.get("onset_potential", "Not detected")),
        ("Direction", summary.get("direction", "Not available")),
        *threshold_items[:1],
        (summary.get("primary_current_label", "Max current"), summary.get("primary_current_value", "Not available")),
        (summary.get("primary_potential_label", "Potential at max current"), summary.get("primary_potential", "Not available")),
        ("Reference electrode", summary.get("reference_electrode", "Not selected")),
        ("Metadata completeness", summary.get("metadata_completeness", "Not available")),
    ]
    summary_items.append(("Scan rate", str(summary.get("scan_rate", "Not available"))))
    if summary.get("electrode_area_metadata") not in {None, "", "Not detected"}:
        summary_items.append(("Electrode area", str(summary["electrode_area_metadata"])))
    summary_items.extend(
        [
            ("Analysis status", status),
            ("Onset reliability", summary.get("onset_confidence", "Not available")),
        ]
    )
    normalization_note = str(summary.get("normalization_note") or "").strip()
    normalization_status = str(summary.get("normalization_status") or "").strip()
    diagnostic_items = [
        ("Unit status", summary.get("unit_status", "Confirmed")),
        (
            "Suggested current-density unit"
            if "current density" in str(summary.get("onset_method", "")).lower()
            or "/cm" in str(summary.get("suggested_unit", ""))
            else "Suggested current unit",
            summary.get("suggested_unit") or "Not needed",
        ),
        ("Onset method", summary.get("onset_method", "Threshold crossing")),
        *threshold_items,
        ("Threshold source", summary.get("threshold_source", "User-defined")),
        ("Threshold rule", summary.get("threshold_rule", "User-defined threshold.")),
        ("Potential range", summary.get("potential_range", "Not available")),
        ("Scan rate", summary.get("scan_rate", "Not available")),
        ("Metadata completeness", summary.get("metadata_completeness", "Not available")),
        ("Missing metadata", summary.get("missing_metadata", "None")),
        ("Reference electrode source", summary.get("reference_electrode_source", "not available")),
        ("Scan rate source", summary.get("scan_rate_source", "not available")),
    ]
    if normalization_status:
        diagnostic_items.append(("Normalization status", normalization_status))
    if summary.get("unit_status") == "Unconfirmed":
        diagnostic_items.append(
            (
                "Unit confirmation",
                "Unit confirmation required before interpreting current-dependent metrics.",
            )
        )
    if normalization_note:
        diagnostic_items.append(("Normalization", normalization_note))
    if summary.get("rejected_crossing_reason"):
        diagnostic_items.append(("Rejected crossing reason", summary["rejected_crossing_reason"]))
    metric_html = "\n".join(
        f"""
        <div class="analysis-metric">
            <span>{escape_html(label)}</span>
            <strong>{escape_html(value)}</strong>
        </div>
        """
        for label, value in summary_items
    )
    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-card-header">
                <h4>Analysis overview</h4>
                <span class="status-badge status-{escape_html(status_class)}">{escape_html(status)}</span>
            </div>
            <div class="analysis-grid">{metric_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if normalization_status:
        st.caption(normalization_status)
    if summary.get("method_note"):
        st.caption(summary["method_note"])
    if summary.get("reference_note"):
        st.caption(summary["reference_note"])
    diagnostics_expanded = (
        status != "Passed"
        or summary.get("onset_confidence") not in {None, "", "Reliable"}
        or bool(summary.get("review_note"))
        or bool(summary.get("rejected_crossing_reason"))
        or "unit unconfirmed" in " ".join(str(value).lower() for value in summary.values())
    )
    with st.expander("Diagnostics", expanded=diagnostics_expanded):
        render_readable_table(
            pd.DataFrame(
                {
                    "Field": [label for label, _value in diagnostic_items],
                    "Value": [value for _label, value in diagnostic_items],
                }
            )
        )


def format_lsv_display_state(
    display_units: dict[str, Any],
    threshold_a: Optional[float],
    reference_electrode: Optional[str],
    electrode_area_metadata: Optional[str] = None,
    unit_review_required: bool = False,
    scan_rate: Optional[str] = None,
    normalization_status: Optional[str] = None,
) -> str:
    current_source = display_units.get("current_source", "Current")
    electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
    y_axis = display_units.get("y_axis", "Current density")
    x_axis = display_units.get("x_axis", "Potential")
    if y_axis in {"Current", "Current density"}:
        display_quantity = y_axis
    elif x_axis in {"Current", "Current density"}:
        display_quantity = x_axis
    else:
        display_quantity = "Current density"
    if threshold_a is None:
        threshold_text = "not set"
    elif unit_review_required:
        threshold_text = f"{format_summary_number(abs(threshold_a))} unit unconfirmed"
    else:
        threshold_text = format_lsv_threshold(
            threshold_a,
            display_units.get("current", "mA/cm^2"),
            current_source,
            electrode_area_cm2,
        )
    reference_text = reference_electrode or "Not selected"

    analyzed_quantity = "current density" if display_quantity == "Current density" else "current"
    if unit_review_required:
        unit_items = ["Units: unconfirmed"]
        suggested_unit = str(display_units.get("suggested_current_unit") or "").strip()
        if suggested_unit:
            unit_items.append(f"Suggested: {display_unit_label(suggested_unit)}")
    else:
        unit_items = [f"Units: {display_unit_label(display_units.get('current', 'mA/cm^2'))}"]
    context_items = [
        f"Analyzed as: {analyzed_quantity} vs potential",
        *unit_items,
        f"Threshold: {threshold_text}",
    ]
    if reference_text:
        context_items.append(f"Reference: {reference_text}")
    if electrode_area_metadata:
        context_items.append(f"Area: {electrode_area_metadata}")
    elif display_quantity == "Current density" and current_source == "Current":
        area_text = f"{electrode_area_cm2:.1f}" if electrode_area_cm2 == round(electrode_area_cm2) else f"{electrode_area_cm2:g}"
        context_items.append(f"Area: {area_text} cm\u00b2")
    if scan_rate:
        context_items.append(f"Scan rate: {scan_rate.lower() if scan_rate == 'Not available' else scan_rate}")
    if normalization_status:
        context_items.append(str(normalization_status))
    return " \u00b7 ".join(context_items)


def format_cv_run_context(
    current_display_unit: str,
    current_source: str,
    scan_mode: str,
    cycle_count: Any,
    analyzed_cycle_label: Optional[str] = None,
    cycle_assignment_uncertain: bool = False,
) -> str:
    y_label = "current density" if current_source == "Current density" else "current"
    details = [
        f"Analyzed as: {y_label} vs potential",
        f"Units: {display_unit_label(current_display_unit)}",
        f"Scan mode: {scan_mode.lower()}",
    ]
    if analyzed_cycle_label:
        details.append(f"Analyzed cycle: {analyzed_cycle_label}")
    elif cycle_count not in {None, "", "Not detected"}:
        cycle_count_text = str(cycle_count)
        suffix = " (inferred)" if cycle_assignment_uncertain and "inferred" not in cycle_count_text.lower() else ""
        details.append(f"Cycle count: {cycle_count_text}{suffix}")
    return " \u00b7 ".join(details)


def build_lsv_detailed_metrics(
    record: dict[str, Any],
    parsed_dataset: Optional[ParsedDataset],
    threshold_a: Optional[float],
    display_units: dict[str, Any],
    reference_electrode: Optional[str] = None,
    threshold_source: str = "User-defined",
    threshold_rule: str = "User-defined threshold.",
) -> pd.DataFrame:
    potential_v = np.asarray(record["data"].get("potential", np.array([])), dtype=float)
    current_values = np.asarray(record["data"].get("current", np.array([])), dtype=float)
    current_source = display_units.get("current_source", "Current")
    electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
    current_unit = display_units.get("current")
    if not current_unit:
        current_unit = readable_lsv_current_unit(
            current_values,
            current_source,
            electrode_area_cm2,
            prefer_density=display_units.get("y_axis", "Current density") == "Current density",
        )
    is_density = is_current_density_unit(current_unit)
    current_label = "current density" if is_density else "current"
    current_label_title = "current density" if is_density else "current"

    direction = lsv_analysis_direction(potential_v, current_values)
    onset_result = analyze_lsv_onset(potential_v, current_values, threshold_a, direction)
    onset_potential = onset_result.onset_potential
    onset_direction = onset_result.onset_direction or direction
    if onset_direction == "cathodic":
        onset_label = "Cathodic onset potential"
    elif onset_direction == "anodic":
        onset_label = "Anodic onset potential"
    else:
        onset_label = "Onset potential"

    max_current = None
    max_potential = None
    min_current = None
    min_potential = None
    if current_values.size and current_values.size == potential_v.size:
        finite_pair = np.isfinite(current_values) & np.isfinite(potential_v)
        if np.any(finite_pair):
            finite_indices = np.flatnonzero(finite_pair)
            max_idx = int(finite_indices[np.nanargmax(current_values[finite_pair])])
            min_idx = int(finite_indices[np.nanargmin(current_values[finite_pair])])
            max_current = float(current_values[max_idx])
            max_potential = float(potential_v[max_idx])
            min_current = float(current_values[min_idx])
            min_potential = float(potential_v[min_idx])

    scan_rate = "Not available"
    scan_rate_note = ""
    electrode_area_metadata = "Not detected"
    reference_metadata = "Not detected"
    scan_rate_method = "Estimated from dE/dt when time and potential columns are available."
    scan_rate_source_value = "not available"
    if parsed_dataset is not None:
        potential_col = record.get("columns", {}).get("potential")
        scan_rate_estimate = scan_rate_estimate_summary(parsed_dataset, potential_col)
        scan_rate = summarize_scan_rate_for_card(scan_rate_estimate) or "Not available"
        if scan_rate == "Not available":
            scan_rate_note = scan_rate_unavailable_reason(parsed_dataset, potential_col)
        scan_rate_method = scan_rate_method_summary(parsed_dataset, potential_col)
        if scan_rate != "Not available":
            if scan_rate_from_metadata(parsed_dataset.metadata) is not None:
                scan_rate_source_value = "metadata"
                scan_rate_method = "Parsed from file metadata."
            else:
                scan_rate_source_value = "estimated"
        electrode_area_metadata = electrode_area_metadata_summary(parsed_dataset.metadata) or "Not detected"
        reference_metadata = reference_electrode_from_metadata(parsed_dataset.metadata) or "Not detected"

    direction_method = (
        "Uses the dominant LSV response direction when available; otherwise falls back to scan direction."
    )
    threshold_method = threshold_rule
    crossing_method = (
        f"Onset was estimated using the selected threshold rule: linearly interpolated potential where "
        f"{current_label} crosses the selected threshold and stays past it for at least "
        f"{onset_result.sustained_points_required} points / "
        f"{onset_result.sustained_window_v * 1000:.0f} mV."
    )
    if onset_result.smoothing_method != "None":
        crossing_method += (
            f" Onset detection uses a {onset_result.smoothing_window}-point moving average; raw trace is preserved."
        )
    if onset_result.warning:
        crossing_method = onset_result.warning
    threshold_magnitude, crossing_threshold = lsv_threshold_magnitude_and_crossing(
        threshold_a,
        current_unit,
        current_source,
        electrode_area_cm2,
        onset_direction,
    )
    unit_review_needed = bool(display_units.get("unit_review_required"))
    if unit_review_needed:
        threshold_magnitude = f"{format_summary_number(abs(threshold_a)) if threshold_a is not None else 'Not available'} unit unconfirmed"
        crossing_threshold = (
            f"-{format_summary_number(abs(threshold_a))} unit unconfirmed"
            if threshold_a is not None and onset_direction == "cathodic"
            else threshold_magnitude
        )
        max_current_value = "Pending unit confirmation"
        min_current_value = "Pending unit confirmation"
        onset_method_value = (
            "Sustained current-density threshold crossing"
            if current_source == "Current density"
            else "Sustained signal threshold crossing"
        )
        crossing_method = "Unit confirmation required before interpreting current-dependent metrics."
    else:
        max_current_value = format_lsv_current_value(max_current, current_unit, current_source, electrode_area_cm2)
        min_current_value = format_lsv_current_value(min_current, current_unit, current_source, electrode_area_cm2)
        onset_method_value = "Sustained current-density threshold crossing" if is_density else "Sustained current threshold crossing"
    display_quantity = "Current density" if display_units.get("y_axis") == "Current density" else "Current"
    normalization_status = lsv_normalization_status_line(
        current_source,
        display_quantity,
        electrode_area_cm2,
        electrode_area_metadata,
    )
    threshold_rows = (
        [
            {
                "Metric": "Threshold magnitude",
                "Value": threshold_magnitude,
                "Method": threshold_method,
            },
            {
                "Metric": "Cathodic crossing threshold",
                "Value": crossing_threshold,
                "Method": "Signed threshold used for cathodic onset detection.",
            },
        ]
        if onset_direction == "cathodic"
        else [
            {
                "Metric": "Threshold",
                "Value": threshold_magnitude,
                "Method": threshold_method,
            }
        ]
    )
    rows = [
        {
            "Metric": "Unit status",
            "Value": "Unconfirmed" if unit_review_needed else "Confirmed",
            "Method": "Unit confirmation required before interpreting current-dependent metrics."
            if unit_review_needed
            else "Current/current-density unit is confirmed or confidently detected.",
        },
        {
            "Metric": "Suggested current-density unit" if current_source == "Current density" else "Suggested current unit",
            "Value": display_unit_label(str(display_units.get("suggested_current_unit") or "Not needed")),
            "Method": "Suggested from filename, metadata, or value magnitude when the header unit is missing.",
        },
        {
            "Metric": onset_label,
            "Value": format_lsv_potential(onset_potential, reference_electrode),
            "Method": crossing_method,
        },
        *threshold_rows,
        {
            "Metric": "Threshold source",
            "Value": threshold_source,
            "Method": threshold_rule,
        },
        {
            "Metric": "Onset method",
            "Value": onset_method_value,
            "Method": "Rejects isolated noisy or boundary crossings before selecting onset.",
        },
        {
            "Metric": "Onset reliability",
            "Value": onset_result.confidence,
            "Method": onset_result.warning or "Sustained threshold behavior was detected.",
        },
        {
            "Metric": "Sustained crossing requirement",
            "Value": f"{onset_result.sustained_points_required} points / {onset_result.sustained_window_v * 1000:.0f} mV",
            "Method": "Minimum duration/window required after threshold crossing.",
        },
        {
            "Metric": "Onset detection signal",
            "Value": (
                f"{onset_result.smoothing_method}, {onset_result.smoothing_window} points"
                if onset_result.smoothing_method != "None"
                else "Raw signal"
            ),
            "Method": "Signal used for onset detection; displayed trace remains raw unless user changes display settings.",
        },
        {
            "Metric": "Rejected crossing reason",
            "Value": onset_result.rejected_reason or "None",
            "Method": "Reason for rejecting the first candidate crossing, if any.",
        },
        {
            "Metric": "Direction",
            "Value": onset_direction.title(),
            "Method": direction_method,
        },
        {
            "Metric": f"Maximum anodic {current_label_title}",
            "Value": max_current_value,
            "Method": f"Maximum {current_label} in the normalized LSV trace.",
        },
        {
            "Metric": f"Potential at maximum anodic {current_label_title}",
            "Value": format_lsv_potential(max_potential, reference_electrode),
            "Method": f"Potential value at the maximum {current_label}.",
        },
        {
            "Metric": f"Maximum cathodic {current_label_title} / minimum {current_label_title}",
            "Value": min_current_value,
            "Method": f"Minimum {current_label} in the normalized LSV trace.",
        },
        {
            "Metric": f"Potential at maximum cathodic {current_label_title}",
            "Value": format_lsv_potential(min_potential, reference_electrode),
            "Method": f"Potential value at the minimum {current_label}.",
        },
        {
            "Metric": "Potential range",
            "Value": format_fixed_range(potential_v[np.isfinite(potential_v)], "V", 3),
            "Method": "Minimum to maximum potential after unit normalization.",
        },
        {
            "Metric": "Scan rate",
            "Value": scan_rate,
            "Method": scan_rate_note or scan_rate_method,
        },
        {
            "Metric": "Scan rate source",
            "Value": scan_rate_source_value,
            "Method": scan_rate_method,
        },
        {
            "Metric": "Reference electrode metadata",
            "Value": reference_metadata,
            "Method": "Reference electrode parsed from file metadata, when available.",
        },
        {
            "Metric": "Electrode area metadata",
            "Value": electrode_area_metadata,
            "Method": "Electrode area parsed from file metadata, when available.",
        },
        {
            "Metric": "Normalization status",
            "Value": normalization_status,
            "Method": "Tracks whether the uploaded y-signal was raw current, current density, or normalized by VoltScope.",
        },
        {
            "Metric": "Number of points",
            "Value": str(int(len(potential_v))),
            "Method": "Numeric rows used after parser cleanup and column normalization.",
        },
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Method"])


def build_lsv_onset_annotation(
    record: dict[str, Any],
    threshold_a: Optional[float],
    display_units: dict[str, Any],
    reference_electrode: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if threshold_a is None or threshold_a <= 0:
        return None

    potential_v = np.asarray(record["data"].get("potential", np.array([])), dtype=float)
    current_values = np.asarray(record["data"].get("current", np.array([])), dtype=float)
    if potential_v.size < 2 or current_values.size != potential_v.size:
        return None

    direction = lsv_analysis_direction(potential_v, current_values)
    onset_result = analyze_lsv_onset(potential_v, current_values, threshold_a, direction)
    onset_potential = onset_result.onset_potential
    onset_direction = onset_result.onset_direction
    if onset_result.status != "Passed" or onset_potential is None or onset_direction is None:
        return None

    potential_unit = display_units.get("potential", "V")
    current_unit = display_units.get("current", "uA/cm^2")
    current_source = display_units.get("current_source", "Current")
    electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
    threshold_base = threshold_a if onset_direction == "anodic" else -threshold_a
    potential_display = convert_role_from_base(np.array([onset_potential]), "potential", potential_unit)[0]
    current_display = current_to_display(
        np.array([threshold_base]),
        current_unit,
        current_source,
        electrode_area_cm2,
    )[0]
    reference = f" vs {reference_electrode}" if reference_electrode else ""
    onset_label = f"Onset = {onset_potential:.3f} V{reference}"
    threshold_unit_label = "unit unconfirmed" if display_units.get("unit_review_required") else display_unit_label(current_unit)
    return {
        "onset_potential_v": float(onset_potential),
        "onset_potential_display": float(potential_display),
        "threshold_display": float(current_display),
        "direction": onset_direction,
        "label": onset_label,
        "hover": (
            f"{onset_label}<br>"
            f"{lsv_crossing_threshold_label(onset_direction)}: {current_display:.4g} {threshold_unit_label}<br>"
            f"Reliability: {onset_result.confidence}"
        ),
    }


def add_lsv_onset_annotation_to_plot(
    fig: go.Figure,
    record: dict[str, Any],
    threshold_a: Optional[float],
    display_units: dict[str, Any],
    reference_electrode: Optional[str],
) -> None:
    annotation = build_lsv_onset_annotation(record, threshold_a, display_units, reference_electrode)
    if annotation is None:
        return

    x_axis = display_units.get("x_axis", "Potential")
    y_axis = display_units.get("y_axis", "Current density")
    marker_x = annotation["onset_potential_display"] if x_axis == "Potential" else annotation["threshold_display"]
    marker_y = annotation["threshold_display"] if y_axis != "Potential" else annotation["onset_potential_display"]

    if x_axis == "Potential":
        fig.add_vline(
            x=annotation["onset_potential_display"],
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=annotation["label"],
            annotation_position="top",
        )
    elif y_axis == "Potential":
        fig.add_hline(
            y=annotation["onset_potential_display"],
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=annotation["label"],
            annotation_position="right",
        )

    fig.add_trace(
        go.Scatter(
            x=[marker_x],
            y=[marker_y],
            mode="markers",
            name="Onset",
            marker={"color": "#dc2626", "size": 11, "symbol": "circle"},
            hovertemplate=f"{annotation['hover']}<extra></extra>",
        )
    )



# ---------- Multi-experiment analysis helpers ----------


def experiment_label(experiment_type: str) -> str:
    return EXPERIMENT_TYPES.get(experiment_type, EXPERIMENT_TYPES["cv"])["label"]


def experiment_short_label(experiment_type: str) -> str:
    return EXPERIMENT_TYPES.get(experiment_type, EXPERIMENT_TYPES["cv"])["short_label"]


def effective_experiment_type(experiment: dict[str, Any]) -> str:
    outputs = experiment.get("analysis_outputs", {})
    output_type = outputs.get("experiment_type") if isinstance(outputs, dict) else None
    if output_type in EXPERIMENT_TYPES and output_type != "auto":
        return str(output_type)

    stored_type = experiment.get("experiment_type", "auto")
    if stored_type in EXPERIMENT_TYPES and stored_type != "auto":
        return str(stored_type)

    file_types = {
        file_record.get("summary", {}).get("experiment_type")
        for file_record in experiment.get("files", {}).values()
        if isinstance(file_record, dict) and isinstance(file_record.get("summary"), dict)
    }
    file_types = {str(file_type) for file_type in file_types if file_type in EXPERIMENT_TYPES and file_type != "auto"}
    if len(file_types) == 1:
        return next(iter(file_types))

    return "auto"


def experiment_header_type_label(experiment_type: str, detection_warnings: Optional[list[str]] = None) -> str:
    if detection_warnings:
        return "Auto / needs review"
    return experiment_short_label(experiment_type)


def role_label(role: str) -> str:
    labels = {
        "potential": "Potential",
        "current": "Current",
        "time": "Time",
        "frequency": "Frequency",
        "zreal": "Z real",
        "zimag": "Z imaginary",
        "zmod": "|Z| magnitude",
        "phase": "Phase",
    }
    return labels.get(role, role)


def required_roles_for_experiment(experiment_type: str) -> list[str]:
    if experiment_type == "ca":
        return ["time", "current"]
    if experiment_type == "cp":
        return ["time", "potential"]
    if experiment_type == "lsv":
        return ["potential", "current"]
    if experiment_type == "eis":
        return ["zreal", "zimag"]
    return ["potential", "current"]


def optional_roles_for_experiment(experiment_type: str) -> list[str]:
    if experiment_type == "eis":
        return ["frequency", "zmod", "phase"]
    return []


def text_suggests_cv_or_lsv(dataset: ParsedDataset) -> Optional[str]:
    metadata_text = " ".join([dataset.filename, *dataset.headers, *dataset.metadata.keys(), *dataset.metadata.values()])
    text = metadata_text.lower()
    text = text.replace("_", " ").replace("-", " ")
    if re.search(r"\b(cyclic voltammetry|cyclic voltammogram|cv)\b", text):
        return "cv"
    if re.search(r"\b(linear sweep voltammetry|linear sweep|lsv)\b", text):
        return "lsv"
    return None


def potential_trace_suggests_cv_or_lsv(dataset: ParsedDataset) -> tuple[str, str]:
    detected = detect_experiment_columns(dataset, "cv")
    potential_col = detected.get("potential")
    if not potential_col:
        return "lsv", "No reliable potential column was found, so VoltScope defaulted to LSV."

    potential_values = dataset.dataframe[potential_col].to_numpy(dtype=float)
    potential_unit = resolve_role_unit_from_values("Auto", dataset, "potential", potential_col, potential_values)
    potential_v = convert_potential_to_volts(potential_values, potential_unit)
    potential_v = potential_v[np.isfinite(potential_v)]
    if potential_v.size < 5:
        return "lsv", "Too few potential points for scan-direction detection, so VoltScope defaulted to LSV."

    potential_range = float(np.nanmax(potential_v) - np.nanmin(potential_v))
    if potential_range <= 1e-12:
        return "lsv", "Potential values are effectively constant, so VoltScope defaulted to LSV."

    diffs = np.diff(potential_v)
    significant_step = max(potential_range * 5e-4, 1e-7)
    significant_diffs = diffs[np.abs(diffs) >= significant_step]
    if significant_diffs.size < 3:
        return "lsv", "Potential changes were too small for scan-direction detection, so VoltScope defaulted to LSV."

    signs = np.sign(significant_diffs)
    compressed_signs = [int(signs[0])]
    for sign in signs[1:]:
        sign_int = int(sign)
        if sign_int != compressed_signs[-1]:
            compressed_signs.append(sign_int)

    turning_points = max(0, len(compressed_signs) - 1)
    if turning_points >= 1:
        return "cv", f"Detected {turning_points} scan-direction change(s) in the potential trace."

    increasing_fraction = float(np.mean(diffs >= -significant_step))
    decreasing_fraction = float(np.mean(diffs <= significant_step))
    monotonic_fraction = max(increasing_fraction, decreasing_fraction)
    return "lsv", f"Potential is monotonic across {monotonic_fraction:.0%} of steps."


def detect_uploaded_experiment_type(parsed_datasets: list[ParsedDataset]) -> tuple[str, dict[str, str], list[str]]:
    if not parsed_datasets:
        return "cv", {}, ["No uploaded files were available for experiment-type detection."]

    detections: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for dataset in parsed_datasets:
        text_hint = text_suggests_cv_or_lsv(dataset)
        if text_hint:
            detections[dataset.filename] = text_hint
            reasons[dataset.filename] = "Detected from file metadata, filename, or headers."
            continue
        detected_type, reason = potential_trace_suggests_cv_or_lsv(dataset)
        detections[dataset.filename] = detected_type
        reasons[dataset.filename] = reason

    counts = {experiment_type: list(detections.values()).count(experiment_type) for experiment_type in {"cv", "lsv"}}
    if counts["cv"] == counts["lsv"]:
        selected_type = detections[parsed_datasets[0].filename]
    else:
        selected_type = "cv" if counts["cv"] > counts["lsv"] else "lsv"

    warnings = []
    if len(set(detections.values())) > 1:
        readable = ", ".join(f"{filename}: {experiment_short_label(kind)}" for filename, kind in detections.items())
        warnings.append(
            f"Uploaded files appear to contain mixed CV/LSV data ({readable}). VoltScope will analyze this experiment as {experiment_short_label(selected_type)}."
        )

    return selected_type, reasons, warnings


def format_summary_number(value: Any) -> str:
    if value is None:
        return "Not available"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "Not available"
    return f"{numeric:.6g}"


def display_unit_label(unit: str) -> str:
    return str(unit).replace("uA", "\u00b5A").replace("cm^2", "cm\u00b2")


def format_fixed_range(values: np.ndarray, unit: str, decimals: int) -> str:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return "Not available"
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    suffix = f" {display_unit_label(unit)}" if unit else ""
    return f"{low:.{decimals}f} to {high:.{decimals}f}{suffix}"


def format_numeric_range(values: np.ndarray, unit: str) -> str:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return "Not available"
    low = format_summary_number(np.nanmin(finite))
    high = format_summary_number(np.nanmax(finite))
    suffix = f" {unit}" if unit else ""
    return f"{low} to {high}{suffix}"


def display_delimiter(delimiter: str) -> str:
    labels = {",": "comma", "\t": "tab", ";": "semicolon", "whitespace": "whitespace"}
    return labels.get(delimiter, delimiter)


def filename_suggests_tabular_text(filename: str) -> bool:
    name = Path(filename).name.lower()
    return bool(re.search(r"(^|[_\-.])tab([_\-.]|$)", name))


def display_delimiter_for_dataset(dataset: ParsedDataset) -> str:
    if dataset.delimiter == "whitespace" and filename_suggests_tabular_text(dataset.filename):
        return "tab / whitespace"
    return display_delimiter(dataset.delimiter)


def delimiter_fallback_from_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".tsv":
        return "\t"
    return ","


def current_source_guess_for_column(column: Optional[str]) -> str:
    if column and header_indicates_current_density(column):
        return "Current density"
    return "Current"


def current_source_guess_for_dataset(dataset: ParsedDataset, column: Optional[str]) -> str:
    if column and header_indicates_current_density(column):
        return "Current density"
    if dataset_suggests_current_density_signal(dataset, column):
        return "Current density"
    return "Current"


def uploaded_signal_kind(
    dataset: Optional[ParsedDataset],
    current_col: Optional[str],
    current_source: str,
    selected_current_unit: str = "Auto",
) -> str:
    if current_source == "Current density":
        return "current_density"
    if dataset is not None and current_unit_is_ambiguous(dataset, current_col, selected_current_unit):
        return "unknown"
    return "raw_current"


def readable_parser_current_unit_for_lsv(
    parsed_datasets: list[ParsedDataset],
    current_source_override: Optional[str] = None,
    electrode_area_cm2: float = 1.0,
) -> tuple[Optional[str], Optional[str]]:
    current_arrays: list[np.ndarray] = []
    detected_sources: list[str] = []
    for dataset in parsed_datasets:
        detected = detect_experiment_columns(dataset, "lsv")
        current_col = detected.get("current")
        if not current_col or current_col not in dataset.dataframe:
            continue
        current_values = dataset.dataframe[current_col].to_numpy(dtype=float)
        current_unit = resolve_role_unit_from_values("Auto", dataset, "current", current_col, current_values)
        current_arrays.append(convert_current_to_amps(current_values, current_unit))
        detected_source = current_source_guess_for_dataset(dataset, current_col)
        detected_sources.append(
            "Current density"
            if detected_source == "Current density"
            else current_source_override or detected_source
        )

    if not current_arrays:
        return None, current_source_override

    current_source = current_source_override
    if current_source is None and detected_sources:
        unique_sources = set(detected_sources)
        current_source = detected_sources[0] if len(unique_sources) == 1 else "Current"

    current_source = current_source or "Current"
    display_unit = readable_lsv_current_unit(
        np.concatenate(current_arrays),
        current_source,
        electrode_area_cm2,
        prefer_density=current_source == "Current density",
    )
    return display_unit, current_source


def current_unit_label_for_summary(current_unit: str, current_source: str) -> str:
    if current_source == "Current density":
        return current_density_unit_from_current_unit(current_unit)
    return current_unit


def find_cycle_column(dataset: ParsedDataset) -> Optional[str]:
    for column in dataset.dataframe.columns:
        compact = re.sub(r"[^a-z0-9]+", "", str(column).lower())
        if compact == "cycle" or compact.endswith("cycle"):
            return column
    return None


def infer_cycle_count_from_potential(potential_v: np.ndarray) -> Optional[int]:
    finite = potential_v[np.isfinite(potential_v)]
    if finite.size < 5:
        return None
    segments = split_scan_segments(finite)
    if len(segments) < 2:
        return None
    return max(1, (len(segments) + 1) // 2)


def cycle_count_summary(dataset: ParsedDataset, potential_col: Optional[str] = None) -> str:
    cycle_col = find_cycle_column(dataset)
    if not cycle_col:
        if potential_col and potential_col in dataset.dataframe:
            potential_values = dataset.dataframe[potential_col].to_numpy(dtype=float)
            potential_unit = resolve_role_unit_from_values("Auto", dataset, "potential", potential_col, potential_values)
            inferred_count = infer_cycle_count_from_potential(convert_potential_to_volts(potential_values, potential_unit))
            if inferred_count is not None:
                return f"{inferred_count} (inferred)"
        return "Not detected"
    finite = dataset.dataframe[cycle_col].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return f"Not detected ({cycle_col})"
    count = len(np.unique(finite))
    return str(count)


def available_cycle_values(cycle_values: Optional[np.ndarray]) -> list[float]:
    if cycle_values is None:
        return []
    finite = cycle_values[np.isfinite(cycle_values)]
    if finite.size == 0:
        return []
    return sorted(float(value) for value in np.unique(finite))


def cycle_display_label(cycle_value: float) -> str:
    if float(cycle_value).is_integer():
        return f"Cycle {int(cycle_value)}"
    return f"Cycle {cycle_value:g}"


def cycle_selection_mask(cycle_values: Optional[np.ndarray], selected_cycle: Optional[float], length: int) -> np.ndarray:
    if cycle_values is None or selected_cycle is None or len(cycle_values) != length:
        return np.ones(length, dtype=bool)
    return np.isclose(cycle_values, selected_cycle, rtol=0.0, atol=1e-9)


def analyzed_cycle_label(selected_cycle_value: Optional[float], cycle_count: int) -> Optional[str]:
    if selected_cycle_value is None or cycle_count <= 1:
        return None
    return f"{cycle_display_label(selected_cycle_value)} of {cycle_count}"


def cycle_assignment_messages(
    dataset: AnalyzedDataset,
    cycle_count: int,
    current_cycle_label: Optional[str] = None,
) -> list[tuple[str, str]]:
    if cycle_count <= 1:
        return []

    source_phrase = "detected"
    level = "info"
    if dataset.cycle_source == "column":
        source_phrase = "from cycle column"
    elif dataset.cycle_source == "inferred":
        source_phrase = "inferred from potential turning points"
        level = "warning"

    cycle_context = current_cycle_label or "the selected cycle"
    message = (
        f"Multi-cycle CV detected: {cycle_count} cycles {source_phrase}. "
        f"Currently analyzing: {cycle_context}. "
        "Metrics and peak markers use the selected cycle; plot may show all cycles."
    )
    if dataset.cycle_source == "inferred":
        message += " Cycle splitting may be approximate."
    return [(level, message)]


def selected_cycle_trace_name(cycle_value: float, is_selected_cycle: bool) -> str:
    cycle_name = cycle_display_label(cycle_value)
    if is_selected_cycle:
        return f"{cycle_name} \u2014 selected"
    return cycle_name


def selected_cycle_trace_style(is_selected_cycle: bool) -> tuple[float, float]:
    if is_selected_cycle:
        return 1.0, 4.0
    return 0.28, 1.4


def representative_scan_rate_v_s(time_s: np.ndarray, potential_v: np.ndarray) -> Optional[float]:
    finite = np.isfinite(time_s) & np.isfinite(potential_v)
    time_s = time_s[finite]
    potential_v = potential_v[finite]
    if len(time_s) < 3:
        return None

    segments = split_scan_segments(potential_v)
    if not segments:
        segments = [(0, len(potential_v), "single")]

    segment_rates = []
    for start, end, _direction in segments:
        segment_time = time_s[start:end]
        segment_potential = potential_v[start:end]
        if len(segment_time) < 3:
            continue
        dt = np.diff(segment_time)
        de = np.diff(segment_potential)
        valid = np.isfinite(dt) & np.isfinite(de) & (np.abs(dt) > 1e-15) & (np.abs(de) > 1e-15)
        if not np.any(valid):
            continue
        rates = np.abs(de[valid] / dt[valid])
        rates = rates[np.isfinite(rates)]
        if rates.size:
            segment_rates.append(float(np.nanmedian(rates)))

    if not segment_rates:
        return None
    representative = float(np.nanmedian(segment_rates))
    return representative if np.isfinite(representative) and representative > 0 else None


def scan_rate_estimate_summary(dataset: ParsedDataset, potential_col: Optional[str]) -> str:
    time_col = dataset.detected_time_col
    if not time_col:
        excluded = {column for column in [potential_col, dataset.detected_current_col] if column}
        time_col = detect_column_by_role(dataset.dataframe, "time", excluded=excluded)
    if not time_col or not potential_col or time_col not in dataset.dataframe or potential_col not in dataset.dataframe:
        metadata_scan_rate = scan_rate_from_metadata(dataset.metadata)
        return scan_rate_summary_from_value(metadata_scan_rate) if metadata_scan_rate is not None else "Not available"

    time_values = dataset.dataframe[time_col].to_numpy(dtype=float)
    potential_values = dataset.dataframe[potential_col].to_numpy(dtype=float)
    time_unit = resolve_role_unit_from_values("Auto", dataset, "time", time_col, time_values)
    potential_unit = resolve_role_unit_from_values("Auto", dataset, "potential", potential_col, potential_values)
    time_s = convert_role_to_base(time_values, "time", time_unit)
    potential_v = convert_role_to_base(potential_values, "potential", potential_unit)
    scan_rate = representative_scan_rate_v_s(time_s, potential_v)
    if scan_rate is not None:
        return scan_rate_summary_from_value(scan_rate)
    metadata_scan_rate = scan_rate_from_metadata(dataset.metadata)
    return scan_rate_summary_from_value(metadata_scan_rate) if metadata_scan_rate is not None else "Not available"


def scan_rate_unavailable_reason(dataset: ParsedDataset, potential_col: Optional[str]) -> str:
    if scan_rate_from_metadata(dataset.metadata) is not None:
        return ""
    time_col = dataset.detected_time_col
    if not time_col:
        excluded = {column for column in [potential_col, dataset.detected_current_col] if column}
        time_col = detect_column_by_role(dataset.dataframe, "time", excluded=excluded)
    if not time_col:
        return (
            "Scan rate was not found in metadata and could not be estimated because no time column "
            "was detected. Add scan rate metadata if scan-rate-dependent interpretation is needed."
        )
    if not potential_col or time_col not in dataset.dataframe or potential_col not in dataset.dataframe:
        return "Scan rate could not be estimated because the required time or potential column was unavailable."
    return "Scan rate could not be estimated from the available time and potential values."


def scan_rate_method_summary(dataset: Optional[ParsedDataset], potential_col: Optional[str]) -> str:
    if dataset is None:
        return "Estimated from dE/dt when time and potential columns are available."
    time_col = dataset.detected_time_col
    if not time_col:
        excluded = {column for column in [potential_col, dataset.detected_current_col] if column}
        time_col = detect_column_by_role(dataset.dataframe, "time", excluded=excluded)
    if time_col and potential_col and time_col in dataset.dataframe and potential_col in dataset.dataframe:
        time_values = dataset.dataframe[time_col].to_numpy(dtype=float)
        potential_values = dataset.dataframe[potential_col].to_numpy(dtype=float)
        time_unit = resolve_role_unit_from_values("Auto", dataset, "time", time_col, time_values)
        potential_unit = resolve_role_unit_from_values("Auto", dataset, "potential", potential_col, potential_values)
        scan_rate = representative_scan_rate_v_s(
            convert_role_to_base(time_values, "time", time_unit),
            convert_role_to_base(potential_values, "potential", potential_unit),
        )
        if scan_rate is not None:
            return "Estimated from dE/dt using time and potential columns."
    if scan_rate_from_metadata(dataset.metadata) is not None:
        return "Parsed from file metadata."
    return scan_rate_unavailable_reason(dataset, potential_col)


def scan_rate_note_for_dataset(dataset: Optional[ParsedDataset], potential_col: Optional[str]) -> str:
    if dataset is None:
        return ""
    if scan_rate_estimate_summary(dataset, potential_col) != "Not available":
        return ""
    return scan_rate_unavailable_reason(dataset, potential_col)


def parser_unit_conversion_summary(
    potential_unit: str,
    current_unit: str,
    current_source: str,
    display_current_unit: Optional[str] = None,
) -> str:
    current_display_unit = current_unit_label_for_summary(current_unit, current_source)
    current_base_unit = "A/cm^2" if current_source == "Current density" else "A"
    summary = (
        f"Potential: {display_unit_label(potential_unit)} -> V; "
        f"{current_source}: {display_unit_label(current_display_unit)} -> {display_unit_label(current_base_unit)}"
    )
    if display_current_unit:
        summary += f"; Display unit: {display_unit_label(display_current_unit)}"
    return summary


def column_label_looks_inferred(column: Optional[str]) -> bool:
    if column is None:
        return False
    text = str(column).strip()
    if not text:
        return True
    if re.fullmatch(r"column\s+\d+", text, flags=re.IGNORECASE):
        return True
    return bool(_NUMBER_RE.fullmatch(text.replace(",", "")))


def user_facing_column_label(dataset: ParsedDataset, column: Optional[str]) -> str:
    if not column:
        return "Not detected"
    if column_label_looks_inferred(column):
        try:
            column_number = dataset.headers.index(column) + 1
        except ValueError:
            column_number = 0
        if column_number:
            return f"Inferred column {column_number}"
        return "Inferred column"
    return str(column)


def parser_summary_details(
    dataset: ParsedDataset,
    experiment_type: str,
    display_current_unit: Optional[str] = None,
    current_source_override: Optional[str] = None,
    electrode_area_cm2: float = 1.0,
    current_input_unit_override: Optional[str] = None,
) -> dict[str, Any]:
    detected = detect_experiment_columns(dataset, experiment_type)
    potential_col = detected.get("potential")
    current_col = detected.get("current")

    potential_unit = "Not detected"
    potential_range = "Not available"
    raw_potential_range = "Not available"
    if potential_col:
        potential_values = dataset.dataframe[potential_col].to_numpy(dtype=float)
        potential_unit = resolve_role_unit_from_values("Auto", dataset, "potential", potential_col, potential_values)
        raw_potential_range = format_numeric_range(potential_values, potential_unit)
        potential_range = format_fixed_range(convert_potential_to_volts(potential_values, potential_unit), "V", 3)

    current_unit = "Not detected"
    current_range = "Not available"
    raw_current_range = "Not available"
    analysis_current_range = "Not available"
    original_current_range = "Not available"
    suggested_current_unit = "Not needed"
    current_unit_warning = ""
    current_magnitude_warning = ""
    unit_header_warning = ""
    unit_header_audit = ""
    detected_current_source = current_source_guess_for_dataset(dataset, current_col)
    current_source = (
        "Current density"
        if detected_current_source == "Current density"
        else current_source_override or detected_current_source
    )
    if current_col:
        current_values = dataset.dataframe[current_col].to_numpy(dtype=float)
        current_unit = (
            current_input_unit_override
            or resolve_role_unit_from_values("Auto", dataset, "current", current_col, current_values)
        )
        current_label = current_unit_label_for_summary(current_unit, current_source)
        original_current_range = format_numeric_range(current_values, "raw values")
        if current_unit_is_ambiguous(dataset, current_col, current_input_unit_override or "Auto"):
            suggested_unit = suggested_current_unit_for_context(current_values, current_source)
            suggested_label = display_unit_label(current_unit_label_for_summary(suggested_unit, current_source))
            suggested_current_unit = (
                f"{suggested_label} "
                "based on value magnitude"
            )
            current_unit_warning = current_unit_magnitude_warning(
                dataset,
                current_col,
                current_input_unit_override or "Auto",
                current_unit,
                current_source,
                electrode_area_cm2,
            )
        raw_current_range = format_numeric_range(
            current_values, display_unit_label(current_label)
        )
        current_base_values = convert_current_to_amps(current_values, current_unit)
        analysis_unit = "A/cm^2" if current_source == "Current density" else "A"
        analysis_current_range = format_numeric_range(current_base_values, display_unit_label(analysis_unit))
        current_magnitude_warning = current_magnitude_sanity_warning(current_base_values, current_source)
        if display_current_unit:
            current_range = format_fixed_range(
                current_to_display(current_base_values, display_current_unit, current_source, electrode_area_cm2),
                display_current_unit,
                2,
            )
        else:
            current_range = raw_current_range

    unit_header_warning = missing_unit_header_warning(
        dataset,
        potential_col,
        current_col,
        current_input_unit_override or "Auto",
    )
    unit_header_audit = unit_header_audit_note(
        current_col,
        current_input_unit_override or "Auto",
        current_source,
    )
    cleanup_notice = dropped_rows_cleanup_notice(len(dataset.dataframe), dataset.rows_dropped)
    cleanup_messages = [cleanup_notice] if cleanup_notice else []
    warning_messages = [*dataset.warnings, *cleanup_messages]
    warning_messages = unique_messages([message for message in warning_messages if message])
    warnings = "; ".join(warning_messages) if warning_messages else "None"
    cleanup_severity = dropped_rows_severity(len(dataset.dataframe), dataset.rows_dropped)
    current_label = current_unit_label_for_summary(current_unit, current_source)
    current_label_display = display_unit_label(current_label)
    analysis_current_unit = "A/cm^2" if current_source == "Current density" else "A"
    analysis_current_unit_display = display_unit_label(analysis_current_unit)
    display_current_unit_display = display_unit_label(display_current_unit) if display_current_unit else current_label_display
    if display_current_unit:
        units = (
            f"Potential {display_unit_label(potential_unit)}; "
            f"{current_source} {current_label_display} \u2192 analysis {analysis_current_unit_display} "
            f"\u2192 displayed as {display_current_unit_display}"
        )
    else:
        units = f"Potential: {display_unit_label(potential_unit)}; {current_source}: {current_label_display}"

    metadata_rows_before_header = max((dataset.header_row or dataset.data_start_row) - 1, 0)
    scan_rate_estimate = scan_rate_estimate_summary(dataset, potential_col)
    scan_rate_method = scan_rate_method_summary(dataset, potential_col)
    scan_rate_source = scan_rate_source_from_method(scan_rate_method)
    scan_rate_note = ""
    if scan_rate_estimate == "Not available":
        scan_rate_note = scan_rate_unavailable_reason(dataset, potential_col)
    reference_electrode_metadata = reference_electrode_from_metadata(dataset.metadata) or "Not detected"
    electrode_area_metadata = electrode_area_metadata_summary(dataset.metadata) or "Not detected"

    return {
        "file_name": dataset.filename,
        "experiment_type": experiment_short_label(experiment_type),
        "rows_imported": len(dataset.dataframe),
        "y_quantity": current_source,
        "potential_column": user_facing_column_label(dataset, potential_col),
        "current_column": user_facing_column_label(dataset, current_col),
        "raw_potential_column": potential_col or "Not detected",
        "raw_current_column": current_col or "Not detected",
        "units": units,
        "potential_range": potential_range,
        "current_range": current_range,
        "raw_potential_range": raw_potential_range,
        "raw_current_range": raw_current_range,
        "original_current_values": original_current_range,
        "analysis_current_range": analysis_current_range,
        "raw_current_unit": current_label_display,
        "analysis_current_unit": analysis_current_unit_display,
        "display_current_unit": display_current_unit_display,
        "warnings": warnings,
        "current_unit_confidence": current_unit_confidence(
            dataset,
            current_col,
            current_input_unit_override or "Auto",
        ),
        "current_unit_confidence_detail": current_unit_confidence_detail(
            dataset,
            current_col,
            current_input_unit_override or "Auto",
        ),
        "suggested_current_unit": suggested_current_unit,
        "unit_header_warning": unit_header_warning,
        "unit_header_audit": unit_header_audit,
        "current_unit_warning": current_unit_warning,
        "current_magnitude_warning": current_magnitude_warning,
        "delimiter": display_delimiter_for_dataset(dataset),
        "header_row": f"Line {dataset.header_row}" if dataset.header_row else "Generated headers",
        "metadata_rows_before_header": metadata_rows_before_header,
        "rows_skipped": dataset.rows_skipped,
        "missing_values": dataset.missing_values,
        "rows_dropped": dataset.rows_dropped,
        "rows_dropped_reason": dropped_rows_reason(dataset.rows_dropped),
        "rows_dropped_fraction": dropped_rows_fraction_label(len(dataset.dataframe), dataset.rows_dropped),
        "parser_cleanup_notice": cleanup_notice,
        "parser_cleanup_severity": cleanup_severity,
        "unit_conversion": parser_unit_conversion_summary(potential_unit, current_unit, current_source, display_current_unit)
        if potential_col and current_col
        else "Not available",
        "scan_rate_estimate": scan_rate_estimate,
        "scan_rate_method": scan_rate_method,
        "scan_rate_source": scan_rate_source,
        "scan_rate_note": scan_rate_note,
        "cycle_count": cycle_count_summary(dataset, potential_col),
        "reference_electrode_metadata": reference_electrode_metadata,
        "electrode_area_metadata": electrode_area_metadata,
    }


def make_parser_summary(
    parsed_datasets: list[ParsedDataset],
    experiment_type: str,
    display_current_unit: Optional[str] = None,
    current_source_override: Optional[str] = None,
    electrode_area_cm2: float = 1.0,
    current_input_unit_override: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    roles = required_roles_for_experiment(experiment_type) + optional_roles_for_experiment(experiment_type)
    for dataset in parsed_datasets:
        detected = detect_experiment_columns(dataset, experiment_type)
        details = parser_summary_details(
            dataset,
            experiment_type,
            display_current_unit,
            current_source_override,
            electrode_area_cm2,
            current_input_unit_override,
        )
        row = {
            "file": details["file_name"],
            "experiment_type": details["experiment_type"],
            "rows_imported": details["rows_imported"],
            "columns": len(dataset.headers),
            "delimiter": dataset.delimiter,
            "metadata_rows_before_header": details["metadata_rows_before_header"],
            "header_row": details["header_row"],
            "rows_skipped_before_numeric_data": details["rows_skipped"],
            "missing_values": details["missing_values"],
            "rows_dropped": details["rows_dropped"],
            "rows_dropped_reason": details["rows_dropped_reason"],
            "rows_dropped_fraction": details["rows_dropped_fraction"],
            "parser_cleanup_severity": details["parser_cleanup_severity"],
            "metadata_fields": len(dataset.metadata),
            "potential_range": details["potential_range"],
            "current_range": details["current_range"],
            "unit_conversion": details["unit_conversion"],
            "current_unit_confidence": details["current_unit_confidence"],
            "current_unit_confidence_detail": details["current_unit_confidence_detail"],
            "suggested_current_unit": details["suggested_current_unit"],
            "unit_header_warning": details["unit_header_warning"],
            "unit_header_audit": details["unit_header_audit"],
            "current_magnitude_warning": details["current_magnitude_warning"],
            "scan_rate_estimate": details["scan_rate_estimate"],
            "reference_electrode_metadata": details["reference_electrode_metadata"],
            "electrode_area_metadata": details["electrode_area_metadata"],
            "cycle_count": details["cycle_count"],
            "warnings": details["warnings"],
        }
        for role in roles:
            row[f"detected_{role}"] = detected.get(role)
            column = detected.get(role)
            if column and role in {"potential", "current"}:
                values = dataset.dataframe[column].to_numpy(dtype=float)
                row[f"{role}_unit"] = resolve_role_unit_from_values("Auto", dataset, role, column, values)
        if experiment_type in {"cv", "lsv"} and detected.get("current"):
            row["current_source_guess"] = (
                current_source_guess_for_dataset(dataset, detected["current"])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cv_lsv_backend_warnings(parsed_datasets: list[ParsedDataset], experiment_type: str) -> list[str]:
    if experiment_type not in {"cv", "lsv"}:
        return []

    warnings: list[str] = []
    for dataset in parsed_datasets:
        detected = detect_experiment_columns(dataset, experiment_type)
        potential_col = detected.get("potential")
        current_col = detected.get("current")
        if not potential_col:
            warnings.append(f"{dataset.filename}: potential column could not be detected reliably.")
        if not current_col:
            warnings.append(f"{dataset.filename}: current/current-density column could not be detected reliably.")
        if potential_col and current_col and potential_col == current_col:
            warnings.append(f"{dataset.filename}: detected potential and current columns are the same.")

        if potential_col:
            values = dataset.dataframe[potential_col].to_numpy(dtype=float)
            unit = resolve_role_unit_from_values("Auto", dataset, "potential", potential_col, values)
            converted = convert_potential_to_volts(values, unit)
            finite = converted[np.isfinite(converted)]
            if finite.size:
                potential_range = float(np.nanmax(finite) - np.nanmin(finite))
                if potential_range <= 1e-9:
                    warnings.append(f"{dataset.filename}: detected potential column is effectively constant.")
                elif potential_range > 20:
                    warnings.append(f"{dataset.filename}: detected potential range is unusually large after conversion.")

    if len(parsed_datasets) > 1:
        detected_pairs = [
            tuple(detect_experiment_columns(dataset, experiment_type).get(role) for role in ["potential", "current"])
            for dataset in parsed_datasets
        ]
        if len(set(detected_pairs)) > 1:
            warnings.append(
                "Uploaded files do not all use the same detected potential/current column names. Auto mode will normalize each file independently."
            )
        current_source_guesses = []
        for dataset in parsed_datasets:
            current_col = detect_experiment_columns(dataset, experiment_type).get("current")
            if current_col:
                current_source_guesses.append(current_source_guess_for_dataset(dataset, current_col))
        if len(set(current_source_guesses)) > 1:
            warnings.append(
                "Uploaded files appear to mix raw current and current-density columns. Use separate experiments or verify the current-source control before analysis."
            )

    return warnings


def column_is_negative_imaginary(column: Optional[str]) -> bool:
    if not column:
        return False
    label = normalize_label(column)
    return label.startswith("-") or "-z" in label or "minus" in label


def normalize_generic_dataset(
    dataset: ParsedDataset,
    experiment_type: str,
    role_columns: dict[str, Optional[str]],
    selected_units: dict[str, str],
) -> Optional[dict[str, Any]]:
    warnings = list(dataset.warnings)
    sample_name = dataset.filename.rsplit(".", 1)[0]

    if experiment_type == "eis":
        zreal_col = role_columns.get("zreal")
        zimag_col = role_columns.get("zimag")
        zmod_col = role_columns.get("zmod")
        phase_col = role_columns.get("phase")
        frequency_col = role_columns.get("frequency")

        if zreal_col and zimag_col:
            used_cols = [zreal_col, zimag_col]
            if frequency_col:
                used_cols.append(frequency_col)
            working = dataset.dataframe[used_cols].dropna().copy()
            zreal_unit = resolve_role_unit(selected_units.get("zreal", "Auto"), dataset, "zreal", zreal_col)
            zimag_unit = resolve_role_unit(selected_units.get("zimag", "Auto"), dataset, "zimag", zimag_col)
            zreal_ohm = convert_role_to_base(working[zreal_col].to_numpy(dtype=float), "zreal", zreal_unit)
            zimag_ohm = convert_role_to_base(working[zimag_col].to_numpy(dtype=float), "zimag", zimag_unit)
            if column_is_negative_imaginary(zimag_col):
                zimag_ohm = -zimag_ohm
            units = {"zreal": zreal_unit, "zimag": zimag_unit}
            columns = {"zreal": zreal_col, "zimag": zimag_col}
        elif zmod_col and phase_col:
            used_cols = [zmod_col, phase_col]
            if frequency_col:
                used_cols.append(frequency_col)
            working = dataset.dataframe[used_cols].dropna().copy()
            zmod_unit = resolve_role_unit(selected_units.get("zmod", "Auto"), dataset, "zmod", zmod_col)
            phase_unit = resolve_role_unit(selected_units.get("phase", "Auto"), dataset, "phase", phase_col)
            zmod_ohm = convert_role_to_base(working[zmod_col].to_numpy(dtype=float), "zmod", zmod_unit)
            phase_deg = convert_role_to_base(working[phase_col].to_numpy(dtype=float), "phase", phase_unit)
            phase_rad = np.deg2rad(phase_deg)
            zreal_ohm = zmod_ohm * np.cos(phase_rad)
            zimag_ohm = zmod_ohm * np.sin(phase_rad)
            units = {"zmod": zmod_unit, "phase": phase_unit}
            columns = {"zmod": zmod_col, "phase": phase_col}
            warnings.append("Calculated Z real and Z imaginary from |Z| and phase.")
        else:
            warnings.append("EIS requires Z real + Z imaginary, or |Z| + phase columns.")
            return None

        frequency_hz = None
        if frequency_col and frequency_col in working.columns:
            frequency_unit = resolve_role_unit(selected_units.get("frequency", "Auto"), dataset, "frequency", frequency_col)
            frequency_hz = convert_role_to_base(working[frequency_col].to_numpy(dtype=float), "frequency", frequency_unit)
            units["frequency"] = frequency_unit
            columns["frequency"] = frequency_col

        mask = finite_pair_mask(zreal_ohm, zimag_ohm)
        if frequency_hz is not None:
            mask &= np.isfinite(frequency_hz)
            frequency_hz = frequency_hz[mask]
        zreal_ohm = zreal_ohm[mask]
        zimag_ohm = zimag_ohm[mask]
        if len(zreal_ohm) < 3:
            warnings.append("Too few EIS rows for reliable plotting.")

        return {
            "experiment_type": experiment_type,
            "filename": dataset.filename,
            "sample_name": sample_name,
            "columns": columns,
            "units": units,
            "metadata": dataset.metadata,
            "warnings": warnings,
            "data": {
                "zreal_ohm": zreal_ohm,
                "zimag_ohm": zimag_ohm,
                "frequency_hz": frequency_hz,
            },
        }

    required_roles = required_roles_for_experiment(experiment_type)
    missing = [role for role in required_roles if not role_columns.get(role)]
    if missing:
        warnings.append("Missing required columns: " + ", ".join(role_label(role) for role in missing))
        return None

    used_cols = [role_columns[role] for role in required_roles if role_columns.get(role)]
    if len(set(used_cols)) != len(used_cols):
        warnings.append("Required roles must use different columns.")
        return None

    working = dataset.dataframe[used_cols].dropna().copy()
    if working.empty:
        warnings.append("Selected columns have no usable numeric rows.")
        return None

    normalized_data: dict[str, np.ndarray] = {}
    columns: dict[str, str] = {}
    units: dict[str, str] = {}
    for role in required_roles:
        column = role_columns[role]
        values = working[column].to_numpy(dtype=float)
        unit = resolve_role_unit_from_values(selected_units.get(role, "Auto"), dataset, role, column, values)
        normalized_data[role] = convert_role_to_base(values, role, unit)
        columns[role] = column
        units[role] = unit

    mask = finite_pair_mask(*normalized_data.values())
    normalized_data = {role: values[mask] for role, values in normalized_data.items()}
    if len(next(iter(normalized_data.values()))) < 5:
        warnings.append("Too few numeric rows for reliable analysis.")
    if experiment_type in {"cv", "lsv"}:
        warnings.extend(
            validate_cv_data(
                dataset.filename,
                normalized_data.get("potential", np.array([])),
                normalized_data.get("current", np.array([])),
            )
        )

    return {
        "experiment_type": experiment_type,
        "filename": dataset.filename,
        "sample_name": sample_name,
        "columns": columns,
        "units": units,
        "metadata": dataset.metadata,
        "warnings": warnings,
        "data": normalized_data,
    }


def summarize_generic_record(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data", {})
    summary = {
        "filename": record.get("filename"),
        "sample_name": record.get("sample_name"),
        "experiment_type": record.get("experiment_type"),
        "points": 0,
        "warnings": record.get("warnings", []),
    }

    for key, values in data.items():
        if values is None:
            continue
        array = np.asarray(values, dtype=float)
        summary["points"] = max(summary["points"], int(len(array)))
        finite = array[np.isfinite(array)]
        if finite.size:
            summary[f"{key}_min"] = float(np.min(finite))
            summary[f"{key}_max"] = float(np.max(finite))

    return summary


def serialize_generic_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": summarize_generic_record(record),
        "metadata": record.get("metadata", {}),
        "warnings": record.get("warnings", []),
        "columns": record.get("columns", {}),
        "units": record.get("units", {}),
        "parsed": (
            serialize_parsed_dataset(record["parsed_dataset"])
            if isinstance(record.get("parsed_dataset"), ParsedDataset)
            else None
        ),
        "trace": to_jsonable(record.get("data", {})),
    }


def store_generic_experiment_analysis(
    project_workspace: dict[str, Any],
    experiment_id: str,
    experiment_type: str,
    records: list[dict[str, Any]],
    parser_summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    settings: dict[str, Any],
) -> None:
    experiment = project_workspace.setdefault("experiments", {}).setdefault(
        experiment_id,
        {
            "name": "Untitled experiment",
            "created_at": now_timestamp(),
            "updated_at": now_timestamp(),
            "files": {},
            "analysis_outputs": {},
            "notes": "",
            "experiment_type": experiment_type,
        },
    )
    experiment["experiment_type"] = experiment_type
    files = experiment.setdefault("files", {})
    for record in records:
        files[record["filename"]] = serialize_generic_record(record)

    experiment["updated_at"] = now_timestamp()
    experiment["analysis_outputs"] = {
        "experiment_type": experiment_type,
        "parser_summary": dataframe_records(parser_summary_df),
        "results": dataframe_records(results_df),
        "settings": settings,
    }
    save_project_workspaces()


def build_generic_results(
    records: list[dict[str, Any]],
    experiment_type: str,
    lsv_threshold_a: Optional[float] = None,
    lsv_threshold_display: Optional[float] = None,
    lsv_threshold_unit: Optional[str] = None,
    lsv_current_source: str = "Current",
    lsv_electrode_area_cm2: float = 1.0,
    lsv_selected_current_unit: str = "Auto",
) -> pd.DataFrame:
    rows = []
    for record in records:
        data = record["data"]
        row: dict[str, Any] = {
            "sample_name": record["sample_name"],
            "file_name": record["filename"],
            "points": int(max(len(values) for values in data.values() if values is not None)),
        }

        if experiment_type == "ca":
            time_s = data["time"]
            current_a = data["current"]
            tail_start = max(0, int(len(current_a) * 0.9) - 1)
            row.update(
                {
                    "duration_s": float(np.nanmax(time_s) - np.nanmin(time_s)),
                    "initial_current_A": float(current_a[0]),
                    "final_current_A": float(current_a[-1]),
                    "steady_state_current_A": float(np.nanmean(current_a[tail_start:])),
                    "charge_C": float(np.trapezoid(current_a, time_s)) if len(time_s) > 1 else None,
                }
            )
        elif experiment_type == "cp":
            time_s = data["time"]
            potential_v = data["potential"]
            duration = float(np.nanmax(time_s) - np.nanmin(time_s)) if len(time_s) else 0.0
            row.update(
                {
                    "duration_s": duration,
                    "initial_potential_V": float(potential_v[0]),
                    "final_potential_V": float(potential_v[-1]),
                    "delta_potential_V": float(potential_v[-1] - potential_v[0]),
                    "average_slope_V_s": float((potential_v[-1] - potential_v[0]) / duration) if duration else None,
                }
            )
        elif experiment_type == "lsv":
            potential_v = data["potential"]
            current_values = data["current"]
            current_summary_key = "current_density_A_cm2" if lsv_current_source == "Current density" else "current_A"
            parsed_dataset = record.get("parsed_dataset")
            scan_rate_estimate = "Not available"
            reference_metadata = None
            electrode_area_metadata = None
            electrode_area_value = None
            unit_confidence = "Not available"
            unit_review_required = False
            current_col = record.get("columns", {}).get("current")
            if isinstance(parsed_dataset, ParsedDataset):
                potential_col = record.get("columns", {}).get("potential")
                scan_rate_estimate = scan_rate_estimate_summary(parsed_dataset, potential_col)
                reference_metadata = reference_electrode_from_metadata(parsed_dataset.metadata)
                electrode_area_metadata = electrode_area_metadata_summary(parsed_dataset.metadata)
                electrode_area_value = electrode_area_metadata_value(parsed_dataset.metadata)
                unit_confidence = current_unit_confidence(parsed_dataset, current_col, lsv_selected_current_unit)
                unit_review_required = unit_confidence == "Low"
            signal_kind = uploaded_signal_kind(parsed_dataset if isinstance(parsed_dataset, ParsedDataset) else None, current_col, lsv_current_source, lsv_selected_current_unit)
            normalization_applied = lsv_current_source == "Current" and bool(
                lsv_threshold_unit and is_current_density_unit(lsv_threshold_unit)
            )
            confirmed_unit = None
            confirmed_unit_type = None
            unit_confirmation_source = None
            if unit_confidence == "Confirmed by user":
                confirmed_unit = display_unit_label(
                    current_unit_label_for_summary(lsv_selected_current_unit, lsv_current_source)
                )
                confirmed_unit_type = confirmation_unit_type(lsv_current_source)
                unit_confirmation_source = "user"
            row.update(
                {
                    "potential_min_V": float(np.nanmin(potential_v)),
                    "potential_max_V": float(np.nanmax(potential_v)),
                    f"{current_summary_key}_min": float(np.nanmin(current_values)),
                    f"{current_summary_key}_max": float(np.nanmax(current_values)),
                    "scan_rate_estimate": scan_rate_estimate,
                    "reference_electrode_metadata": reference_metadata,
                    "electrode_area_metadata": electrode_area_metadata,
                    "electrode_area_cm2": electrode_area_value or lsv_electrode_area_cm2,
                    "unit_confidence": unit_confidence,
                    "unit_review_required": unit_review_required,
                    "y_axis_quantity": "Current density" if lsv_current_source == "Current density" else "Current",
                    "uploaded_signal_kind": signal_kind,
                    "normalization_applied": normalization_applied,
                    "confirmed_unit": confirmed_unit,
                    "confirmed_unit_type": confirmed_unit_type,
                    "unit_confirmation_source": unit_confirmation_source,
                }
            )
            if lsv_threshold_a is not None and lsv_threshold_a > 0:
                direction = lsv_analysis_direction(potential_v, current_values)
                onset_result = analyze_lsv_onset(potential_v, current_values, lsv_threshold_a, direction)
                threshold_magnitude, crossing_threshold = lsv_threshold_magnitude_and_crossing(
                    lsv_threshold_a,
                    lsv_threshold_unit or "display",
                    lsv_current_source,
                    lsv_electrode_area_cm2,
                    onset_result.onset_direction or direction,
                )
                row.update(
                    {
                        f"threshold_{lsv_threshold_unit or 'display'}": lsv_threshold_display,
                        "threshold_magnitude": threshold_magnitude,
                        "crossing_threshold": crossing_threshold,
                        "onset_direction": onset_result.onset_direction or direction,
                        "onset_potential_V": onset_result.onset_potential,
                        "onset_status": onset_result.status,
                        "onset_confidence": onset_result.confidence,
                        "onset_detection_method": onset_result.method,
                        "onset_min_consecutive_points": onset_result.sustained_points_required,
                        "onset_min_potential_window_V": onset_result.sustained_window_v,
                        "onset_smoothing_method": onset_result.smoothing_method,
                        "onset_smoothing_window": onset_result.smoothing_window,
                        "rejected_onset_potential_V": onset_result.rejected_potential,
                        "rejected_onset_reason": onset_result.rejected_reason,
                        "operational_ESW_V": None,
                    }
                )
        elif experiment_type == "eis":
            zreal = data["zreal_ohm"]
            zimag = data["zimag_ohm"]
            minus_zimag = -zimag
            row.update(
                {
                    "zreal_min_ohm": float(np.nanmin(zreal)),
                    "zreal_max_ohm": float(np.nanmax(zreal)),
                    "max_minus_zimag_ohm": float(np.nanmax(minus_zimag)),
                    "estimated_rs_ohm": float(zreal[int(np.nanargmin(np.abs(minus_zimag)))]),
                    "estimated_diameter_ohm": float(np.nanmax(zreal) - np.nanmin(zreal)),
                }
            )
            frequency = data.get("frequency_hz")
            if frequency is not None and len(frequency):
                row["frequency_min_Hz"] = float(np.nanmin(frequency))
                row["frequency_max_Hz"] = float(np.nanmax(frequency))

        rows.append(row)

    return pd.DataFrame(rows)


def add_generic_trace(
    fig: go.Figure,
    record: dict[str, Any],
    experiment_type: str,
    display_units: dict[str, str],
    show_markers: bool,
) -> None:
    mode = "lines+markers" if show_markers else "lines"
    data = record["data"]
    sample_name = record["sample_name"]

    if experiment_type == "ca":
        x_unit = display_units["time"]
        y_unit = display_units["current"]
        fig.add_trace(
            go.Scatter(
                x=convert_role_from_base(data["time"], "time", x_unit),
                y=convert_role_from_base(data["current"], "current", y_unit),
                mode=mode,
                name=sample_name,
                hovertemplate=f"Time: %{{x:.4g}} {x_unit}<br>Current: %{{y:.4g}} {y_unit}<extra></extra>",
            )
        )
    elif experiment_type == "cp":
        x_unit = display_units["time"]
        y_unit = display_units["potential"]
        fig.add_trace(
            go.Scatter(
                x=convert_role_from_base(data["time"], "time", x_unit),
                y=convert_role_from_base(data["potential"], "potential", y_unit),
                mode=mode,
                name=sample_name,
                hovertemplate=f"Time: %{{x:.4g}} {x_unit}<br>Potential: %{{y:.4g}} {y_unit}<extra></extra>",
            )
        )
    elif experiment_type == "lsv":
        potential_unit = display_units["potential"]
        current_density_unit = display_units["current"]
        electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
        current_source = display_units.get("current_source", "Current")

        def lsv_axis_values(axis_quantity: str) -> tuple[np.ndarray, str, str]:
            if axis_quantity == "Current density":
                return (
                    current_density_from_base(data["current"], current_density_unit, current_source, electrode_area_cm2),
                    "Current density",
                    current_density_unit,
                )
            if axis_quantity == "Current":
                return (
                    current_to_display(data["current"], current_density_unit, current_source, electrode_area_cm2),
                    "Current",
                    current_density_unit,
                )
            return (
                convert_role_from_base(data["potential"], "potential", potential_unit),
                "Potential",
                potential_unit,
            )

        x_values, x_label, x_unit = lsv_axis_values(display_units.get("x_axis", "Potential"))
        y_values, y_label, y_unit = lsv_axis_values(display_units.get("y_axis", "Current density"))
        if display_units.get("unit_review_required"):
            if x_label != "Potential":
                x_label = "Current density" if current_source == "Current density" else "Signal"
                x_unit = "unit unconfirmed"
            if y_label != "Potential":
                y_label = "Current density" if current_source == "Current density" else "Signal"
                y_unit = "unit unconfirmed"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode=mode,
                name=sample_name,
                hovertemplate=(
                    f"{x_label}: %{{x:.4g}} {x_unit}<br>"
                    f"{y_label}: %{{y:.4g}} {y_unit}<extra></extra>"
                ),
            )
        )
    elif experiment_type == "eis":
        z_unit = display_units["impedance"]
        fig.add_trace(
            go.Scatter(
                x=convert_role_from_base(data["zreal_ohm"], "zreal", z_unit),
                y=convert_role_from_base(-data["zimag_ohm"], "zimag", z_unit),
                mode=mode,
                name=sample_name,
                hovertemplate=f"Z real: %{{x:.4g}} {z_unit}<br>-Z imaginary: %{{y:.4g}} {z_unit}<extra></extra>",
            )
        )


def render_detected_metadata(metadata: dict[str, str], manual_metadata_key: str) -> None:
    detected_metadata = dict(metadata)
    st.markdown("**Detected metadata**")
    parsed_rows, unparsed_rows = metadata_rows_for_display(detected_metadata)

    if parsed_rows:
        st.dataframe(pd.DataFrame(parsed_rows), width="stretch", hide_index=True)
    elif unparsed_rows:
        st.info("No structured metadata fields were detected.")
    else:
        st.write("No metadata rows detected before the numeric table.")

    if unparsed_rows:
        with st.expander("Unparsed metadata lines", expanded=False):
            st.dataframe(pd.DataFrame(unparsed_rows), width="stretch", hide_index=True)

    st.markdown("**Manual file metadata**")
    manual_mode_key = f"{manual_metadata_key}::mode"
    manual_mode = st.radio(
        "Manual metadata mode",
        ["Add/complement detected metadata", "Override detected metadata with manual entries"],
        horizontal=False,
        key=manual_mode_key,
    )
    st.caption(
        "Use add/complement to fill missing metadata without replacing detected values. "
        "Use override when the file metadata is wrong and the manual entry should be used instead."
    )

    structured_manual: dict[str, str] = {}
    field_columns = st.columns(2)
    for idx, (field_id, label, placeholder) in enumerate(MANUAL_METADATA_FIELDS):
        with field_columns[idx % 2]:
            value = st.text_input(
                label,
                placeholder=placeholder,
                key=f"{manual_metadata_key}::{field_id}",
            )
        if str(value or "").strip():
            structured_manual[label] = str(value).strip()

    freeform_text = st.text_area(
        "Additional metadata lines",
        placeholder=(
            "Optional: add extra metadata as 'Field: value' lines, such as electrolyte, scan rate, "
            "instrument, reference electrode, or other file-specific metadata."
        ),
        key=manual_metadata_key,
    )
    freeform_manual = parse_manual_metadata_text(str(freeform_text or ""))
    effective_metadata = merge_manual_metadata(
        detected_metadata,
        structured_manual,
        freeform_manual,
        manual_mode,
    )
    metadata.clear()
    metadata.update(effective_metadata)

    manual_applied = bool(structured_manual or freeform_manual)
    if manual_applied:
        st.caption(
            "Manual metadata applied: "
            + ("matching detected fields are overridden." if "override" in manual_mode.lower() else "missing fields are complemented.")
        )
        effective_rows, _effective_unparsed = metadata_rows_for_display(effective_metadata)
        if effective_rows:
            st.markdown("**Metadata used by analysis**")
            st.dataframe(pd.DataFrame(effective_rows), width="stretch", hide_index=True)

    if effective_metadata:
        with st.expander("Advanced/debug raw metadata", expanded=False):
            st.json(effective_metadata)


def render_generic_experiment_analysis(
    project_name: str,
    project_workspace: dict[str, Any],
    experiment_id: str,
    experiment_type: str,
    parsed_datasets: list[ParsedDataset],
) -> None:
    experiment = project_workspace.get("experiments", {}).get(experiment_id, {})
    saved_analysis_outputs = experiment.get("analysis_outputs", {}) if isinstance(experiment, dict) else {}
    saved_analysis_settings = (
        saved_analysis_outputs.get("settings", {})
        if isinstance(saved_analysis_outputs, dict)
        else {}
    )
    is_lsv = experiment_type == "lsv"
    saved_display_units = saved_analysis_settings.get("display_units", {}) if isinstance(saved_analysis_settings, dict) else {}
    parser_display_current_unit = None
    parser_current_source = None
    parser_electrode_area = 1.0
    if is_lsv:
        saved_current_source = (
            saved_analysis_settings.get("lsv_current_source")
            or saved_display_units.get("current_source")
        )
        parser_electrode_area = float(saved_display_units.get("electrode_area_cm2") or 1.0)
        parser_display_current_unit, parser_current_source = readable_parser_current_unit_for_lsv(
            parsed_datasets,
            str(saved_current_source) if saved_current_source else None,
            parser_electrode_area,
        )

    parser_summary = pd.DataFrame()
    lsv_summary_container = st.container() if is_lsv else None
    lsv_plot_container = st.container() if is_lsv else None
    lsv_metrics_container = st.container() if is_lsv else None
    lsv_controls_container = st.container() if is_lsv else None
    lsv_mapping_container = st.container() if is_lsv else None
    lsv_details_container = st.container() if is_lsv else None
    lsv_download_container = st.container() if is_lsv else None
    lsv_controls_expander = (
        lsv_controls_container.expander("Display & threshold controls", expanded=False)
        if lsv_controls_container is not None
        else nullcontext()
    )
    mapping_context = (
        lsv_mapping_container.expander("Data Mapping & Units", expanded=False)
        if lsv_mapping_container is not None
        else nullcontext()
    )

    if not is_lsv:
        parser_summary = render_parser_summary(
            parsed_datasets,
            experiment_type,
            key_prefix=f"generic_parser::{project_name}::{experiment_id}",
            display_current_unit=parser_display_current_unit,
            current_source_override=parser_current_source,
            electrode_area_cm2=parser_electrode_area,
        )

    with mapping_context:
        selected_filename = st.selectbox(
            "Analyzing file",
            [dataset.filename for dataset in parsed_datasets],
            key=f"generic_preview::{project_name}::{experiment_id}",
        )
        selected_dataset = next(dataset for dataset in parsed_datasets if dataset.filename == selected_filename)
        selected_detected = detect_experiment_columns(selected_dataset, experiment_type)

        if not is_lsv:
            with st.expander("Detected metadata"):
                render_detected_metadata(
                    selected_dataset.metadata,
                    f"manual_metadata::{project_name}::{experiment_id}::{selected_dataset.filename}",
                )

        if is_lsv:
            st.markdown("**Column and unit controls**")
        else:
            st.subheader("Column and unit controls")
        column_mode = st.radio(
            "Column selection mode",
            ["Use detected columns per file", "Manually select columns"],
            horizontal=True,
            key=f"generic_column_mode::{project_name}::{experiment_id}",
        )

        manual_columns: dict[str, Optional[str]] = {}
        roles = required_roles_for_experiment(experiment_type) + optional_roles_for_experiment(experiment_type)
        column_count = 3 if experiment_type == "eis" else 2
        for idx, role in enumerate(roles):
            if idx % column_count == 0:
                cols = st.columns(column_count)
            options = selected_dataset.headers if role in required_roles_for_experiment(experiment_type) else ["None"] + selected_dataset.headers
            detected_column = selected_detected.get(role)
            if detected_column in options:
                default_index = options.index(detected_column)
            else:
                default_index = 0
            with cols[idx % column_count]:
                selected_column = st.selectbox(
                    f"{role_label(role)} column",
                    options,
                    index=default_index,
                    disabled=column_mode == "Use detected columns per file",
                    key=f"generic_col::{project_name}::{experiment_id}::{role}",
                )
            manual_columns[role] = None if selected_column == "None" else selected_column

        selected_units: dict[str, str] = {}
        unit_cols = st.columns(min(3, len(roles)))
        for idx, role in enumerate(roles):
            with unit_cols[idx % len(unit_cols)]:
                unit_options = unit_options_for_role(role)
                unit_index = 0
                unit_widget_key = f"generic_unit::{project_name}::{experiment_id}::{role}"
                detected_column_for_role = (
                    selected_detected.get(role)
                    if column_mode == "Use detected columns per file"
                    else manual_columns.get(role)
                )
                if is_lsv and role == "current":
                    current_source_key_for_unit = lsv_current_source_state_key(
                        project_name,
                        experiment_id,
                        selected_filename,
                        detected_column_for_role,
                    )
                    confirmed_source_for_unit = confirmed_unit_source_for_lsv(
                        selected_dataset,
                        detected_column_for_role,
                        current_source_key_for_unit,
                    )
                    confirmed_unit_key = (
                        f"lsv_confirmed_current_unit::{project_name}::{experiment_id}::"
                        f"{selected_dataset.filename}::{detected_column_for_role or role}"
                    )
                    confirmed_unit = st.session_state.get(confirmed_unit_key)
                    confirmed_input_unit = unit_input_from_confirmed_unit(
                        confirmed_unit,
                        confirmed_source_for_unit,
                    )
                    if confirmed_input_unit in unit_options:
                        unit_index = unit_options.index(confirmed_input_unit)
                        if st.session_state.get(unit_widget_key) != confirmed_input_unit:
                            st.session_state[unit_widget_key] = confirmed_input_unit
                selected_units[role] = st.selectbox(
                    f"{role_label(role)} input unit",
                    unit_options,
                    index=unit_index,
                    key=unit_widget_key,
                )

    if is_lsv and lsv_details_container is not None:
        with lsv_details_container.expander("Parser details", expanded=False):
            parser_summary = render_parser_summary(
                parsed_datasets,
                experiment_type,
                key_prefix=f"generic_parser::{project_name}::{experiment_id}",
                display_current_unit=parser_display_current_unit,
                current_source_override=parser_current_source,
                electrode_area_cm2=parser_electrode_area,
                current_input_unit_override=(
                    selected_units.get("current")
                    if selected_units.get("current") != "Auto"
                    else None
                ),
                heading="Parser details",
                show_heading=False,
            )
        with lsv_details_container.expander("Detected metadata", expanded=False):
            render_detected_metadata(
                selected_dataset.metadata,
                f"manual_metadata::{project_name}::{experiment_id}::{selected_dataset.filename}",
            )

    display_units: dict[str, str] = {}
    with lsv_controls_expander:
        if is_lsv:
            st.markdown("**Display controls**")
        else:
            st.subheader("Display controls")
        if experiment_type == "ca":
            col1, col2, col3 = st.columns(3)
            with col1:
                display_units["time"] = st.selectbox("Time display unit", display_unit_options_for_role("time"), key=f"disp_time::{project_name}::{experiment_id}")
            with col2:
                display_units["current"] = st.selectbox("Current display unit", display_unit_options_for_role("current"), key=f"disp_current::{project_name}::{experiment_id}")
            with col3:
                show_markers = st.checkbox("Show data markers", value=False, key=f"generic_markers::{project_name}::{experiment_id}")
        elif experiment_type == "cp":
            col1, col2, col3 = st.columns(3)
            with col1:
                display_units["time"] = st.selectbox("Time display unit", display_unit_options_for_role("time"), key=f"disp_time::{project_name}::{experiment_id}")
            with col2:
                display_units["potential"] = st.selectbox("Potential display unit", display_unit_options_for_role("potential"), key=f"disp_potential::{project_name}::{experiment_id}")
            with col3:
                show_markers = st.checkbox("Show data markers", value=False, key=f"generic_markers::{project_name}::{experiment_id}")
        elif experiment_type == "eis":
            col1, col2 = st.columns(2)
            with col1:
                display_units["impedance"] = st.selectbox("Impedance display unit", display_unit_options_for_role("impedance"), key=f"disp_impedance::{project_name}::{experiment_id}")
            with col2:
                show_markers = st.checkbox("Show data markers", value=True, key=f"generic_markers::{project_name}::{experiment_id}")
        elif experiment_type == "lsv":
            potential_col_for_units = selected_detected.get("potential") if column_mode == "Use detected columns per file" else manual_columns.get("potential")
            current_col_for_units = selected_detected.get("current") if column_mode == "Use detected columns per file" else manual_columns.get("current")
            potential_values_for_units = (
                selected_dataset.dataframe[potential_col_for_units].to_numpy(dtype=float)
                if potential_col_for_units in selected_dataset.dataframe.columns
                else np.array([])
            )
            potential_display_unit = resolve_role_unit_from_values(
                selected_units.get("potential", "Auto"),
                selected_dataset,
                "potential",
                potential_col_for_units,
                potential_values_for_units,
            )
            if potential_display_unit not in display_unit_options_for_role("potential"):
                potential_display_unit = "V"
            current_unit_for_density = resolve_role_unit(
                selected_units.get("current", "Auto"),
                selected_dataset,
                "current",
                current_col_for_units,
            )

            inferred_current_source = current_source_guess_for_dataset(selected_dataset, current_col_for_units)
            current_source_default = 1 if inferred_current_source == "Current density" else 0
            current_source_key = lsv_current_source_state_key(
                project_name,
                experiment_id,
                selected_filename,
                current_col_for_units,
            )
            if current_source_default == 1 and current_source_key not in st.session_state:
                st.session_state[current_source_key] = "Current density"
            display_units["current_source"] = st.selectbox(
                "Uploaded current column contains",
                ["Current", "Current density"],
                index=current_source_default,
                key=current_source_key,
            )
            axis_options, default_y_axis = lsv_axis_options_for_current_source(display_units["current_source"])
            y_axis_default_index = axis_options.index(default_y_axis)
            axis_key_scope = (
                f"{project_name}::{experiment_id}::{selected_filename}::"
                f"{display_units['current_source']}"
            )
            x_axis_key = f"lsv_x_axis::{axis_key_scope}"
            y_axis_key = f"lsv_y_axis::{axis_key_scope}"
            if st.session_state.get(x_axis_key) not in {None, *axis_options}:
                st.session_state[x_axis_key] = "Potential"
            if st.session_state.get(y_axis_key) not in {None, *axis_options}:
                st.session_state[y_axis_key] = default_y_axis
            col1, col2, col3 = st.columns(3)
            with col1:
                display_units["x_axis"] = st.selectbox(
                    "X-axis",
                    axis_options,
                    index=0,
                    key=x_axis_key,
                )
            with col2:
                display_units["y_axis"] = st.selectbox(
                    "Y-axis",
                    axis_options,
                    index=y_axis_default_index,
                    key=y_axis_key,
                )
            with col3:
                show_markers = st.checkbox("Show data markers", value=False, key=f"generic_markers::{project_name}::{experiment_id}")

            if display_units["x_axis"] == display_units["y_axis"]:
                st.warning("Choose different quantities for the X-axis and Y-axis.")

            display_units["potential"] = potential_display_unit
            uses_current_density_axis = "Current density" in {
                display_units.get("x_axis", "Potential"),
                display_units.get("y_axis", default_y_axis),
            }
            display_units["current"] = (
                current_density_unit_from_current_unit(current_unit_for_density)
                if uses_current_density_axis
                else current_unit_for_density
            )
            if display_units["current_source"] == "Current density":
                display_units["electrode_area_cm2"] = 1.0
                if current_col_for_units and header_indicates_current_density(current_col_for_units):
                    st.info("The uploaded current column is already normalized, so electrode-area normalization is skipped.")
                else:
                    st.info("Uploaded signal appears to be current density; electrode-area normalization was skipped.")
                detected_area_metadata = electrode_area_metadata_summary(selected_dataset.metadata)
                if detected_area_metadata:
                    st.info(
                        f"Electrode area metadata: {detected_area_metadata}. "
                        "No additional normalization applied."
                    )
            elif uses_current_density_axis:
                area_key = f"lsv_electrode_area::{project_name}::{experiment_id}"
                metadata_area_value = electrode_area_metadata_value(selected_dataset.metadata)
                if metadata_area_value is not None and area_key not in st.session_state:
                    st.session_state[area_key] = metadata_area_value
                display_units["electrode_area_cm2"] = st.number_input(
                    "Electrode surface area (cm\u00b2)",
                    min_value=0.000001,
                    value=metadata_area_value or 1.0,
                    step=0.1,
                    format="%.6f",
                    key=area_key,
                )
                area_text = (
                    f"{float(display_units['electrode_area_cm2']):.1f}"
                    if float(display_units["electrode_area_cm2"]) == round(float(display_units["electrode_area_cm2"]))
                    else f"{float(display_units['electrode_area_cm2']):g}"
                )
                st.info(
                    "Current density calculated from raw current using electrode area = "
                    f"{area_text} cm\u00b2. Enter the actual electrode area for meaningful "
                    "current-density values."
                )
            else:
                display_units["electrode_area_cm2"] = 1.0
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                display_units["potential"] = st.selectbox("Potential display unit", display_unit_options_for_role("potential"), key=f"disp_potential::{project_name}::{experiment_id}")
            with col2:
                display_units["current"] = st.selectbox("Current display unit", display_unit_options_for_role("current"), key=f"disp_current::{project_name}::{experiment_id}")
            with col3:
                show_markers = st.checkbox("Show data markers", value=False, key=f"generic_markers::{project_name}::{experiment_id}")

    records: list[dict[str, Any]] = []
    for dataset in parsed_datasets:
        role_columns = detect_experiment_columns(dataset, experiment_type) if column_mode == "Use detected columns per file" else manual_columns
        record = normalize_generic_dataset(dataset, experiment_type, role_columns, selected_units)
        if record is None:
            st.warning(f"Skipping {dataset.filename}: required columns could not be normalized for {experiment_short_label(experiment_type)}.")
            continue
        record["parsed_dataset"] = dataset
        records.append(record)

    if not records:
        st.error("No uploaded files could be analyzed with the selected experiment type and column settings.")
        st.stop()
        return

    for record in records:
        for warning in record.get("warnings", []):
            st.warning(f"{record['filename']}: {warning}")

    if experiment_type == "lsv":
        electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
        current_source = display_units.get("current_source", "Current")
        selected_dataset_for_review = next(
            (dataset for dataset in parsed_datasets if dataset.filename == selected_filename),
            selected_dataset,
        )
        selected_current_col_for_review = (
            detect_experiment_columns(selected_dataset_for_review, "lsv").get("current")
        )
        display_units["unit_review_required"] = current_unit_is_ambiguous(
            selected_dataset_for_review,
            selected_current_col_for_review,
            selected_units.get("current", "Auto"),
        )
        display_units.pop("suggested_current_unit", None)
        if (
            display_units["unit_review_required"]
            and selected_current_col_for_review
            and selected_current_col_for_review in selected_dataset_for_review.dataframe
        ):
            suggested_input_unit = suggested_current_unit_for_context(
                selected_dataset_for_review.dataframe[selected_current_col_for_review].to_numpy(dtype=float),
                current_source,
            )
            display_units["suggested_current_unit"] = current_unit_label_for_summary(
                suggested_input_unit,
                current_source,
            )
        display_units["current"] = lsv_display_unit_for_records(
            records,
            current_source,
            electrode_area_cm2,
            prefer_density="Current density" in {
                display_units.get("x_axis", "Potential"),
                display_units.get("y_axis", "Current density"),
            },
        )

    lsv_threshold_display = None
    lsv_threshold_a = None
    lsv_threshold_unit = None
    lsv_threshold_source = "Default"
    lsv_threshold_rule = "Default threshold."
    reference_electrode = None
    reference_electrode_source = "default"
    selected_lsv_summary_for_interpretation: Optional[dict[str, str]] = None
    if experiment_type == "lsv":
        y_unit = display_units["current"]
        electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
        current_source = display_units.get("current_source", "Current")
        selected_record_for_threshold = next(
            (record for record in records if record["filename"] == selected_filename),
            records[0],
        )
        selected_threshold_direction = lsv_dominant_response_direction(
            np.asarray(
                selected_record_for_threshold["data"].get("current", np.array([])),
                dtype=float,
            )
        ) or lsv_scan_direction(
            np.asarray(
                selected_record_for_threshold["data"].get("potential", np.array([])),
                dtype=float,
            )
        )
        all_lsv_y = np.concatenate(
            [
                np.abs(current_to_display(record["data"]["current"], y_unit, current_source, electrode_area_cm2))
                for record in records
            ]
        )
        finite_lsv_y = all_lsv_y[np.isfinite(all_lsv_y)]
        if finite_lsv_y.size and float(np.nanmax(finite_lsv_y)) > 0:
            default_threshold = float(np.nanmax(finite_lsv_y) * 0.1)
            default_threshold_source = "Auto-selected"
        else:
            default_threshold = 0.5
            default_threshold_source = "Default"
        lsv_threshold_unit = y_unit
        with lsv_controls_expander:
            st.markdown("**Threshold and reference**")
            threshold_preset_options = lsv_threshold_preset_options(y_unit)
            threshold_preset = st.selectbox(
                "Threshold preset",
                threshold_preset_options,
                index=0,
                key=f"lsv_threshold_preset::{project_name}::{experiment_id}::{y_unit}",
            )
            recommended_threshold, lsv_threshold_source, lsv_threshold_rule = lsv_threshold_recommendation(
                threshold_preset,
                y_unit,
                current_source,
                electrode_area_cm2,
                default_threshold,
                default_threshold_source,
                selected_threshold_direction,
            )
            lsv_threshold_display = st.number_input(
                f"{lsv_threshold_quantity_label(y_unit).title()} threshold ({display_unit_label(y_unit)})",
                min_value=0.0,
                value=recommended_threshold,
                step=max(recommended_threshold * 0.1, 1e-9),
                format="%.6e",
                key=f"lsv_threshold::{project_name}::{experiment_id}::{y_unit}::{threshold_preset}",
            )
            if not threshold_values_match(lsv_threshold_display, recommended_threshold):
                lsv_threshold_source, lsv_threshold_rule = lsv_threshold_source_details(
                    "User-defined",
                    lsv_threshold_quantity_label(y_unit),
                    selected_threshold_direction,
                )
            st.caption(lsv_threshold_rule)
        lsv_threshold_a = current_display_value_to_base(lsv_threshold_display, y_unit, current_source, electrode_area_cm2)
        selected_reference_from_metadata = reference_electrode_from_metadata(selected_dataset.metadata)
        reference_options = list(REFERENCE_ELECTRODE_OPTIONS)
        if selected_reference_from_metadata and selected_reference_from_metadata not in reference_options:
            reference_options.insert(-1, selected_reference_from_metadata)
        saved_reference_electrode = str(saved_analysis_settings.get("reference_electrode") or "").strip()
        if saved_reference_electrode and saved_reference_electrode not in reference_options:
            reference_options.append(saved_reference_electrode)
        if saved_reference_electrode and (
            saved_reference_electrode != DEFAULT_REFERENCE_ELECTRODE or not selected_reference_from_metadata
        ):
            default_reference_value = saved_reference_electrode
        elif selected_reference_from_metadata:
            default_reference_value = selected_reference_from_metadata
        else:
            default_reference_value = DEFAULT_REFERENCE_ELECTRODE
        reference_index = reference_options.index(default_reference_value) if default_reference_value in reference_options else 0
        with lsv_controls_expander:
            reference_electrode = st.selectbox(
                "Reference electrode",
                reference_options,
                index=reference_index,
                key=f"lsv_reference_electrode::{project_name}::{experiment_id}::{selected_filename}",
            )
            if reference_electrode == "Other":
                custom_reference = st.text_input(
                    "Reference electrode name",
                    key=f"lsv_reference_electrode_custom::{project_name}::{experiment_id}::{selected_filename}",
                ).strip()
                reference_electrode = custom_reference or "Other"
            if selected_reference_from_metadata and reference_electrode == selected_reference_from_metadata:
                reference_electrode_source = "metadata"
            elif reference_electrode == DEFAULT_REFERENCE_ELECTRODE and not selected_reference_from_metadata:
                reference_electrode_source = "default"
            else:
                reference_electrode_source = "user"

        if lsv_summary_container is not None:
            selected_record = next(
                (record for record in records if record["filename"] == selected_filename),
                records[0],
            )
            selected_parsed_dataset = selected_record.get("parsed_dataset")
            with lsv_summary_container:
                selected_metadata_area = None
                if isinstance(selected_parsed_dataset, ParsedDataset):
                    selected_metadata_area = electrode_area_metadata_summary(selected_parsed_dataset.metadata)
                selected_current_col = selected_record.get("columns", {}).get("current")
                selected_current_unit_review_needed = (
                    isinstance(selected_parsed_dataset, ParsedDataset)
                    and current_unit_is_ambiguous(
                        selected_parsed_dataset,
                        selected_current_col,
                        selected_units.get("current", "Auto"),
                    )
                )
                if (
                    selected_current_unit_review_needed
                    and isinstance(selected_parsed_dataset, ParsedDataset)
                    and selected_current_col in selected_parsed_dataset.dataframe
                ):
                    selected_current_values = selected_parsed_dataset.dataframe[selected_current_col].to_numpy(dtype=float)
                    selected_suggested_unit = suggested_current_unit_for_context(
                        selected_current_values,
                        display_units.get("current_source", "Current"),
                    )
                    selected_unit_review_warning = current_unit_magnitude_warning(
                        selected_parsed_dataset,
                        selected_current_col,
                        selected_units.get("current", "Auto"),
                        "A",
                        display_units.get("current_source", "Current"),
                        float(display_units.get("electrode_area_cm2", 1.0)),
                    )
                    render_unit_review_panel(
                        current_col=selected_current_col,
                        current_col_label=user_facing_column_label(selected_parsed_dataset, selected_current_col),
                        suggested_unit=selected_suggested_unit,
                        current_source=display_units.get("current_source", "Current"),
                        current_unit_warning=selected_unit_review_warning,
                        confirmed_current_unit_key=(
                            f"lsv_confirmed_current_unit::{project_name}::{experiment_id}::"
                            f"{selected_parsed_dataset.filename}::{selected_current_col}"
                        ),
                        confirmed_unit_type_key=(
                            f"lsv_confirmed_current_unit_type::{project_name}::{experiment_id}::"
                            f"{selected_parsed_dataset.filename}::{selected_current_col}"
                        ),
                        unit_confirmation_source_key=(
                            f"lsv_unit_confirmation_source::{project_name}::{experiment_id}::"
                            f"{selected_parsed_dataset.filename}::{selected_current_col}"
                        ),
                        selectbox_key=(
                            f"lsv_confirm_current_unit_choice::{project_name}::{experiment_id}::"
                            f"{selected_parsed_dataset.filename}::{selected_current_col}"
                        ),
                        button_key=(
                            f"lsv_apply_current_unit::{project_name}::{experiment_id}::"
                            f"{selected_parsed_dataset.filename}::{selected_current_col}"
                        ),
                    )
                selected_current_magnitude_review_needed = False
                if isinstance(selected_parsed_dataset, ParsedDataset):
                    selected_details = parser_summary_details(
                        selected_parsed_dataset,
                        "lsv",
                        display_units.get("current"),
                        display_units.get("current_source"),
                        float(display_units.get("electrode_area_cm2", 1.0)),
                        selected_units.get("current") if selected_units.get("current") != "Auto" else None,
                    )
                    selected_current_magnitude_review_needed = bool(
                        selected_details.get("current_magnitude_warning")
                    ) and not bool(
                        st.session_state.get(
                            current_magnitude_ack_key(
                                f"generic_parser::{project_name}::{experiment_id}",
                                selected_parsed_dataset.filename,
                            )
                        )
                    )
                selected_lsv_summary_for_interpretation = build_lsv_analysis_summary(
                    selected_record,
                    selected_parsed_dataset if isinstance(selected_parsed_dataset, ParsedDataset) else selected_dataset,
                    lsv_threshold_a,
                    display_units,
                    reference_electrode,
                    lsv_threshold_source,
                    lsv_threshold_rule,
                    reference_electrode_source,
                    selected_current_unit_review_needed or selected_current_magnitude_review_needed,
                )
                st.markdown(
                    f"<div class=\"analysis-file\">{escape_html(format_lsv_display_state(display_units, lsv_threshold_a, reference_electrode, selected_metadata_area, bool(display_units.get('unit_review_required')), selected_lsv_summary_for_interpretation.get('scan_rate'), selected_lsv_summary_for_interpretation.get('normalization_status')))}</div>",
                    unsafe_allow_html=True,
                )
                lsv_quality_items, lsv_quality_review = lsv_data_quality_items(
                    selected_lsv_summary_for_interpretation,
                    selected_current_unit_review_needed,
                )
                render_data_quality_strip(lsv_quality_items, lsv_quality_review)
                render_lsv_analysis_summary(selected_lsv_summary_for_interpretation)
                render_metadata_status_notice(selected_lsv_summary_for_interpretation)
                render_recommended_actions(
                    [
                        str(selected_lsv_summary_for_interpretation.get("review_note") or ""),
                        str(selected_lsv_summary_for_interpretation.get("scan_rate_note") or ""),
                    ],
                    key_prefix=f"lsv_actions::{project_name}::{experiment_id}",
                )

    plot_context = lsv_plot_container if lsv_plot_container is not None else nullcontext()
    with plot_context:
        st.subheader(f"Interactive {experiment_short_label(experiment_type)} plot")
        if experiment_type == "lsv":
            lsv_plot_quantity = (
                "Current density"
                if "Current density"
                in {display_units.get("x_axis", "Potential"), display_units.get("y_axis", "Current density")}
                else "Current"
            )
            reference_caption = f", reported vs {reference_electrode}" if reference_electrode else ""
            review_caption = " \u00b7 unit confirmation required" if display_units.get("unit_review_required") else ""
            st.caption(f"{lsv_plot_quantity} vs potential{reference_caption}{review_caption}")
        fig = go.Figure()
        for record in records:
            add_generic_trace(fig, record, experiment_type, display_units, show_markers)

        if experiment_type == "lsv" and lsv_threshold_display is not None and lsv_threshold_display > 0:
            positive_threshold_label, negative_threshold_label = lsv_threshold_plot_labels(
                lsv_threshold_display,
                lsv_threshold_unit,
                selected_threshold_direction,
                bool(display_units.get("unit_review_required")),
            )
            if display_units.get("y_axis") in {"Current", "Current density"}:
                fig.add_hline(
                    y=lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=positive_threshold_label,
                )
                fig.add_hline(
                    y=-lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=negative_threshold_label,
                )
            elif display_units.get("x_axis") in {"Current", "Current density"}:
                fig.add_vline(
                    x=lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=positive_threshold_label,
                )
                fig.add_vline(
                    x=-lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=negative_threshold_label,
                )

            selected_lsv_record = next(
                (record for record in records if record["filename"] == selected_filename),
                records[0],
            )
            add_lsv_onset_annotation_to_plot(
                fig,
                selected_lsv_record,
                lsv_threshold_a,
                display_units,
                reference_electrode,
            )

        if experiment_type == "ca":
            fig.update_layout(xaxis_title=f"Time / {display_units['time']}", yaxis_title=f"Current / {display_units['current']}")
        elif experiment_type == "cp":
            fig.update_layout(xaxis_title=f"Time / {display_units['time']}", yaxis_title=f"Potential / {display_units['potential']}")
        elif experiment_type == "eis":
            fig.update_layout(xaxis_title=f"Z real / {display_units['impedance']}", yaxis_title=f"-Z imaginary / {display_units['impedance']}")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        elif experiment_type == "lsv":
            fig.update_layout(
                xaxis_title=lsv_axis_title(display_units.get("x_axis", "Potential"), display_units, reference_electrode),
                yaxis_title=lsv_axis_title(display_units.get("y_axis", "Current density"), display_units, reference_electrode),
            )
        else:
            fig.update_layout(xaxis_title=f"Potential / {display_units['potential']}", yaxis_title=f"Current / {display_units['current']}")

        fig.update_layout(
            template="plotly_white",
            height=620,
            margin={"l": 78, "r": 24, "t": 36, "b": 88},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "title_text": "",
            },
        )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, width="stretch")

        if experiment_type == "eis" and any(record["data"].get("frequency_hz") is not None for record in records):
            bode_mag = go.Figure()
            bode_phase = go.Figure()
            z_unit = display_units["impedance"]
            for record in records:
                frequency = record["data"].get("frequency_hz")
                if frequency is None or not len(frequency):
                    continue
                zreal = record["data"]["zreal_ohm"]
                zimag = record["data"]["zimag_ohm"]
                zmod = np.sqrt(zreal**2 + zimag**2)
                phase = np.rad2deg(np.arctan2(zimag, zreal))
                bode_mag.add_trace(
                    go.Scatter(
                        x=frequency,
                        y=convert_role_from_base(zmod, "zmod", z_unit),
                        mode="lines+markers" if show_markers else "lines",
                        name=record["sample_name"],
                    )
                )
                bode_phase.add_trace(
                    go.Scatter(
                        x=frequency,
                        y=phase,
                        mode="lines+markers" if show_markers else "lines",
                        name=record["sample_name"],
                    )
                )
            if bode_mag.data:
                bode_tabs = st.tabs(["Bode magnitude", "Bode phase"])
                with bode_tabs[0]:
                    bode_mag.update_layout(
                        xaxis_title="Frequency / Hz",
                        yaxis_title=f"|Z| / {z_unit}",
                        xaxis_type="log",
                        template="plotly_white",
                        height=430,
                    )
                    st.plotly_chart(bode_mag, width="stretch")
                with bode_tabs[1]:
                    bode_phase.update_layout(
                        xaxis_title="Frequency / Hz",
                        yaxis_title="Phase / deg",
                        xaxis_type="log",
                        template="plotly_white",
                        height=430,
                    )
                    st.plotly_chart(bode_phase, width="stretch")

        plot_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        st.download_button(
            "Download interactive plot as HTML",
            data=plot_html.encode("utf-8"),
            file_name=f"voltscope_{experiment_type}_plot.html",
            mime="text/html",
            key=f"download_generic_plot::{project_name}::{experiment_id}::{experiment_type}",
            on_click="ignore",
        )
        if is_lsv:
            lsv_quick_interpretation = lsv_interpretation_text(selected_lsv_summary_for_interpretation)
            selected_lsv_warnings = []
            if selected_lsv_summary_for_interpretation:
                selected_lsv_warnings = compact_text_list(
                    [
                        selected_lsv_summary_for_interpretation.get("review_note", ""),
                        selected_lsv_summary_for_interpretation.get("scan_rate_note", ""),
                    ]
                )
            render_interpretation_panel(
                lsv_quick_interpretation,
                key=f"lsv_ai_interpretation_placeholder::{project_name}::{experiment_id}",
                ai_payload=build_lsv_ai_interpretation_payload(
                    filename=selected_filename,
                    quick_interpretation=lsv_quick_interpretation,
                    summary=selected_lsv_summary_for_interpretation,
                    warnings=selected_lsv_warnings,
                    notes=str(project_workspace.get("experiments", {}).get(experiment_id, {}).get("notes", "")),
                ),
                project_workspace=project_workspace,
                experiment_id=experiment_id,
                interpretation_id=f"lsv::{selected_filename}",
            )

    results_df = build_generic_results(
        records,
        experiment_type,
        lsv_threshold_a=lsv_threshold_a,
        lsv_threshold_display=lsv_threshold_display,
        lsv_threshold_unit=lsv_threshold_unit,
        lsv_current_source=display_units.get("current_source", "Current"),
        lsv_electrode_area_cm2=float(display_units.get("electrode_area_cm2", 1.0)),
        lsv_selected_current_unit=selected_units.get("current", "Auto"),
    )
    metrics_context = lsv_metrics_container if lsv_metrics_container is not None else nullcontext()
    selected_metrics_df = None
    with metrics_context:
        st.subheader("Onset audit" if is_lsv else "Analysis results")
        if is_lsv:
            selected_lsv_record = next(
                (record for record in records if record["filename"] == selected_filename),
                records[0],
            )
            selected_parsed_dataset = selected_lsv_record.get("parsed_dataset")
            selected_metrics_df = build_lsv_detailed_metrics(
                selected_lsv_record,
                selected_parsed_dataset if isinstance(selected_parsed_dataset, ParsedDataset) else selected_dataset,
                lsv_threshold_a,
                display_units,
                reference_electrode,
                lsv_threshold_source,
                lsv_threshold_rule,
            )
            visible_lsv_metric_names = {
                "Anodic onset potential",
                "Cathodic onset potential",
                "Onset potential",
                "Onset method",
                "Threshold",
                "Threshold magnitude",
                "Cathodic crossing threshold",
                "Onset reliability",
                "Direction",
            }
            visible_lsv_metrics_df = selected_metrics_df[
                selected_metrics_df["Metric"].isin(visible_lsv_metric_names)
            ]
            if visible_lsv_metrics_df.empty:
                visible_lsv_metrics_df = selected_metrics_df.head(6)
            render_readable_table(visible_lsv_metrics_df)
            with st.expander("Metric calculation details", expanded=False):
                render_readable_table(selected_metrics_df)
        else:
            st.dataframe(results_df, width="stretch", hide_index=True)

        if not is_lsv:
            export_buffer = io.StringIO()
            export_buffer.write("# Parser summary\n")
            export_buffer.write(parser_summary.to_csv(index=False))
            export_buffer.write("\n# Raw analysis results\n")
            export_buffer.write(results_df.to_csv(index=False))
            st.download_button(
                "Download analysis results as CSV",
                data=export_buffer.getvalue().encode("utf-8"),
                file_name=f"voltscope_{experiment_type}_analysis_results.csv",
                mime="text/csv",
                key=f"download_generic_analysis::{project_name}::{experiment_id}::{experiment_type}",
                on_click="ignore",
            )

    if is_lsv and lsv_download_container is not None:
        with lsv_download_container:
            with st.expander("Raw/internal results", expanded=False):
                st.dataframe(results_df, width="stretch", hide_index=True)
            export_buffer = io.StringIO()
            export_buffer.write("# Parser summary\n")
            export_buffer.write(parser_summary.to_csv(index=False))
            if selected_metrics_df is not None:
                export_buffer.write("\n# Detailed LSV metrics\n")
                export_buffer.write(selected_metrics_df.to_csv(index=False))
            export_buffer.write("\n# Raw analysis results\n")
            export_buffer.write(results_df.to_csv(index=False))
            st.download_button(
                "Download analysis results as CSV",
                data=export_buffer.getvalue().encode("utf-8"),
                file_name=f"voltscope_{experiment_type}_analysis_results.csv",
                mime="text/csv",
                key=f"download_generic_analysis::{project_name}::{experiment_id}::{experiment_type}",
                on_click="ignore",
            )

    store_generic_experiment_analysis(
        project_workspace,
        experiment_id,
        experiment_type,
        records,
        parser_summary,
        results_df,
        {
            "column_mode": column_mode,
            "selected_units": selected_units,
            "display_units": display_units,
            "lsv_threshold": lsv_threshold_display,
            "lsv_threshold_unit": lsv_threshold_unit,
            "lsv_threshold_source": lsv_threshold_source,
            "lsv_threshold_rule": lsv_threshold_rule,
            "lsv_current_source": display_units.get("current_source", "Current"),
            "reference_electrode": reference_electrode,
            "reference_electrode_source": reference_electrode_source,
        },
    )
    render_experiment_notes(project_name, project_workspace, experiment_id)


# ---------- Streamlit UI ----------


def now_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_timestamp(timestamp: Optional[str]) -> str:
    if not timestamp:
        return ""
    try:
        return datetime.fromisoformat(timestamp).strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return timestamp


def escape_html(value: Any) -> str:
    return html.escape(str(value), quote=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    return to_jsonable(dataframe.to_dict(orient="records"))


def serialize_parsed_dataset(dataset: ParsedDataset) -> dict[str, Any]:
    cleanup_notice = dropped_rows_cleanup_notice(len(dataset.dataframe), dataset.rows_dropped)
    return {
        "filename": dataset.filename,
        "dataframe": dataframe_records(dataset.dataframe),
        "headers": dataset.headers,
        "metadata": dataset.metadata,
        "delimiter": dataset.delimiter,
        "has_header": dataset.has_header,
        "header_row": dataset.header_row,
        "data_start_row": dataset.data_start_row,
        "rows_skipped": dataset.rows_skipped,
        "rows_dropped": dataset.rows_dropped,
        "rows_dropped_reason": dropped_rows_reason(dataset.rows_dropped),
        "rows_dropped_fraction": dropped_rows_fraction_label(len(dataset.dataframe), dataset.rows_dropped),
        "parser_cleanup_notice": cleanup_notice,
        "parser_cleanup_severity": dropped_rows_severity(len(dataset.dataframe), dataset.rows_dropped),
        "missing_values": dataset.missing_values,
        "detected_potential_col": dataset.detected_potential_col,
        "detected_current_col": dataset.detected_current_col,
        "detected_time_col": dataset.detected_time_col,
        "detected_units": dataset.detected_units,
        "warnings": dataset.warnings,
    }


def parsed_dataset_from_payload(filename: str, payload: dict[str, Any]) -> Optional[ParsedDataset]:
    rows = payload.get("dataframe")
    if not isinstance(rows, list) or not rows:
        return None

    dataframe = pd.DataFrame(rows)
    headers = payload.get("headers") or list(dataframe.columns)
    headers = [str(header) for header in headers if str(header) in dataframe.columns]
    if headers:
        dataframe = dataframe[headers]
    else:
        headers = list(dataframe.columns)

    for column in dataframe.columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    delimiter = str(payload.get("delimiter") or delimiter_fallback_from_filename(filename))
    if delimiter in {"saved", "saved trace"}:
        delimiter = delimiter_fallback_from_filename(filename)

    return ParsedDataset(
        filename=str(payload.get("filename") or filename),
        dataframe=dataframe,
        headers=list(dataframe.columns),
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
        delimiter=delimiter,
        has_header=bool(payload.get("has_header", True)),
        header_row=payload.get("header_row"),
        data_start_row=int(payload.get("data_start_row", 1) or 1),
        rows_skipped=int(payload.get("rows_skipped", 0) or 0),
        rows_dropped=int(payload.get("rows_dropped", 0) or 0),
        missing_values=int(payload.get("missing_values", int(dataframe.isna().sum().sum())) or 0),
        detected_potential_col=payload.get("detected_potential_col"),
        detected_current_col=payload.get("detected_current_col"),
        detected_time_col=payload.get("detected_time_col"),
        detected_units=payload.get("detected_units", {}) if isinstance(payload.get("detected_units"), dict) else {},
        warnings=payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else [],
    )


def parsed_dataset_from_saved_file(
    filename: str,
    file_record: dict[str, Any],
    experiment_type: str,
    saved_settings: Optional[dict[str, Any]] = None,
) -> Optional[ParsedDataset]:
    parsed_payload = file_record.get("parsed")
    if isinstance(parsed_payload, dict):
        parsed = parsed_dataset_from_payload(filename, parsed_payload)
        if parsed is not None:
            return parsed

    trace = file_record.get("trace", {})
    if not isinstance(trace, dict):
        return None

    settings = saved_settings or {}
    display_units = settings.get("display_units", {})
    if not isinstance(display_units, dict):
        display_units = {}
    current_source = (
        settings.get("current_source")
        or settings.get("lsv_current_source")
        or display_units.get("current_source")
        or "Current"
    )

    if experiment_type in {"cv", "lsv"}:
        potential = trace.get("potential_v") or trace.get("potential") or []
        current = trace.get("raw_current_a") or trace.get("current") or []
        if not potential or not current:
            return None
        current_col = "CurrentDensity_A_cm2" if current_source == "Current density" else "Current_A"
        dataframe = pd.DataFrame({"Potential_V": potential, current_col: current})
        cycle_values = trace.get("cycle_values")
        if isinstance(cycle_values, list) and len(cycle_values) == len(dataframe):
            dataframe["Cycle"] = pd.to_numeric(pd.Series(cycle_values), errors="coerce")
        headers = list(dataframe.columns)
        detected_units = {"potential:Potential_V": "V", f"current:{current_col}": "A"}
        return ParsedDataset(
            filename=filename,
            dataframe=dataframe,
            headers=headers,
            metadata=file_record.get("metadata", {}) if isinstance(file_record.get("metadata"), dict) else {},
            delimiter=delimiter_fallback_from_filename(filename),
            has_header=True,
            header_row=1,
            data_start_row=2,
            rows_skipped=0,
            rows_dropped=0,
            missing_values=int(dataframe.isna().sum().sum()),
            detected_potential_col="Potential_V",
            detected_current_col=current_col,
            detected_time_col=None,
            detected_units=detected_units,
            warnings=file_record.get("warnings", []) if isinstance(file_record.get("warnings"), list) else [],
        )

    return None


def restore_parsed_datasets_from_experiment(experiment: dict[str, Any]) -> list[ParsedDataset]:
    outputs = experiment.get("analysis_outputs", {})
    saved_settings = outputs.get("settings", {}) if isinstance(outputs.get("settings"), dict) else {}
    experiment_type = outputs.get("experiment_type") or experiment.get("experiment_type", "cv")
    restored = []
    for filename, file_record in experiment.get("files", {}).items():
        if not isinstance(file_record, dict):
            continue
        parsed = parsed_dataset_from_saved_file(filename, file_record, experiment_type, saved_settings)
        if parsed is not None:
            restored.append(parsed)
    return restored


def empty_project() -> dict[str, Any]:
    timestamp = now_timestamp()
    return {"created_at": timestamp, "last_opened_at": timestamp, "experiments": {}}


def normalize_project_workspace(project_name: str, workspace: Any) -> dict[str, Any]:
    if not isinstance(workspace, dict):
        return empty_project()

    if "experiments" in workspace:
        workspace.setdefault("created_at", now_timestamp())
        workspace.setdefault("last_opened_at", workspace["created_at"])
        if not isinstance(workspace.get("experiments"), dict):
            workspace["experiments"] = {}
        for experiment in workspace["experiments"].values():
            if not isinstance(experiment, dict):
                continue
            experiment.setdefault("created_at", now_timestamp())
            experiment.setdefault("updated_at", experiment["created_at"])
            experiment.setdefault("files", {})
            experiment.setdefault("analysis_outputs", {})
            experiment.setdefault("notes", "")
            if "experiment_type" not in experiment:
                experiment["experiment_type"] = "cv" if experiment.get("files") else "auto"
        return workspace

    timestamp = now_timestamp()
    project = {"created_at": timestamp, "last_opened_at": timestamp, "experiments": {}}
    summaries = workspace.get("analysis_summaries", {})
    notes = workspace.get("notes", {})
    legacy_filenames = sorted(set(summaries.keys()) | set(notes.keys()))

    if legacy_filenames:
        imported_notes = []
        for filename in legacy_filenames:
            if notes.get(filename):
                imported_notes.append(f"## {filename}\n{notes[filename]}")
        project["experiments"]["imported-files"] = {
            "name": "Imported files",
            "created_at": now_timestamp(),
            "updated_at": now_timestamp(),
            "files": {
                filename: {"summary": summaries.get(filename, {}), "trace": {}, "metadata": {}}
                for filename in legacy_filenames
            },
            "analysis_outputs": {},
            "notes": "\n\n".join(imported_notes),
            "experiment_type": "cv",
        }

    return project


def load_project_workspaces() -> dict[str, dict[str, Any]]:
    if not WORKSPACE_STORE_PATH.exists():
        return {}

    try:
        with WORKSPACE_STORE_PATH.open("r", encoding="utf-8") as file:
            stored = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    return {
        project_name: normalize_project_workspace(project_name, workspace)
        for project_name, workspace in stored.items()
    }


def save_project_workspaces() -> None:
    stored = to_jsonable(st.session_state.get("project_workspaces", {}))
    try:
        with WORKSPACE_STORE_PATH.open("w", encoding="utf-8") as file:
            json.dump(stored, file, indent=2)
    except OSError as exc:
        st.warning(f"Could not save project workspace data: {exc}")


def get_project_workspace(project_name: str) -> dict[str, Any]:
    if "project_workspaces" not in st.session_state:
        st.session_state.project_workspaces = load_project_workspaces()
    else:
        st.session_state.project_workspaces = {
            project_name: normalize_project_workspace(project_name, workspace)
            for project_name, workspace in st.session_state.project_workspaces.items()
        }

    workspaces = st.session_state.project_workspaces
    if project_name not in workspaces:
        workspaces[project_name] = empty_project()
        save_project_workspaces()

    return workspaces[project_name]


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "experiment"


def create_experiment(
    project_workspace: dict[str, Any],
    experiment_name: str,
    experiment_type: str = "auto",
) -> str:
    experiments = project_workspace.setdefault("experiments", {})
    base_id = slugify_name(experiment_name)
    experiment_id = base_id
    suffix = 2
    while experiment_id in experiments:
        experiment_id = f"{base_id}-{suffix}"
        suffix += 1

    created_at = now_timestamp()
    experiments[experiment_id] = {
        "name": experiment_name.strip() or "Untitled experiment",
        "created_at": created_at,
        "updated_at": created_at,
        "files": {},
        "analysis_outputs": {},
        "notes": "",
        "experiment_type": experiment_type,
    }
    save_project_workspaces()
    return experiment_id


def clear_session_keys_containing(fragment: str) -> None:
    for key in list(st.session_state.keys()):
        if fragment in str(key):
            st.session_state.pop(key, None)


def delete_project(project_name: str) -> None:
    st.session_state.get("project_workspaces", {}).pop(project_name, None)
    if st.session_state.get("active_project_name") == project_name:
        st.session_state.pop("active_project_name", None)
    clear_session_keys_containing(f"::{project_name}")
    clear_session_keys_containing(f"::{project_name}::")
    save_project_workspaces()


def delete_experiment(project_name: str, project_workspace: dict[str, Any], experiment_id: str) -> None:
    project_workspace.get("experiments", {}).pop(experiment_id, None)
    active_key = active_experiment_key(project_name)
    if st.session_state.get(active_key) == experiment_id:
        st.session_state.pop(active_key, None)
    clear_session_keys_containing(f"::{project_name}::{experiment_id}")
    save_project_workspaces()


def rename_experiment(project_workspace: dict[str, Any], experiment_id: str, new_name: str) -> None:
    experiment = project_workspace.get("experiments", {}).get(experiment_id)
    if not experiment:
        return

    cleaned_name = new_name.strip()
    if not cleaned_name:
        return

    experiment["name"] = cleaned_name
    experiment["updated_at"] = now_timestamp()
    save_project_workspaces()


def experiment_options(project_workspace: dict[str, Any]) -> dict[str, str]:
    experiments = project_workspace.get("experiments", {})
    return {
        f"{experiment.get('name', experiment_id)} · {format_timestamp(experiment.get('created_at'))}": experiment_id
        for experiment_id, experiment in experiments.items()
    }


def summarize_analysis(dataset: AnalyzedDataset) -> dict[str, Any]:
    potential = dataset.potential_v[np.isfinite(dataset.potential_v)]
    current = dataset.raw_current_a[np.isfinite(dataset.raw_current_a)]
    current_key_prefix = "current_density_a_cm2" if dataset.current_source == "Current density" else "current"
    return {
        "filename": dataset.filename,
        "sample_name": dataset.sample_name,
        "potential_col": dataset.potential_col,
        "current_col": dataset.current_col,
        "potential_unit": dataset.potential_unit,
        "current_unit": dataset.current_unit,
        "current_source": dataset.current_source,
        "cycle_source": dataset.cycle_source,
        "points": int(len(dataset.potential_v)),
        "potential_min_v": float(np.min(potential)) if potential.size else None,
        "potential_max_v": float(np.max(potential)) if potential.size else None,
        f"{current_key_prefix}_min": float(np.min(current)) if current.size else None,
        f"{current_key_prefix}_max": float(np.max(current)) if current.size else None,
        "warnings": dataset.warnings,
    }


def serialize_analyzed_dataset(
    dataset: AnalyzedDataset,
    parsed_dataset: Optional[ParsedDataset] = None,
) -> dict[str, Any]:
    return {
        "summary": summarize_analysis(dataset),
        "metadata": dataset.metadata,
        "warnings": dataset.warnings,
        "parsed": serialize_parsed_dataset(parsed_dataset) if parsed_dataset is not None else None,
        "trace": {
            "potential_v": dataset.potential_v.tolist(),
            "raw_current_a": dataset.raw_current_a.tolist(),
            "smoothed_current_a": (
                dataset.smoothed_current_a.tolist() if dataset.smoothed_current_a is not None else None
            ),
            "cycle_values": dataset.cycle_values.tolist() if dataset.cycle_values is not None else None,
            "cycle_source": dataset.cycle_source,
        },
    }


def store_experiment_analysis(
    project_workspace: dict[str, Any],
    experiment_id: str,
    analyzed: list[AnalyzedDataset],
    parsed_datasets: list[ParsedDataset],
    parser_summary_df: pd.DataFrame,
    esw_df: pd.DataFrame,
    peak_metric_df: pd.DataFrame,
    compact_metrics_df: pd.DataFrame,
    settings: dict[str, Any],
) -> None:
    experiment = project_workspace.setdefault("experiments", {}).setdefault(
        experiment_id,
        {
            "name": "Untitled experiment",
            "created_at": now_timestamp(),
            "updated_at": now_timestamp(),
            "files": {},
            "analysis_outputs": {},
            "notes": "",
        },
    )
    files = experiment.setdefault("files", {})
    parsed_by_filename = {dataset.filename: dataset for dataset in parsed_datasets}
    for dataset in analyzed:
        files[dataset.filename] = serialize_analyzed_dataset(dataset, parsed_by_filename.get(dataset.filename))

    experiment["updated_at"] = now_timestamp()
    experiment["experiment_type"] = "cv"
    experiment["analysis_outputs"] = {
        "experiment_type": "cv",
        "parser_summary": dataframe_records(parser_summary_df),
        "esw_summary": dataframe_records(esw_df),
        "peak_metrics": dataframe_records(peak_metric_df),
        "compact_metrics": dataframe_records(compact_metrics_df),
        "settings": settings,
    }
    save_project_workspaces()


def experiments_dataframe(project_workspace: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for experiment_id, experiment in project_workspace.get("experiments", {}).items():
        files = experiment.get("files", {})
        rows.append(
            {
                "experiment": experiment.get("name", experiment_id),
                "created": format_timestamp(experiment.get("created_at")),
                "updated": format_timestamp(experiment.get("updated_at")),
                "type": experiment_short_label(experiment.get("experiment_type", "cv")),
                "files": len(files),
                "notes": "yes" if experiment.get("notes") else "",
            }
        )
    return pd.DataFrame(rows)


def render_project_experiment_cards(project_name: str, project_workspace: dict[str, Any]) -> None:
    experiments = project_workspace.get("experiments", {})
    sorted_experiments = sorted(
        experiments.items(),
        key=lambda item: item[1].get("updated_at", item[1].get("created_at", "")),
        reverse=True,
    )

    with st.container(border=True):
        st.markdown("### Project experiments")
        for experiment_id, experiment in sorted_experiments:
            files = experiment.get("files", {})
            experiment_name = experiment.get("name", experiment_id)
            experiment_type = experiment_short_label(experiment.get("experiment_type", "cv"))
            file_label = f"{len(files)} file" if len(files) == 1 else f"{len(files)} files"
            created_label = format_timestamp(experiment.get("created_at"))
            updated_label = format_timestamp(experiment.get("updated_at"))
            notes = str(experiment.get("notes") or "").strip()
            notes_key = f"dashboard_notes_expanded::{project_name}::{experiment_id}"
            notes_expanded = bool(st.session_state.get(notes_key))
            notes_is_long = len(notes) > 180
            visible_notes = notes
            if notes_is_long and not notes_expanded:
                visible_notes = notes[:180].rstrip() + "..."
            if not visible_notes:
                visible_notes = "No notes yet."

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="project-experiment-tile">
                        <h4>{escape_html(experiment_name)}</h4>
                        <p>{escape_html(experiment_type)} · {escape_html(file_label)}</p>
                        <p>Created {escape_html(created_label)} · Updated {escape_html(updated_label)}</p>
                        <p class="experiment-notes-preview"><strong>Notes:</strong> {escape_html(visible_notes)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                action_cols = st.columns([0.82, 0.18])
                with action_cols[0]:
                    if notes_is_long:
                        if st.button(
                            "Show less" if notes_expanded else "Show more",
                            key=f"dashboard_toggle_notes::{project_name}::{experiment_id}",
                        ):
                            st.session_state[notes_key] = not notes_expanded
                            st.rerun()

                with action_cols[1]:
                    if st.button(
                        "Open",
                        key=f"dashboard_open_experiment::{project_name}::{experiment_id}",
                        type="primary",
                    ):
                        set_active_experiment(project_name, experiment_id)
                        st.rerun()


def render_experiment_notes(project_name: str, project_workspace: dict[str, Any], experiment_id: str) -> None:
    experiment = project_workspace["experiments"][experiment_id]
    st.subheader("Experiment notes")
    note_key = f"experiment_notes::{project_name}::{experiment_id}"
    if note_key not in st.session_state:
        st.session_state[note_key] = experiment.get("notes", "")
    experiment["notes"] = st.text_area(
        "Notes and comments",
        key=note_key,
        height=190,
        placeholder=(
            "Record observations, electrode prep, electrolyte concentration, reference electrode, "
            "what changed, interpretation, or follow-up steps."
        ),
    )
    experiment["updated_at"] = now_timestamp()
    save_project_workspaces()


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #0b0b0b;
            --muted: #555555;
            --line: #e6e6e6;
            --panel: #ffffff;
            --accent: #111111;
            --accent-dark: #000000;
            --shadow: rgba(0, 0, 0, 0.12);
        }
        .stApp {
            background: #ffffff;
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        h1, h2, h3, h4, p, label, span {
            color: var(--ink);
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f7f7f7 100%);
            border-right: 1px solid #dedede;
            box-shadow:
                14px 0 34px rgba(0, 0, 0, 0.12),
                inset -1px 0 0 rgba(255, 255, 255, 0.8);
        }
        div[data-testid="stMetric"],
        .project-hero,
        .workspace-card,
        .experiment-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-shadow: 0 16px 36px var(--shadow), 0 2px 0 rgba(255,255,255,0.95) inset;
        }
        .project-hero {
            padding: 2.6rem 2.8rem;
            margin: 0.7rem 0 1.3rem;
        }
        .project-hero h1 {
            font-size: clamp(2.1rem, 4vw, 4.8rem);
            line-height: 0.98;
            margin: 0.15rem 0 0.8rem;
        }
        .project-hero p {
            color: var(--muted);
            max-width: 760px;
            font-size: 1.04rem;
            line-height: 1.55;
            margin: 0;
        }
        .eyebrow {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .workspace-card,
        .experiment-card {
            padding: 1.1rem 1.2rem;
            margin: 0.6rem 0 1rem;
        }
        .workspace-card h3,
        .experiment-card h3 {
            margin: 0 0 0.35rem;
        }
        .workspace-card p,
        .experiment-card p {
            color: var(--muted);
            margin: 0;
        }
        .analysis-card {
            border: 1px solid #dddddd;
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1rem;
            background: #ffffff;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.07);
        }
        .analysis-card h4 {
            margin: 0 0 0.2rem;
            font-size: 1.05rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .analysis-card-header {
            align-items: center;
            display: flex;
            gap: 0.75rem;
            justify-content: space-between;
            margin-bottom: 0.35rem;
        }
        .analysis-card-header h4 {
            margin: 0;
        }
        .status-badge {
            border-radius: 999px;
            border: 1px solid #d8d8d8;
            color: #1f2937;
            display: inline-block;
            flex: 0 0 auto;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.35rem 0.55rem;
        }
        .status-passed {
            background: #e9f8ef;
            border-color: #9ad7b0;
            color: #176236;
        }
        .status-review-recommended {
            background: #fff6df;
            border-color: #e8ca77;
            color: #73510c;
        }
        .status-review-needed {
            background: #fff6df;
            border-color: #e8ca77;
            color: #73510c;
        }
        .status-incomplete {
            background: #eef2ff;
            border-color: #bac7ff;
            color: #334087;
        }
        .status-failed {
            background: #fdecec;
            border-color: #f3aaaa;
            color: #8a1f1f;
        }
        .analysis-card .analysis-file {
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 0.8rem;
            overflow-wrap: anywhere;
            white-space: pre-line;
        }
        .type-badge {
            background: #f3f4f6;
            border: 1px solid #d8d8d8;
            border-radius: 999px;
            color: #374151;
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1;
            margin: 0.2rem 0 0.85rem;
            padding: 0.35rem 0.55rem;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
            gap: 0.55rem;
        }
        .analysis-metric {
            border: 1px solid #eeeeee;
            border-radius: 8px;
            padding: 0.65rem 0.7rem;
            background: #fafafa;
        }
        .analysis-metric span {
            color: var(--muted);
            display: block;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: none;
        }
        .analysis-metric strong {
            display: block;
            font-size: 1rem;
            margin-top: 0.15rem;
            overflow-wrap: anywhere;
        }
        .analysis-metric .metric-lines {
            font-size: 0.92rem;
            line-height: 1.45;
            margin-top: 0.2rem;
            overflow-wrap: anywhere;
        }
        .analysis-metric .metric-lines b {
            font-weight: 700;
        }
        .readable-table {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin: 0.35rem 0 0.85rem;
            overflow-x: auto;
        }
        .readable-table table {
            border-collapse: collapse;
            font-size: 0.88rem;
            width: 100%;
        }
        .readable-table th,
        .readable-table td {
            border-bottom: 1px solid #eeeeee;
            padding: 0.58rem 0.65rem;
            text-align: left;
            vertical-align: top;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .readable-table th {
            background: #f8f8f8;
            color: #4b5563;
            font-weight: 700;
        }
        .readable-table tr:last-child td {
            border-bottom: 0;
        }
        .report-table table {
            min-width: 760px;
            table-layout: auto;
        }
        .report-table th,
        .report-table td {
            overflow-wrap: normal;
            white-space: nowrap;
            word-break: normal;
        }
        .report-table th:last-child,
        .report-table td:last-child {
            min-width: 260px;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .report-table th:nth-child(1),
        .report-table td:nth-child(1) {
            min-width: 110px;
        }
        .report-table th:nth-child(2),
        .report-table td:nth-child(2),
        .report-table th:nth-child(3),
        .report-table td:nth-child(3) {
            text-align: right;
            min-width: 82px;
        }
        .recommended-actions,
        .interpretation-panel {
            border: 1px solid #dddddd;
            border-radius: 10px;
            background: #ffffff;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.06);
            margin: 0.75rem 0 1rem;
            padding: 1rem 1.1rem;
        }
        .recommended-actions h4,
        .interpretation-panel h4 {
            font-size: 1.02rem;
            margin: 0 0 0.75rem;
        }
        .recommended-action {
            align-items: center;
            background: #fffaf0;
            border: 1px solid #f2d99c;
            border-radius: 8px;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-top: 0.55rem;
            padding: 0.75rem 0.85rem;
        }
        .recommended-action p {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.4;
            margin: 0.2rem 0 0;
        }
        .recommended-action span {
            background: #ffffff;
            border: 1px solid #d6c48e;
            border-radius: 999px;
            color: #5f460b;
            flex: 0 0 auto;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.35rem 0.55rem;
        }
        .interpretation-panel p {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0;
        }
        .data-quality-strip {
            border: 1px solid #e4e4e4;
            border-radius: 999px;
            color: #4b5563;
            display: inline-flex;
            font-size: 0.86rem;
            gap: 0.35rem;
            line-height: 1.35;
            margin: 0.25rem 0 0.7rem;
            max-width: 100%;
            padding: 0.45rem 0.75rem;
        }
        .data-quality-strip.review {
            background: #fffaf0;
            border-color: #f2d99c;
            color: #6f4e08;
        }
        .data-quality-strip.pass {
            background: #f4fbf6;
            border-color: #bde3c7;
            color: #245b38;
        }
        .project-experiment-tile {
            padding: 0.25rem 0 0.35rem;
        }
        .project-experiment-tile h4 {
            font-size: 1.2rem;
            line-height: 1.2;
            margin: 0 0 0.35rem;
        }
        .project-experiment-tile p {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0.15rem 0;
        }
        .project-experiment-tile .experiment-notes-preview {
            color: var(--ink);
            margin-top: 0.55rem;
            overflow-wrap: anywhere;
        }
        .home-action-label {
            color: var(--muted);
            font-size: 0.92rem;
            margin: 0.3rem 0 1rem;
        }
        .stButton {
            transition: transform 160ms ease, filter 160ms ease;
        }
        .stButton > button {
            width: 100%;
            min-height: 2.75rem;
            border-radius: 10px;
            border: 1px solid #d4d4d4;
            background: #ffffff;
            color: var(--ink);
            font-weight: 700;
            box-shadow: 0 7px 18px rgba(0, 0, 0, 0.08), 0 1px 0 rgba(255,255,255,0.9) inset;
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background 160ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.01);
            border-color: #111111;
            box-shadow: 0 16px 30px rgba(0, 0, 0, 0.15), 0 1px 0 rgba(255,255,255,0.95) inset;
            background: #ffffff;
        }
        .stButton > button[kind="primary"] {
            background: #f2f2f2;
            color: var(--ink);
            border: 1px solid #111111;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.14), 0 1px 0 rgba(255,255,255,0.95) inset;
        }
        .stButton > button[kind="primary"] * {
            color: var(--ink);
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button {
            width: 2rem;
            min-width: 2rem;
            height: 2rem;
            min-height: 2rem;
            padding: 0;
            border-radius: 8px;
            font-size: 1.1rem;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08), 0 1px 0 rgba(255,255,255,0.9) inset;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button p {
            margin: 0;
            line-height: 1;
        }
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
        }
        [data-testid="stFileUploader"],
        div[data-baseweb="select"] > div,
        textarea,
        input {
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }
        [data-testid="stFileUploader"]:hover,
        div[data-baseweb="select"] > div:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.10);
        }
        textarea:focus,
        input:focus {
            box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.08);
        }
        [data-testid="InputInstructions"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_active_project(project_name: str) -> None:
    st.session_state.active_project_name = project_name
    st.session_state.pop("active_upload_experiment_id", None)
    if "project_workspaces" in st.session_state and project_name in st.session_state.project_workspaces:
        st.session_state.project_workspaces[project_name]["last_opened_at"] = now_timestamp()
        save_project_workspaces()


def sorted_project_names(workspaces: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        workspaces,
        key=lambda project_name: workspaces[project_name].get(
            "last_opened_at",
            workspaces[project_name].get("created_at", ""),
        ),
        reverse=True,
    )


def render_home_project_sidebar() -> None:
    workspaces = st.session_state.project_workspaces

    with st.sidebar:
        st.header("Projects")
        st.caption("Most recently opened")

        if not workspaces:
            st.write("No saved projects yet.")
            return

        for project_name in sorted_project_names(workspaces):
            workspace = workspaces[project_name]
            if st.button(
                project_name,
                key=f"home_open_project::{project_name}",
            ):
                set_active_project(project_name)
                st.rerun()
            st.caption(
                f"{len(workspace.get('experiments', {}))} experiments · "
                f"{format_timestamp(workspace.get('last_opened_at') or workspace.get('created_at'))}"
            )

        st.divider()
        with st.expander("Delete project"):
            selected_project = st.selectbox(
                "Project",
                sorted_project_names(workspaces),
                key="home_delete_project_select",
            )
            st.warning(
                "Deleting this project permanently removes its experiments, saved analyses, files, notes, and comments."
            )
            confirm_project_name = st.text_input(
                f'Type "{selected_project}" to confirm',
                key=f"home_delete_project_confirm::{selected_project}",
            )
            if st.button(
                "Delete project permanently",
                disabled=confirm_project_name != selected_project,
                key=f"home_delete_project::{selected_project}",
            ):
                delete_project(selected_project)
                st.rerun()


def active_experiment_key(project_name: str) -> str:
    return f"active_experiment_id::{project_name}"


def set_active_experiment(project_name: str, experiment_id: str) -> None:
    st.session_state[active_experiment_key(project_name)] = experiment_id


def get_active_experiment_id(project_name: str, project_workspace: dict[str, Any]) -> Optional[str]:
    experiment_id = st.session_state.get(active_experiment_key(project_name))
    if experiment_id in project_workspace.get("experiments", {}):
        return experiment_id
    return None


def render_create_experiment_form(
    project_name: str,
    project_workspace: dict[str, Any],
    key_prefix: str,
) -> None:
    st.markdown(
        "<div class='workspace-card'><h3>Create experiment</h3><p>Create the experiment, then upload a file. VoltScope will detect whether it is CV or LSV from the data.</p></div>",
        unsafe_allow_html=True,
    )
    default_name = f"Experiment {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    with st.form(key=f"{key_prefix}_create_experiment_form::{project_name}"):
        experiment_name = st.text_input(
            "Experiment name",
            value=default_name,
            key=f"{key_prefix}_new_experiment::{project_name}",
        )
        submitted = st.form_submit_button("Create experiment")

    if submitted:
        cleaned_name = experiment_name.strip()
        if not cleaned_name:
            st.error("Enter an experiment name.")
            return

        experiment_id = create_experiment(
            project_workspace,
            cleaned_name,
            "auto",
        )
        set_active_experiment(project_name, experiment_id)
        st.rerun()


def render_project_sidebar(project_name: str, project_workspace: dict[str, Any]) -> Optional[str]:
    experiments = project_workspace.get("experiments", {})

    with st.sidebar:
        st.header(project_name)
        st.caption(f"{len(experiments)} experiments")
        if st.button("Switch project", key="switch_project"):
            st.session_state.pop("active_project_name", None)
            st.rerun()

        st.divider()
        if experiments:
            if st.button("New experiment", key=f"sidebar_new_experiment_button::{project_name}"):
                st.session_state.pop(active_experiment_key(project_name), None)
                st.session_state.pop(f"experiment_actions_menu::{project_name}", None)
                st.rerun()

        st.subheader("Experiments")

        active_experiment_id = get_active_experiment_id(project_name, project_workspace)
        menu_key = f"experiment_actions_menu::{project_name}"

        if not experiments:
            st.caption("No experiments yet.")

        for experiment_id, experiment in sorted(
            experiments.items(),
            key=lambda item: item[1].get("updated_at", item[1].get("created_at", "")),
            reverse=True,
        ):
            experiment_name = experiment.get("name", experiment_id)
            is_active = experiment_id == active_experiment_id
            label = f"● {experiment_name}" if is_active else experiment_name
            experiment_col, actions_col = st.columns([0.84, 0.16])

            with experiment_col:
                if st.button(
                    label,
                    key=f"open_experiment::{project_name}::{experiment_id}",
                    type="primary" if is_active else "secondary",
                ):
                    set_active_experiment(project_name, experiment_id)
                    st.rerun()

            with actions_col:
                if st.button(
                    "…",
                    key=f"experiment_menu_button::{project_name}::{experiment_id}",
                    help="Experiment actions",
                ):
                    current_menu_id = st.session_state.get(menu_key)
                    st.session_state[menu_key] = None if current_menu_id == experiment_id else experiment_id
                    st.rerun()

            st.caption(format_timestamp(experiment.get("updated_at") or experiment.get("created_at")))

            if st.session_state.get(menu_key) == experiment_id:
                with st.container():
                    st.markdown("**Experiment options**")
                    with st.form(key=f"rename_experiment_form::{project_name}::{experiment_id}"):
                        new_experiment_name = st.text_input(
                            "Experiment name",
                            value=experiment_name,
                            key=f"rename_experiment_input::{project_name}::{experiment_id}",
                        )
                        rename_submitted = st.form_submit_button("Rename")

                    if rename_submitted:
                        cleaned_name = new_experiment_name.strip()
                        if not cleaned_name:
                            st.error("Enter an experiment name.")
                        elif cleaned_name != experiment_name:
                            rename_experiment(project_workspace, experiment_id, cleaned_name)
                            st.session_state[menu_key] = None
                            st.rerun()
                        else:
                            st.info("The experiment already has that name.")

                    st.warning(
                        "Deleting this experiment permanently removes its uploaded files, saved analyses, notes, and comments."
                    )
                    confirm_experiment_name = st.text_input(
                        f'Type "{experiment_name}" to delete this experiment',
                        key=f"delete_experiment_confirm::{project_name}::{experiment_id}",
                    )
                    if st.button(
                        "Delete experiment",
                        disabled=confirm_experiment_name != experiment_name,
                        key=f"delete_experiment::{project_name}::{experiment_id}",
                    ):
                        delete_experiment(project_name, project_workspace, experiment_id)
                        st.session_state[menu_key] = None
                        st.rerun()

    return active_experiment_id

def render_project_home() -> None:
    render_home_project_sidebar()
    st.markdown(
        """
        <section class="project-hero">
            <div class="eyebrow">VoltScope workspace</div>
            <h1>Open your electrochemistry workbench.</h1>
            <p>Create a project for a new study, or open an existing project from the sidebar to continue experiments, notes, and saved analysis results.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        "<div class='workspace-card'><h3>Create new project</h3><p>Name the workspace that will hold experiments, files, outputs, and notes.</p></div>",
        unsafe_allow_html=True,
    )
    project_name = st.text_input("Project name", placeholder="Example: Nickel electrolyte screening")
    if st.button("Create project", type="primary"):
        cleaned_name = project_name.strip()
        if not cleaned_name:
            st.error("Enter a project name before continuing.")
        else:
            get_project_workspace(cleaned_name)
            set_active_project(cleaned_name)
            st.rerun()


def format_result_value(value: Any, unit: str = "") -> str:
    if value is None:
        return "Not found"
    if isinstance(value, float) and not np.isfinite(value):
        return "Not found"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}{unit}"
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude != 0 and (magnitude < 0.001 or magnitude >= 10000):
            return f"{float(value):.3e}{unit}"
        return f"{float(value):.3f}{unit}"
    return f"{value}{unit}"


def prettify_result_label(key: str) -> str:
    known_labels = {
        "points": "Points",
        "potential_min_V": "Potential min",
        "potential_max_V": "Potential max",
        "current_min_A": "Current min",
        "current_max_A": "Current max",
        "current_density_A_cm2_min": "Current density min",
        "current_density_A_cm2_max": "Current density max",
        "cathodic_limit_V": "Cathodic limit",
        "anodic_limit_V": "Anodic limit",
        "operational_ESW_V": "Operational ESW",
        "duration_s": "Duration",
        "initial_current_A": "Initial current",
        "final_current_A": "Final current",
        "steady_state_current_A": "Steady-state current",
        "charge_C": "Charge",
        "initial_potential_V": "Initial potential",
        "final_potential_V": "Final potential",
        "delta_potential_V": "Potential change",
        "average_slope_V_s": "Average slope",
        "zreal_min_ohm": "Z real min",
        "zreal_max_ohm": "Z real max",
        "max_minus_zimag_ohm": "Max -Z imaginary",
        "estimated_rs_ohm": "Estimated Rs",
        "estimated_diameter_ohm": "Estimated diameter",
        "frequency_min_Hz": "Frequency min",
        "frequency_max_Hz": "Frequency max",
    }
    if key.startswith("threshold_"):
        return "Threshold"
    return known_labels.get(key, key.replace("_", " ").title())


def result_value_unit(key: str) -> str:
    if key.endswith("_V"):
        return " V"
    if key.endswith("_A"):
        return " A"
    if key.endswith("_A_cm2_min") or key.endswith("_A_cm2_max"):
        return " A/cm\u00b2"
    if key.endswith("_s"):
        return " s"
    if key.endswith("_C"):
        return " C"
    if key.endswith("_V_s"):
        return " V/s"
    if key.endswith("_ohm"):
        return " ohm"
    if key.endswith("_Hz"):
        return " Hz"
    if key.startswith("threshold_"):
        return f" {display_unit_label(key.replace('threshold_', ''))}"
    return ""


def result_metric_items(row: dict[str, Any]) -> list[tuple[str, str]]:
    preferred_keys = [
        "points",
        "potential_min_V",
        "potential_max_V",
        "current_min_A",
        "current_max_A",
        "current_density_A_cm2_min",
        "current_density_A_cm2_max",
        "cathodic_limit_V",
        "anodic_limit_V",
        "operational_ESW_V",
        "duration_s",
        "initial_current_A",
        "final_current_A",
        "steady_state_current_A",
        "charge_C",
        "initial_potential_V",
        "final_potential_V",
        "delta_potential_V",
        "average_slope_V_s",
        "zreal_min_ohm",
        "zreal_max_ohm",
        "max_minus_zimag_ohm",
        "estimated_rs_ohm",
        "estimated_diameter_ohm",
        "frequency_min_Hz",
        "frequency_max_Hz",
    ]
    threshold_keys = [key for key in row if key.startswith("threshold_")]
    ordered_keys = preferred_keys + threshold_keys
    items = []
    for key in ordered_keys:
        if key not in row:
            continue
        items.append((prettify_result_label(key), format_result_value(row.get(key), result_value_unit(key))))
    return items


def render_analysis_result_cards(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("No saved analysis results yet.")
        return

    for row in results:
        sample_name = row.get("sample_name") or "Unnamed sample"
        file_name = row.get("file_name") or row.get("filename") or ""
        metric_html = "\n".join(
            f"""
            <div class="analysis-metric">
                <span>{escape_html(label)}</span>
                <strong>{escape_html(value)}</strong>
            </div>
            """
            for label, value in result_metric_items(row)
        )
        st.markdown(
            f"""
            <div class="analysis-card">
                <h4>{escape_html(str(sample_name))}</h4>
                <div class="analysis-file">{escape_html(str(file_name))}</div>
                <div class="analysis-grid">{metric_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_recommended_actions(messages: list[str], key_prefix: str) -> None:
    visible_messages = unique_messages([message for message in messages if message])
    if not visible_messages:
        return

    action_cards = []
    seen_action_cards: set[tuple[str, str, str]] = set()
    for message in visible_messages:
        lowered = message.lower()
        if "baseline" in lowered or "noisy" in lowered or "ambiguous" in lowered or "similar prominence" in lowered:
            title = "Noise or baseline issue detected"
            message = (
                "This trace is noisy or baseline-sensitive. Try smoothing, baseline correction, "
                "or manual peak override before interpreting peak currents."
            )
            action = "Review peak controls"
        elif "unit" in lowered:
            title = "Confirm units"
            if "density" in lowered:
                message = "Confirm the current-density unit before using current-density-dependent metrics."
            elif "current" in lowered:
                message = "Confirm the current unit before using current-dependent metrics."
            action = "Review unit mapping"
        elif "scan rate" in lowered:
            title = "Add scan-rate metadata"
            message = (
                "Scan rate was not found in metadata and could not be estimated because no time column "
                "was detected. Add scan rate metadata if scan-rate-dependent interpretation is needed."
            )
            action = "Add scan-rate metadata"
        elif "cycle" in lowered:
            title = "Review cycle selection"
            action = "Check selected cycle"
        elif "irreversible" in lowered:
            title = "Irreversible behavior"
            if "oxidation-only" in lowered or "strong oxidation" in lowered:
                message = (
                    "A strong oxidation peak was detected, but no reliable cathodic return peak was found. "
                    "\u0394Ep, E\u00b0\u2032, and |Ipa/Ipc| are not reliable for this trace."
                )
                action = "Review candidate peaks"
            elif "reduction-only" in lowered or "strong reduction" in lowered:
                message = (
                    "A strong reduction peak was detected, but no reliable anodic return peak was found. "
                    "\u0394Ep, E\u00b0\u2032, and |Ipa/Ipc| are not reliable for this trace."
                )
                action = "Review candidate peaks"
            else:
                action = "Review candidate peaks"
        else:
            title = "Review recommended"
            if "selected oxidation peak occurs at a lower potential" in lowered or "invalid" in lowered:
                message = (
                    "Review candidate peaks, smoothing, and baseline correction before using \u0394Ep, "
                    "E\u00b0\u2032, or peak-current ratio."
                )
                action = "Review candidate peaks"
            else:
                action = "Open advanced details"
        card = (title, message, action)
        if card in seen_action_cards:
            continue
        seen_action_cards.add(card)
        action_cards.append(card)

    cards_html = "\n".join(
        f"""
        <div class="recommended-action">
            <div>
                <strong>{escape_html(title)}</strong>
                <p>{escape_html(message)}</p>
            </div>
            <span>{escape_html(action)}</span>
        </div>
        """
        for title, message, action in action_cards
    )
    st.markdown(
        f"""
        <div class="recommended-actions">
            <h4>Recommended actions</h4>
            {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_quality_strip(items: list[str], review_needed: bool = False) -> None:
    visible_items = unique_messages([item for item in items if item])
    if not visible_items:
        return
    text = "Data quality: " + " \u00b7 ".join(visible_items)
    state_class = "review" if review_needed else "pass"
    st.markdown(
        f'<div class="data-quality-strip {state_class}">{escape_html(text)}</div>',
        unsafe_allow_html=True,
    )


def render_readable_table(df: pd.DataFrame, class_name: str = "readable-table") -> None:
    if df.empty:
        return
    safe_class_name = " ".join(
        token for token in str(class_name).split() if token.replace("-", "").replace("_", "").isalnum()
    ) or "readable-table"
    st.markdown(
        f'<div class="{escape_html(safe_class_name)}">{df.to_html(index=False, escape=True)}</div>',
        unsafe_allow_html=True,
    )


def format_peak_label_for_table(value: Any) -> str:
    text = "" if value is None else str(value)
    return {"oxidation": "Oxidation", "reduction": "Reduction"}.get(text.lower(), text)


def format_confidence_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return ""
    normalized = re.sub(r"[\s_]+", "-", text.lower())
    special_labels = {
        "medium-high": "Medium-High",
        "medium-low": "Medium-Low",
        "not-detected": "Not detected",
        "not-detected-/-low": "Not detected / Low",
    }
    if normalized in special_labels:
        return special_labels[normalized]
    return normalized.replace("-", " ").title().replace(" ", "-") if "medium-" in normalized else text.title()


def cv_report_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    display_df = df.copy()
    for column in display_df.columns:
        column_text = str(column)
        if column_text == "potential_V":
            display_df[column] = display_df[column].map(format_voltage_table_value)
        elif column_text.startswith(("current_", "raw_current_", "baseline_current_", "corrected_current_", "prominence_")):
            display_df[column] = display_df[column].map(format_current_table_value)
    if "peak" in display_df.columns:
        display_df["peak"] = display_df["peak"].map(format_peak_label_for_table)
    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].map(format_confidence_label)

    rename_map: dict[str, str] = {
        "role": "Role",
        "status": "Status",
        "peak": "Peak",
        "metric": "Metric",
        "value": "Value",
        "method": "Method",
        "potential_V": "E / V",
        "scan": "Scan",
        "confidence": "Confidence",
        "reason": "Reason",
        "couple_id": "Couple",
        "id": "ID",
        "extremum": "Extremum",
        "index": "Index",
    }
    for column in list(display_df.columns):
        column_text = str(column)
        if column_text.startswith("current_"):
            unit = column_text.replace("current_", "")
            rename_map[column_text] = f"I / {unit}"
        elif column_text.startswith("raw_current_"):
            unit = column_text.replace("raw_current_", "")
            rename_map[column_text] = f"Raw I / {unit}"
        elif column_text.startswith("baseline_current_"):
            unit = column_text.replace("baseline_current_", "")
            rename_map[column_text] = f"Baseline I / {unit}"
        elif column_text.startswith("corrected_current_"):
            unit = column_text.replace("corrected_current_", "")
            rename_map[column_text] = f"Corrected I / {unit}"
        elif column_text.startswith("prominence_"):
            unit = column_text.replace("prominence_", "")
            rename_map[column_text] = f"Prominence / {unit}"
    return display_df.rename(columns={key: value for key, value in rename_map.items() if key in display_df.columns})


def render_cv_report_table(df: pd.DataFrame) -> None:
    render_readable_table(cv_report_display_table(df), class_name="readable-table report-table")


def cv_data_quality_items(
    status: str,
    unit_review_required: bool,
    quality_messages: list[str],
    parser_confidence: str,
    current_scale_review_needed: bool,
    cv_behavior: Optional[CVBehaviorResult],
) -> tuple[list[str], bool]:
    items = ["Parsed successfully" if parser_confidence != "Low" else "Parser review recommended"]
    behavior = cv_behavior.behavior if cv_behavior is not None else ""
    if unit_review_required:
        items.extend(["Unit confirmation required", "Current-dependent metrics may be scaled incorrectly"])
        return items, True
    if current_scale_review_needed:
        items.append("Current scale review recommended")
    if behavior == "irreversible_oxidation_only":
        items.extend(["Oxidation peak detected", "No reliable cathodic peak", "Review recommended"])
        return items, True
    if behavior == "irreversible_reduction_only":
        items.extend(["Reduction peak detected", "No reliable anodic peak", "Review recommended"])
        return items, True
    if quality_messages or behavior in {"noisy_ambiguous", "no_reliable_peaks"}:
        if any("noisy" in message.lower() for message in quality_messages) or behavior == "noisy_ambiguous":
            items.append("Noisy trace detected")
        items.append("Peak selection review recommended")
    elif status == "Passed":
        items.extend(["Units detected", "Peak pair found", "No review required"])
    else:
        items.append("Review recommended")
    return items, status != "Passed" or current_scale_review_needed or bool(quality_messages)


def lsv_data_quality_items(summary: Optional[dict[str, str]], unit_review_required: bool) -> tuple[list[str], bool]:
    items = ["Parsed successfully"]
    if unit_review_required:
        items.extend(["Unit confirmation required", "Current-dependent metrics may be scaled incorrectly"])
        return items, True
    if not summary:
        items.append("Analysis pending")
        return items, True
    status = summary.get("status", "Review needed")
    if summary.get("onset_potential") and summary.get("onset_potential") != "Not detected":
        items.extend(["Onset detected", "Sustained threshold crossing confirmed"])
    else:
        items.append("Onset review recommended")
    metadata_status = summary.get("metadata_completeness")
    if status == "Passed":
        if metadata_status == "Complete":
            items.append("No review required")
    elif status:
        items.append(status)
    return items, status != "Passed" or metadata_status == "Review recommended"


def metadata_status_notice(summary: Optional[dict[str, Any]]) -> Optional[str]:
    if not summary:
        return None
    metadata_status = str(summary.get("metadata_completeness") or "").strip()
    if metadata_status not in {"Incomplete", "Review recommended"}:
        return None
    missing = str(summary.get("missing_metadata") or "").strip()
    if missing and missing != "None":
        return f"Metadata completeness: {metadata_status}. Missing or unconfirmed: {missing}."
    return f"Metadata completeness: {metadata_status}."


def render_metadata_status_notice(summary: Optional[dict[str, Any]]) -> None:
    notice = metadata_status_notice(summary)
    if not notice:
        return
    st.warning(
        notice
        + " Analysis can still pass when the numerical result is valid, but confirm metadata before publication-level interpretation."
    )


def cv_metadata_summary_from_parser_details(details: dict[str, Any], current_source: str) -> dict[str, Any]:
    missing: list[str] = []
    if details.get("scan_rate_estimate") == "Not available":
        missing.append("scan rate")
    if details.get("reference_electrode_metadata") == "Not detected":
        missing.append("reference electrode")
    if current_source == "Current density" and details.get("electrode_area_metadata") == "Not detected":
        missing.append("electrode area")
    return {
        "metadata_completeness": "Incomplete" if missing else "Complete",
        "missing_metadata": ", ".join(missing) if missing else "None",
    }


def compact_text_list(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    compacted: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        compacted.append(text)
    return compacted


def formatted_ai_float(value: Any, unit: str = "", precision: int = 4) -> str:
    if value is None:
        return "Not available"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "Not available"
    suffix = f" {display_unit_label(unit)}" if unit else ""
    return f"{numeric:.{precision}g}{suffix}"


def ai_current_metric_value(
    current_a: Optional[float],
    display_unit: str,
    current_source: str,
    electrode_area_cm2: float,
    unit_confidence: str,
    precision: int = 4,
) -> str:
    if pending_current_metric(unit_confidence):
        return "Pending unit confirmation"
    if current_a is None:
        return "Not detected"
    display_value = current_to_display(
        np.array([current_a]),
        display_unit,
        current_source,
        electrode_area_cm2,
    )[0]
    return formatted_ai_float(display_value, display_unit, precision)


def metric_status_record(
    value: str,
    unit: str,
    status: str,
    reason: str = "",
) -> dict[str, str]:
    record = {
        "value": value,
        "unit": display_unit_label(unit) if unit else "",
        "status": status,
    }
    if reason:
        record["reason"] = reason
    return record


def metadata_status_record(
    value: Any = None,
    unit: str = "",
    status: str = "missing",
    source: str = "not found in uploaded file",
) -> dict[str, str]:
    text = str(value or "").strip()
    if not text or text in {"Not available", "Not detected", "None"}:
        text = ""
    return {
        "value": text,
        "unit": display_unit_label(unit) if unit else "",
        "status": status,
        "source": source,
    }


def metadata_value_is_available(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text not in {"Not available", "Not detected", "None"}


def strip_reference_source_label(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s*\((?:from metadata|default|confirmed|user-selected)\)\s*$", "", text)


def metadata_lookup(metadata: Optional[dict[str, str]], aliases: set[str]) -> Optional[str]:
    if not metadata:
        return None
    for key, value in metadata.items():
        compact_key = re.sub(r"[^a-z0-9]+", "", str(key).lower())
        compact_value = str(value or "").strip()
        if not compact_value:
            continue
        if compact_key in aliases or any(alias in compact_key for alias in aliases):
            return compact_value
    return None


def available_metadata_from_context(
    *,
    metadata: Optional[dict[str, str]] = None,
    scan_rate: Any = None,
    electrode_area: Any = None,
    reference_electrode: Any = None,
    reference_electrode_source: str = "",
) -> dict[str, Any]:
    available: dict[str, Any] = {}
    scan_rate_text = str(scan_rate or "").strip()
    if scan_rate_text and scan_rate_text != "Not available":
        available["scan_rate"] = scan_rate_text

    area_text = str(electrode_area or "").strip()
    if area_text and area_text not in {"Not detected", "Not available"}:
        available["electrode_area"] = area_text

    reference_text = strip_reference_source_label(reference_electrode)
    if reference_text and reference_electrode_source not in {"default", "not available"}:
        available["reference_electrode"] = reference_text

    metadata = metadata or {}
    field_aliases = {
        "electrolyte": {"electrolyte", "solution"},
        "concentration": {"concentration", "molarity", "molality"},
        "working_electrode": {"workingelectrode", "working", "we"},
        "counter_electrode": {"counterelectrode", "counter", "ce", "auxiliaryelectrode"},
        "instrument": {"instrument", "potentiostat"},
    }
    for field, aliases in field_aliases.items():
        value = metadata_lookup(metadata, aliases)
        if value:
            available[field] = value

    metadata_reference = reference_electrode_from_metadata(metadata)
    if metadata_reference:
        available["reference_electrode"] = metadata_reference
    metadata_area = electrode_area_metadata_summary(metadata)
    if metadata_area:
        available["electrode_area"] = metadata_area
    metadata_scan_rate = scan_rate_from_metadata(metadata)
    if metadata_scan_rate is not None:
        available["scan_rate"] = scan_rate_summary_from_value(metadata_scan_rate)

    return available


def metadata_status_from_context(
    *,
    metadata: Optional[dict[str, str]] = None,
    scan_rate: Any = None,
    scan_rate_source: str = "",
    electrode_area: Any = None,
    electrode_area_source: str = "",
    reference_electrode: Any = None,
    reference_electrode_source: str = "",
    cycle_count: Any = None,
    cycle_assignment_uncertain: bool = False,
    scan_mode: Any = None,
) -> dict[str, dict[str, str]]:
    metadata = metadata or {}
    status: dict[str, dict[str, str]] = {
        "scan_rate": metadata_status_record(unit="mV/s"),
        "electrode_area": metadata_status_record(unit="cm²"),
        "reference_electrode": metadata_status_record(),
        "electrolyte": metadata_status_record(),
        "concentration": metadata_status_record(),
        "working_electrode": metadata_status_record(),
        "counter_electrode": metadata_status_record(),
        "cycle_count": metadata_status_record(status="not_applicable", source="not applicable for this technique"),
        "scan_mode": metadata_status_record(status="not_applicable", source="not applicable for this technique"),
    }

    metadata_scan_rate = scan_rate_from_metadata(metadata)
    if metadata_scan_rate is not None:
        status["scan_rate"] = metadata_status_record(
            scan_rate_summary_from_value(metadata_scan_rate),
            "mV/s",
            "detected",
            "parsed from file metadata",
        )
    elif metadata_value_is_available(scan_rate):
        source = str(scan_rate_source or "").lower()
        if "metadata" in source:
            scan_status = "detected"
            scan_source = "parsed from file metadata"
        elif "manual" in source or "user" in source:
            scan_status = "manual"
            scan_source = "manual user entry"
        else:
            scan_status = "inferred"
            scan_source = "inferred from time and potential columns"
        status["scan_rate"] = metadata_status_record(scan_rate, "mV/s", scan_status, scan_source)

    metadata_area = electrode_area_metadata_summary(metadata)
    if metadata_area:
        status["electrode_area"] = metadata_status_record(
            metadata_area,
            "cm²",
            "detected",
            "parsed from file metadata",
        )
    elif metadata_value_is_available(electrode_area):
        area_source = str(electrode_area_source or "").lower()
        if "metadata" in area_source:
            status["electrode_area"] = metadata_status_record(
                electrode_area,
                "cm²",
                "detected",
                "parsed from file metadata",
            )
        elif "manual" in area_source or "user" in area_source:
            status["electrode_area"] = metadata_status_record(
                electrode_area,
                "cm²",
                "manual",
                "manual user entry",
            )
        else:
            status["electrode_area"] = metadata_status_record(
                electrode_area,
                "cm²",
                "inferred",
                "provided by app context",
            )

    metadata_reference = reference_electrode_from_metadata(metadata)
    if metadata_reference:
        status["reference_electrode"] = metadata_status_record(
            metadata_reference,
            "",
            "detected",
            "parsed from file metadata",
        )
    elif metadata_value_is_available(reference_electrode):
        reference_value = strip_reference_source_label(reference_electrode)
        source = str(reference_electrode_source or "").lower()
        if "metadata" in source:
            status["reference_electrode"] = metadata_status_record(
                reference_value,
                "",
                "detected",
                "parsed from file metadata",
            )
        elif "default" in source:
            status["reference_electrode"] = metadata_status_record(
                reference_value,
                "",
                "defaulted",
                "application default",
            )
        elif "manual" in source or "user" in source or "confirmed" in source:
            status["reference_electrode"] = metadata_status_record(
                reference_value,
                "",
                "manual",
                "manual user entry",
            )

    field_aliases = {
        "electrolyte": {"electrolyte", "solution"},
        "concentration": {"concentration", "molarity", "molality"},
        "working_electrode": {"workingelectrode", "working", "we"},
        "counter_electrode": {"counterelectrode", "counter", "ce", "auxiliaryelectrode"},
    }
    for field, aliases in field_aliases.items():
        value = metadata_lookup(metadata, aliases)
        if value:
            status[field] = metadata_status_record(value, "", "detected", "parsed from file metadata")

    if metadata_value_is_available(cycle_count):
        cycle_status = "inferred" if cycle_assignment_uncertain else "detected"
        cycle_source = (
            "inferred from potential turning points"
            if cycle_assignment_uncertain
            else "detected from cycle column or parsed numeric data"
        )
        status["cycle_count"] = metadata_status_record(cycle_count, "", cycle_status, cycle_source)

    if metadata_value_is_available(scan_mode):
        status["scan_mode"] = metadata_status_record(
            scan_mode,
            "",
            "inferred",
            "inferred from potential direction",
        )

    return status


def missing_metadata_fields(
    available_metadata: dict[str, Any],
    required_fields: Optional[list[str]] = None,
) -> list[str]:
    fields = required_fields or [
        "scan_rate",
        "reference_electrode",
        "electrode_area",
        "electrolyte",
        "concentration",
        "working_electrode",
        "counter_electrode",
        "instrument",
    ]
    return [field for field in fields if field not in available_metadata]


def cv_metric_validity(
    metrics: dict[str, Optional[float]],
    *,
    display_current_unit: str,
    current_source: str,
    electrode_area_cm2: float,
    unit_confidence: str,
    cv_behavior: Optional[CVBehaviorResult],
    pair_metric_review: Optional[str],
    pair_metrics_valid: bool,
) -> dict[str, dict[str, str]]:
    behavior = cv_behavior.behavior if cv_behavior is not None else "unknown"
    pair_reason = pair_metric_review or ""
    epa = metrics.get("epa_V")
    epc = metrics.get("epc_V")
    ipa = metrics.get("ipa_A")
    ipc = metrics.get("ipc_A")
    delta_ep = metrics.get("delta_ep_V")
    formal_potential = (epa + epc) / 2 if epa is not None and epc is not None else None
    current_pending = pending_current_metric(unit_confidence)

    validity = {
        "Epa": metric_status_record(
            formatted_ai_float(epa, "V") if epa is not None else "Not detected",
            "V",
            "valid" if epa is not None else "not_detected",
            "" if epa is not None else "No reliable oxidation peak was selected.",
        ),
        "Ipa": metric_status_record(
            ai_current_metric_value(ipa, display_current_unit, current_source, electrode_area_cm2, unit_confidence),
            display_current_unit,
            "pending_unit_confirmation" if current_pending else ("valid" if ipa is not None else "not_detected"),
            "Current unit must be confirmed before interpreting peak current."
            if current_pending
            else ("" if ipa is not None else "No reliable oxidation peak current was selected."),
        ),
        "Epc": metric_status_record(
            formatted_ai_float(epc, "V") if epc is not None else "Not detected",
            "V",
            "valid" if epc is not None else "not_detected",
            "" if epc is not None else "No reliable reduction peak was selected.",
        ),
        "Ipc": metric_status_record(
            ai_current_metric_value(ipc, display_current_unit, current_source, electrode_area_cm2, unit_confidence),
            display_current_unit,
            "pending_unit_confirmation" if current_pending else ("valid" if ipc is not None else "not_detected"),
            "Current unit must be confirmed before interpreting peak current."
            if current_pending
            else ("" if ipc is not None else "No reliable reduction peak current was selected."),
        ),
    }

    if pair_metrics_valid and delta_ep is not None:
        validity["\u0394Ep"] = metric_status_record(formatted_ai_float(delta_ep * 1000, "mV"), "mV", "valid")
    else:
        pair_status = "not_applicable" if behavior.startswith("irreversible") else "invalid"
        validity["\u0394Ep"] = metric_status_record(
            "Not applicable" if pair_status == "not_applicable" else "Invalid",
            "mV",
            pair_status,
            pair_reason or "A reliable reversible Epa/Epc pair was not selected.",
        )

    if pair_metrics_valid and formal_potential is not None:
        validity["E\u00b0\u2032"] = metric_status_record(formatted_ai_float(formal_potential, "V"), "V", "valid")
    else:
        pair_status = "not_applicable" if behavior.startswith("irreversible") else "invalid"
        validity["E\u00b0\u2032"] = metric_status_record(
            "Not applicable" if pair_status == "not_applicable" else "Invalid",
            "V",
            pair_status,
            pair_reason or "A reliable reversible Epa/Epc pair was not selected.",
        )

    ratio = metrics.get("ipa_ipc_ratio")
    if current_pending:
        validity["|Ipa/Ipc|"] = metric_status_record(
            "Pending unit confirmation",
            "",
            "pending_unit_confirmation",
            "Current unit must be confirmed before interpreting current-dependent metrics.",
        )
    elif pair_metrics_valid and ratio is not None:
        validity["|Ipa/Ipc|"] = metric_status_record(formatted_ai_float(ratio, ""), "", "valid")
    else:
        pair_status = "not_applicable" if behavior.startswith("irreversible") else "invalid"
        validity["|Ipa/Ipc|"] = metric_status_record(
            "Not meaningful" if pair_status == "not_applicable" else "Invalid",
            "",
            pair_status,
            pair_reason or "A reliable reversible Epa/Epc pair was not selected.",
        )

    return validity


def cv_metric_status(
    metric_validity: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    not_applicable = metric_status_record(
        "Not applicable",
        "",
        "not_applicable",
        "This metric applies to LSV onset analysis, not CV peak analysis.",
    )
    return {
        "Epa": metric_validity.get("Epa", metric_status_record("Not detected", "V", "not_detected")),
        "Ipa": metric_validity.get("Ipa", metric_status_record("Not detected", "", "not_detected")),
        "Epc": metric_validity.get("Epc", metric_status_record("Not detected", "V", "not_detected")),
        "Ipc": metric_validity.get("Ipc", metric_status_record("Not detected", "", "not_detected")),
        "delta_Ep": metric_validity.get("ΔEp", metric_status_record("Not applicable", "mV", "not_applicable")),
        "E0_prime": metric_validity.get("E°′", metric_status_record("Not applicable", "V", "not_applicable")),
        "peak_current_ratio": metric_validity.get(
            "|Ipa/Ipc|",
            metric_status_record("Not applicable", "", "not_applicable"),
        ),
        "onset_potential": not_applicable,
        "threshold": not_applicable,
        "max_current": not_applicable,
        "max_current_density": not_applicable,
    }


def lsv_metric_status_from_summary(summary: dict[str, Any], key: str) -> str:
    value = str(summary.get(key, "") or "")
    if "unit unconfirmed" in value.lower() or "Pending unit confirmation" in value:
        return "pending_unit_confirmation"
    if value in {"", "Not available", "Not detected"}:
        return "not_detected" if key == "onset_potential" else "not_applicable"
    if summary.get("status") not in {"Passed", "Review needed"} and key == "onset_potential":
        return "not_detected"
    return "valid"


def lsv_metric_validity(summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    summary_text = " ".join(str(value).lower() for value in summary.values())
    unit_review = (
        str(summary.get("unit_status", "")).lower() == "unconfirmed"
        or "unit unconfirmed" in summary_text
        or "pending unit confirmation" in summary_text
    )
    onset_value = str(summary.get("onset_potential") or "Not detected")
    onset_status = lsv_metric_status_from_summary(summary, "onset_potential")
    if unit_review and onset_status == "valid":
        onset_status = "pending_unit_confirmation"
    onset_reason = ""
    if onset_status == "pending_unit_confirmation":
        onset_reason = "Onset uses a current/current-density threshold whose unit is not confirmed."
    elif onset_status == "not_detected":
        onset_reason = summary.get("review_note", "No reliable sustained threshold crossing was detected.")

    threshold_value = str(summary.get("threshold") or summary.get("threshold_magnitude") or "Not available")
    threshold_status = "pending_unit_confirmation" if unit_review else ("valid" if threshold_value != "Not available" else "not_applicable")
    threshold_reason = "Threshold unit is not confirmed." if unit_review else ""

    primary_label = str(summary.get("primary_current_label") or "Max anodic current/current density")
    primary_value = str(summary.get("primary_current_value") or "Not available")
    primary_status = "pending_unit_confirmation" if unit_review else ("valid" if primary_value != "Not available" else "not_applicable")
    primary_reason = "Current/current-density unit must be confirmed before interpreting this metric." if unit_review else ""

    metric_key = primary_label.lower().replace(" ", "_").replace("/", "_")
    return {
        "onset potential": metric_status_record(onset_value, "V", onset_status, onset_reason),
        "threshold": metric_status_record(threshold_value, "", threshold_status, threshold_reason),
        metric_key: metric_status_record(primary_value, "", primary_status, primary_reason),
    }


def lsv_metric_status(
    summary: dict[str, Any],
    metric_validity: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    not_applicable_cv = metric_status_record(
        "Not applicable",
        "",
        "not_applicable",
        "This metric applies to CV peak analysis, not LSV onset analysis.",
    )
    primary_label = str(summary.get("primary_current_label") or "")
    primary_key = primary_label.lower().replace(" ", "_").replace("/", "_")
    primary_record = metric_validity.get(
        primary_key,
        metric_status_record("Not available", "", "not_applicable", "No direction-relevant current metric was available."),
    )
    if "density" in primary_label.lower():
        max_current = metric_status_record(
            "Not applicable",
            "",
            "not_applicable",
            "The uploaded/analyzed LSV quantity is current density.",
        )
        max_current_density = primary_record
    else:
        max_current = primary_record
        max_current_density = metric_status_record(
            "Not applicable",
            "",
            "not_applicable",
            "Current-density display requires current-density input or electrode-area normalization.",
        )
    return {
        "Epa": not_applicable_cv,
        "Ipa": not_applicable_cv,
        "Epc": not_applicable_cv,
        "Ipc": not_applicable_cv,
        "delta_Ep": not_applicable_cv,
        "E0_prime": not_applicable_cv,
        "peak_current_ratio": not_applicable_cv,
        "onset_potential": metric_validity.get(
            "onset potential",
            metric_status_record("Not detected", "V", "not_detected"),
        ),
        "threshold": metric_validity.get("threshold", metric_status_record("Not available", "", "not_applicable")),
        "max_current": max_current,
        "max_current_density": max_current_density,
    }


def build_cv_ai_interpretation_payload(
    *,
    filename: str,
    quick_interpretation: str,
    cv_behavior: Optional[CVBehaviorResult],
    metrics: dict[str, Optional[float]],
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    display_current_unit: str,
    current_source: str,
    electrode_area_cm2: float,
    analysis_status: str,
    parser_confidence: str,
    unit_confidence: str,
    peak_detection_confidence: str,
    warnings: Optional[list[str]] = None,
    scan_info: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, str]] = None,
    notes: str = "",
) -> dict[str, Any]:
    behavior = cv_behavior.behavior if cv_behavior is not None else "unknown"
    behavior_label = cv_behavior.label if cv_behavior is not None else "Unknown"
    epa = metrics.get("epa_V")
    epc = metrics.get("epc_V")
    delta_ep = metrics.get("delta_ep_V")
    formal_potential = (epa + epc) / 2 if epa is not None and epc is not None else None
    pair_metric_review = cv_pair_metrics_review_reason(
        metrics,
        cv_behavior.pair_confidence if cv_behavior is not None else None,
    )

    pair_metrics_valid = (
        pair_metric_review is None
        and behavior not in {"irreversible_oxidation_only", "irreversible_reduction_only", "no_reliable_peaks"}
    )
    if not pair_metrics_valid:
        delta_ep_display = "Invalid" if pair_metric_review and pair_metric_review.startswith("Invalid") else "Not applicable"
        formal_potential_display = delta_ep_display
        ratio_display = "Not meaningful" if behavior.startswith("irreversible") else delta_ep_display
    else:
        delta_ep_display = formatted_ai_float((delta_ep or 0.0) * 1000 if delta_ep is not None else None, "mV")
        formal_potential_display = formatted_ai_float(formal_potential, "V")
        ratio_display = formatted_ai_float(metrics.get("ipa_ipc_ratio"), "")

    selected_peaks = []
    if oxidation_peak is not None:
        selected_peaks.append(
            {
                "role": "Epa",
                "potential": formatted_ai_float(oxidation_peak.potential, "V"),
                "current": ai_current_metric_value(
                    metrics.get("ipa_A"),
                    display_current_unit,
                    current_source,
                    electrode_area_cm2,
                    unit_confidence,
                ),
                "scan": scan_direction_label(oxidation_peak.segment_direction),
                "confidence": oxidation_peak.confidence,
            }
        )
    if reduction_peak is not None:
        selected_peaks.append(
            {
                "role": "Epc",
                "potential": formatted_ai_float(reduction_peak.potential, "V"),
                "current": ai_current_metric_value(
                    metrics.get("ipc_A"),
                    display_current_unit,
                    current_source,
                    electrode_area_cm2,
                    unit_confidence,
                ),
                "scan": scan_direction_label(reduction_peak.segment_direction),
                "confidence": reduction_peak.confidence,
            }
        )

    oxidation_peak_confidence = peak_confidence_display(
        oxidation_peak,
        cv_behavior.rejected_oxidation_peak if cv_behavior is not None else None,
    )
    reduction_peak_confidence = peak_confidence_display(
        reduction_peak,
        cv_behavior.rejected_reduction_peak if cv_behavior is not None else None,
    )
    anodic_peak_wording = ""
    pair_metrics_statement = ""
    recommended_next_steps_guidance = ""
    if behavior == "irreversible_oxidation_only":
        if oxidation_peak is not None and str(oxidation_peak.confidence).lower() == "high":
            anodic_peak_wording = "Describe Epa as a reliable anodic oxidation peak."
        else:
            anodic_peak_wording = (
                f"Describe Epa as a selected candidate anodic peak with "
                f"{format_confidence_label(oxidation_peak.confidence if oxidation_peak else oxidation_peak_confidence)} confidence."
            )
        pair_metrics_statement = (
            "The anodic oxidation feature may be useful, but reversible-pair metrics are not applicable "
            "because no reliable cathodic return peak exists."
        )
        recommended_next_steps_guidance = (
            "Recommended next steps should include confirming the reference electrode, confirming scan rate, "
            "repeating under controlled conditions if reversibility is the goal, and considering a wider "
            "potential window or reverse-scan behavior only when scientifically appropriate."
        )
    elif behavior == "irreversible_reduction_only":
        pair_metrics_statement = (
            "The cathodic reduction feature may be useful, but reversible-pair metrics are not applicable "
            "because no reliable anodic return peak exists."
        )

    payload_warnings = compact_text_list((warnings or []) + (cv_behavior.messages if cv_behavior else []))
    if pair_metric_review:
        payload_warnings.append(pair_metric_review)

    scan_info = scan_info or {}
    available_metadata = available_metadata_from_context(
        metadata=metadata,
        scan_rate=scan_info.get("scan_rate"),
        electrode_area=f"{electrode_area_cm2:g} cm\u00b2" if current_source == "Current density" else None,
        reference_electrode=scan_info.get("reference_electrode"),
        reference_electrode_source=str(scan_info.get("reference_electrode_source") or ""),
    )
    required_metadata = ["scan_rate", "reference_electrode"]
    if current_source == "Current density":
        required_metadata.append("electrode_area")
    missing_metadata = missing_metadata_fields(available_metadata, required_metadata)
    unit_confirmation_required = pending_current_metric(unit_confidence)
    scan_rate_available = "scan_rate" in available_metadata
    requires_review = analysis_status != "Passed" or bool(payload_warnings) or unit_confirmation_required
    metric_validity = cv_metric_validity(
        metrics,
        display_current_unit=display_current_unit,
        current_source=current_source,
        electrode_area_cm2=electrode_area_cm2,
        unit_confidence=unit_confidence,
        cv_behavior=cv_behavior,
        pair_metric_review=pair_metric_review,
        pair_metrics_valid=pair_metrics_valid,
    )
    metadata_status = metadata_status_from_context(
        metadata=metadata,
        scan_rate=scan_info.get("scan_rate"),
        scan_rate_source=str(scan_info.get("scan_rate_source") or ""),
        electrode_area=f"{electrode_area_cm2:g} cm²" if current_source == "Current density" else None,
        electrode_area_source="manual" if current_source == "Current density" else "",
        reference_electrode=scan_info.get("reference_electrode"),
        reference_electrode_source=str(scan_info.get("reference_electrode_source") or ""),
        cycle_count=scan_info.get("cycle_count"),
        cycle_assignment_uncertain=bool(scan_info.get("cycle_assignment_uncertain", False)),
        scan_mode=scan_info.get("scan_mode"),
    )
    metric_status = cv_metric_status(metric_validity)
    metadata_completeness = metadata_completeness_status_from_fields(missing_metadata, metadata_status)
    unit_status = unit_status_label(unit_confidence, unit_confirmation_required)
    invalid_reversible_pair = bool(pair_metric_review and pair_metric_review.startswith("Invalid"))
    candidate_peak_language_required = behavior in {"noisy_ambiguous", "no_reliable_peaks"} or invalid_reversible_pair
    interpretation_guidance = {
        "peak_wording": (
            anodic_peak_wording
            if anodic_peak_wording
            else (
            "Use candidate extrema/candidate peaks wording. Do not call the selected oxidation/reduction "
            "features valid peaks for reversible-pair analysis."
            if candidate_peak_language_required
            else "Selected primary peaks can be described as Epa/Epc only when metric status and pair confidence support it."
            )
        ),
        "pair_metrics_statement": (
            pair_metrics_statement
            if pair_metrics_statement
            else (
            "Candidate peaks were detected, but the selected pair is not valid for reversible-pair metrics."
            if invalid_reversible_pair
            else ""
            )
        ),
        "recommended_next_steps": recommended_next_steps_guidance,
    }

    return {
        "payload_version": "voltscope_ai_interpretation_v1",
        "technique": "CV",
        "filename": filename,
        "analysis_status": analysis_status,
        "overall_quality": analysis_status,
        "cv_behavior": behavior_label,
        "cv_behavior_code": behavior,
        "behavior_classification": behavior,
        "quick_interpretation": quick_interpretation,
        "key_metrics": {
            "Epa": formatted_ai_float(epa, "V"),
            "Ipa": ai_current_metric_value(
                metrics.get("ipa_A"),
                display_current_unit,
                current_source,
                electrode_area_cm2,
                unit_confidence,
            ),
            "Epc": formatted_ai_float(epc, "V") if epc is not None else "Not detected",
            "Ipc": ai_current_metric_value(
                metrics.get("ipc_A"),
                display_current_unit,
                current_source,
                electrode_area_cm2,
                unit_confidence,
            ),
            "\u0394Ep": delta_ep_display,
            "E\u00b0\u2032": formal_potential_display,
            "|Ipa/Ipc|": ratio_display,
        },
        "metric_validity": metric_validity,
        "metric_status": metric_status,
        "interpretation_guidance": interpretation_guidance,
        "selected_peaks": selected_peaks,
        "available_metadata": available_metadata,
        "missing_metadata": missing_metadata,
        "metadata_status": metadata_status,
        "status_summary": {
            "analysis_status": analysis_status,
            "metadata_status": metadata_completeness,
            "unit_status": unit_status,
            "ai_interpretation_status": "not_generated",
        },
        "flags": {
            "technique": "CV",
            "analysis_status": analysis_status,
            "overall_quality": analysis_status,
            "unit_confidence": unit_confidence,
            "unit_status": unit_status,
            "parser_confidence": parser_confidence,
            "peak_detection_confidence": peak_detection_confidence,
            "metadata_completeness": metadata_completeness,
            "oxidation_peak_confidence": oxidation_peak_confidence,
            "reduction_peak_confidence": reduction_peak_confidence,
            "cathodic_return_peak_confidence": reduction_peak_confidence if behavior == "irreversible_oxidation_only" else "",
            "anodic_return_peak_confidence": oxidation_peak_confidence if behavior == "irreversible_reduction_only" else "",
            "pair_confidence": cv_behavior.pair_confidence if cv_behavior is not None else "Not available",
            "reversible_pair_confidence": cv_behavior.pair_confidence if cv_behavior is not None else "Not available",
            "behavior_classification": behavior,
            "reversible_pair_metrics_valid": pair_metrics_valid,
            "candidate_peak_language_required": candidate_peak_language_required,
            "requires_review": requires_review,
            "unit_confirmation_required": unit_confirmation_required,
            "scan_rate_available": scan_rate_available,
            "reference_electrode_source": str(scan_info.get("reference_electrode_source") or "not_available"),
        },
        "confidence": {
            "parser": parser_confidence,
            "unit": unit_confidence,
            "peak_detection": peak_detection_confidence,
            "oxidation_peak": oxidation_peak_confidence,
            "reduction_peak": reduction_peak_confidence,
            "cathodic_return_peak": reduction_peak_confidence if behavior == "irreversible_oxidation_only" else "Not applicable",
            "anodic_return_peak": oxidation_peak_confidence if behavior == "irreversible_reduction_only" else "Not applicable",
            "pair": cv_behavior.pair_confidence if cv_behavior is not None else "Not available",
            "reversible_pair": cv_behavior.pair_confidence if cv_behavior is not None else "Not available",
        },
        "context": {
            "current_source": current_source,
            "display_current_unit": display_unit_label(display_current_unit),
            "electrode_area_cm2": electrode_area_cm2 if current_source == "Current density" else None,
            **scan_info,
        },
        "warnings": payload_warnings,
        "experiment_notes": notes.strip(),
    }


def build_lsv_ai_interpretation_payload(
    *,
    filename: str,
    quick_interpretation: str,
    summary: Optional[dict[str, Any]],
    warnings: Optional[list[str]] = None,
    notes: str = "",
) -> dict[str, Any]:
    summary = summary or {}
    summary_text = " ".join(str(value).lower() for value in summary.values())
    unit_confirmation_required = (
        str(summary.get("unit_status", "")).lower() == "unconfirmed"
        or "unit unconfirmed" in summary_text
        or "pending unit confirmation" in summary_text
    )
    reference_source = str(summary.get("reference_electrode_source") or "not_available")
    available_metadata = available_metadata_from_context(
        scan_rate=summary.get("scan_rate"),
        electrode_area=summary.get("electrode_area_metadata") or summary.get("electrode_area"),
        reference_electrode=summary.get("reference_electrode"),
        reference_electrode_source=reference_source,
    )
    required_metadata = ["scan_rate", "reference_electrode"]
    if str(summary.get("primary_current_label", "")).lower().find("density") >= 0:
        required_metadata.append("electrode_area")
    summary_missing = str(summary.get("missing_metadata") or "").strip()
    if summary_missing and summary_missing != "None":
        missing_metadata = [
            part.strip().replace("confirmed ", "").replace(" ", "_")
            for part in summary_missing.split(",")
            if part.strip()
        ]
    else:
        missing_metadata = missing_metadata_fields(available_metadata, required_metadata)
    metric_validity = lsv_metric_validity(summary)
    metadata_status = metadata_status_from_context(
        scan_rate=summary.get("scan_rate"),
        scan_rate_source=str(summary.get("scan_rate_source") or ""),
        electrode_area=summary.get("electrode_area_metadata") or summary.get("electrode_area"),
        electrode_area_source="metadata" if metadata_value_is_available(summary.get("electrode_area_metadata")) else "",
        reference_electrode=summary.get("reference_electrode"),
        reference_electrode_source=reference_source,
    )
    metric_status = lsv_metric_status(summary, metric_validity)
    metadata_completeness = metadata_completeness_status_from_fields(missing_metadata, metadata_status)
    unit_status = unit_status_label("Low" if unit_confirmation_required else "Confirmed", unit_confirmation_required)
    requires_review = (
        str(summary.get("status") or "") != "Passed"
        or unit_confirmation_required
        or str(summary.get("metadata_completeness") or "") == "Review recommended"
    )
    compact_summary = {
        key: value
        for key, value in summary.items()
        if key
        in {
            "status",
            "metadata_completeness",
            "missing_metadata",
            "onset_label",
            "onset_potential",
            "onset_reliability",
            "threshold",
            "threshold_magnitude",
            "signed_threshold",
            "threshold_source",
            "threshold_rule",
            "direction",
            "primary_current_label",
            "primary_current_value",
            "primary_potential_label",
            "primary_potential_value",
            "min_label",
            "min_value",
            "potential_range",
            "scan_rate",
            "scan_rate_source",
            "reference_electrode",
            "reference_electrode_source",
            "electrode_area",
            "electrode_area_metadata",
            "normalization_status",
            "normalization_note",
            "method_note",
            "onset_detection_signal",
            "review_note",
            "scan_rate_note",
        }
    }
    interpretation_guidance = {
        "onset_wording": "Describe LSV onset as an operational threshold-crossing onset estimated using the selected threshold rule.",
        "report_with": (
            "Report onset alongside threshold value, reference electrode, scan rate, electrode area, smoothing/onset "
            "detection signal, and normalization status when available."
        ),
        "normalization_status": summary.get("normalization_status") or summary.get("normalization_note") or "",
        "suggested_lab_notebook_note": (
            f"LSV showed a {str(summary.get('direction') or '').lower()} onset at "
            f"{summary.get('onset_potential', 'not detected')} using a {summary.get('threshold', 'selected')} "
            f"threshold. Scan rate was {summary.get('scan_rate', 'not available')}, electrode area was "
            f"{summary.get('electrode_area_metadata') or summary.get('electrode_area') or 'not available'}. "
            "Onset was detected from a sustained threshold crossing."
            if summary.get("status") == "Passed"
            else ""
        ),
    }
    return {
        "payload_version": "voltscope_ai_interpretation_v1",
        "technique": "LSV",
        "filename": filename,
        "analysis_status": summary.get("status", "Not available"),
        "overall_quality": summary.get("status", "Not available"),
        "quick_interpretation": quick_interpretation,
        "key_metrics": compact_summary,
        "metric_validity": metric_validity,
        "metric_status": metric_status,
        "interpretation_guidance": interpretation_guidance,
        "available_metadata": available_metadata,
        "missing_metadata": missing_metadata,
        "metadata_status": metadata_status,
        "status_summary": {
            "analysis_status": summary.get("status", "Not available"),
            "metadata_status": metadata_completeness,
            "unit_status": unit_status,
            "ai_interpretation_status": "not_generated",
        },
        "flags": {
            "technique": "LSV",
            "analysis_status": summary.get("status", "Not available"),
            "overall_quality": summary.get("status", "Not available"),
            "unit_confidence": "Low" if unit_confirmation_required else "High",
            "unit_status": unit_status,
            "parser_confidence": "High",
            "peak_detection_confidence": "Not applicable",
            "pair_confidence": "Not applicable",
            "metadata_completeness": metadata_completeness,
            "behavior_classification": str(summary.get("direction") or "LSV"),
            "requires_review": requires_review,
            "unit_confirmation_required": unit_confirmation_required,
            "scan_rate_available": "scan_rate" in available_metadata,
            "reference_electrode_source": reference_source,
        },
        "warnings": compact_text_list(
            (warnings or [])
            + [
                summary.get("review_note", ""),
                summary.get("scan_rate_note", ""),
                f"Missing metadata: {summary.get('missing_metadata')}"
                if summary.get("missing_metadata") not in {None, "", "None"}
                else "",
            ]
        ),
        "experiment_notes": notes.strip(),
    }


def build_ai_interpretation_prompt(payload: dict[str, Any]) -> str:
    safe_payload = to_jsonable(payload)
    payload_json = json.dumps(safe_payload, indent=2, sort_keys=True, ensure_ascii=False)
    return (
        "You are VoltScope AI, an electrochemistry interpretation assistant.\n\n"
        "Use only the supplied JSON payload. Do not invent values, file contents, experimental details, "
        "mechanisms, metadata, or certainty that are not present. The payload is intentionally structured: "
        "`metadata_status` gives provenance for each metadata field, `available_metadata` contains only "
        "metadata actually available, `missing_metadata` contains fields that are truly absent, "
        "`metric_status` and `metric_validity` give the status of each important metric, and `flags` "
        "summarizes parser/unit/analysis quality. `status_summary` separates computational analysis status "
        "from metadata completeness and unit status.\n\n"
        "Strict grounding rules:\n"
        "1. Only use values provided in the payload.\n"
        "2. Never infer missing metadata or invent metadata. Do not say a value was measured or provided if "
        "`metadata_status` marks it as inferred, defaulted, or manual.\n"
        "3. If `metadata_status.status` is missing, mention the field as missing only when relevant. If it is "
        "inferred, say it was inferred and mention `source`. If it is defaulted, say it was defaulted and "
        "should be confirmed. If it is manual, identify it as user-provided/manual.\n"
        "4. If a metric status is invalid, not_applicable, pending_unit_confirmation, or not_detected, do not "
        "interpret that metric as a valid quantitative result.\n"
        "5. Treat `analysis_status: Passed` as a valid computation only for the metrics marked valid. If "
        "`status_summary.metadata_status` is Incomplete, do not imply the computation failed; instead state "
        "that deeper literature comparison or publication-ready interpretation needs the missing metadata.\n"
        "6. For reversible-like CVs, interpret Epa, Epc, Ipa, Ipc, ΔEp, E°′/E°, and |Ipa/Ipc| only when their "
        "metric status is valid. Describe the behavior as reversible-like or consistent with reversible-like "
        "behavior, not definitively reversible.\n"
        "7. For noisy or ambiguous CVs, emphasize uncertainty, candidate-peak review, smoothing, and baseline "
        "correction before quantitative interpretation. If `flags.candidate_peak_language_required` is true, "
        "describe selected features as candidate extrema or candidate peaks, not valid oxidation/reduction peaks. "
        "Only call Epa/Epc valid when pair confidence is high and reversible-pair metric statuses are valid. "
        "If `interpretation_guidance.pair_metrics_statement` is provided, include that exact idea in the "
        "`summary` section.\n"
        "8. For irreversible, oxidation-only, or reduction-only CVs, do not discuss ΔEp, E°′/E°, or |Ipa/Ipc| "
        "as meaningful reversible-pair metrics. For oxidation-only CVs, distinguish the anodic oxidation "
        "feature from reversible-pair analysis: the anodic peak may be useful as an oxidation feature, but "
        "pair metrics are not applicable when no reliable cathodic return peak exists. Do not use generic "
        "`valid anodic peak` wording unless the oxidation peak confidence is high; otherwise use wording like "
        "`selected candidate anodic peak with medium-high confidence`. Recommended next steps should include "
        "confirming reference electrode and scan rate, repeating under controlled conditions if evaluating "
        "reversibility, and considering a wider potential window or reverse-scan behavior only when scientifically "
        "appropriate.\n"
        "9. For clean reversible-like CVs, discuss reversibility cautiously. You may mention that ideal "
        "one-electron reversible ΔEp is about 59 mV at room temperature only as contextual background.\n"
        "10. For clean CVs with no metadata rows, clearly separate valid CV metrics from missing experimental "
        "metadata. If scan rate is inferred from time/potential columns, say so. If electrode area, electrolyte, "
        "concentration, working electrode, or counter electrode are missing, explain that current-density "
        "normalization or deeper mechanistic interpretation is limited unless those values are supplied.\n"
        "11. For LSV, focus on the operational threshold-crossing onset potential, threshold value/rule, "
        "reference electrode, scan rate, electrode area, smoothing/onset-detection signal, and normalization "
        "status. Say `onset was estimated using the selected threshold rule`; do not call it the true onset "
        "potential. When metadata are available, use them explicitly, for example `onset potential = 0.397 V "
        "vs SCE`. For passed LSVs, the suggested lab notebook note should concisely include onset potential, "
        "threshold, scan rate, electrode area, and that onset was detected from a sustained threshold crossing.\n"
        "12. For unit-unconfirmed LSVs, clearly state that current/current-density dependent metrics may be "
        "scaled incorrectly until the unit is confirmed.\n"
        "13. Always include a short Suggested lab notebook note.\n\n"
        "Never overstate conclusions. Do not claim diffusion coefficients, rate constants, concentration, "
        "mechanism, or electrode material effects unless those inputs are present in the payload. Use wording "
        "such as suggests, is consistent with, and should be interpreted cautiously where appropriate. Clearly "
        "separate measured/computed VoltScope metrics from scientific interpretation. Always remind the user "
        "that the interpretation should be reviewed by the researcher.\n\n"
        "Return a JSON object with exactly these keys when possible:\n"
        "- summary\n"
        "- key_interpretation\n"
        "- data_quality_and_limitations\n"
        "- recommended_next_steps\n"
        "- suggested_lab_notebook_note\n"
        "- confidence\n"
        "- requires_researcher_review\n\n"
        "Each value should be concise researcher-facing text, except requires_researcher_review should be a "
        "boolean when possible. If JSON output is not possible, use these plain-text section headings: "
        "Summary; Key electrochemical interpretation; Data quality and limitations; Recommended next steps; "
        "Suggested lab notebook note.\n\n"
        "Keep it practical, scientifically cautious, and grounded in the metrics provided.\n\n"
        f"Payload:\n{payload_json}"
    )


def ai_interpretation_signature(payload: dict[str, Any]) -> str:
    serialized = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ai_analysis_settings_signature(payload: dict[str, Any]) -> str:
    settings_payload = {
        "technique": payload.get("technique"),
        "filename": payload.get("filename"),
        "flags": payload.get("flags"),
        "context": payload.get("context"),
        "status_summary": payload.get("status_summary"),
        "metadata_status": payload.get("metadata_status"),
        "metric_status": payload.get("metric_status"),
        "warnings": payload.get("warnings"),
    }
    serialized = json.dumps(to_jsonable(settings_payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ai_interpretation_is_stale(saved_state: Optional[dict[str, Any]], payload: dict[str, Any]) -> bool:
    if not saved_state:
        return False
    return saved_state.get("signature") != ai_interpretation_signature(payload)


def parse_ai_interpretation_sections(content: str) -> tuple[dict[str, Any], bool]:
    text = str(content or "").strip()
    if not text:
        return {}, False
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}, False
    if not isinstance(parsed, dict):
        return {}, False
    section_keys = {
        "summary",
        "key_interpretation",
        "data_quality_and_limitations",
        "recommended_next_steps",
        "suggested_lab_notebook_note",
        "confidence",
        "requires_researcher_review",
    }
    if not any(key in parsed for key in section_keys):
        return {}, False
    return parsed, True


def interpretation_limitations_from_payload(payload: dict[str, Any]) -> list[str]:
    metadata_status = payload.get("metadata_status") if isinstance(payload.get("metadata_status"), dict) else {}
    flags = payload.get("flags") if isinstance(payload.get("flags"), dict) else {}
    missing_metadata = set(payload.get("missing_metadata") or [])
    metric_status = payload.get("metric_status") if isinstance(payload.get("metric_status"), dict) else {}
    behavior = str(flags.get("behavior_classification") or payload.get("behavior_classification") or "").lower()
    technique = str(payload.get("technique") or flags.get("technique") or "").upper()
    limitations: list[str] = []

    reference_status = metadata_status.get("reference_electrode", {}) if isinstance(metadata_status, dict) else {}
    if reference_status.get("status") == "missing" or "reference_electrode" in missing_metadata:
        limitations.append(
            "Reference electrode missing: absolute potentials cannot be compared directly with literature values."
        )
    elif "confirmed_reference_electrode" in missing_metadata:
        limitations.append(
            "Reference electrode is defaulted or unconfirmed: confirm before comparing absolute potentials."
        )
    elif reference_status.get("status") == "defaulted":
        limitations.append(
            "Reference electrode defaulted: confirm before comparing absolute potentials."
        )

    area_status = metadata_status.get("electrode_area", {}) if isinstance(metadata_status, dict) else {}
    if area_status.get("status") == "missing" or "electrode_area" in missing_metadata:
        limitations.append(
            "Electrode area missing: current-density comparisons are limited."
        )

    if flags.get("unit_confirmation_required"):
        limitations.append("Unit confirmation required: current-dependent metrics may be scaled incorrectly.")
    if flags.get("requires_review"):
        if behavior == "noisy_ambiguous":
            limitations.append(
                "Analysis requires review: noisy or ambiguous peak selection should be checked before quantitative use."
            )
        elif behavior == "irreversible_oxidation_only":
            limitations.append(
                "Irreversible oxidation-only trace: no reliable cathodic return peak was detected, so reversible-pair metrics are not applicable."
            )
        elif behavior == "irreversible_reduction_only":
            limitations.append(
                "Irreversible reduction-only trace: no reliable anodic return peak was detected, so reversible-pair metrics are not applicable."
            )
        else:
            limitations.append(
                "Analysis requires review: verify selected peaks, units, and warnings before relying on quantitative interpretation."
            )

    invalid_pair_metrics = [
        label
        for label in ("delta_Ep", "E0_prime", "peak_current_ratio")
        if isinstance(metric_status.get(label), dict)
        and metric_status[label].get("status") in {"invalid", "not_applicable", "pending_unit_confirmation", "not_detected"}
    ]
    if technique == "CV" and invalid_pair_metrics and behavior in {"noisy_ambiguous", "irreversible_oxidation_only", "irreversible_reduction_only"}:
        limitations.append(
            "Reversible-pair metrics are limited or not applicable for this trace; use the metric status fields before interpreting ΔEp, E°′, or |Ipa/Ipc|."
        )

    scan_status = metadata_status.get("scan_rate", {}) if isinstance(metadata_status, dict) else {}
    if scan_status.get("status") == "missing" or "scan_rate" in missing_metadata:
        limitations.append(
            "Scan rate missing: scan-rate-dependent interpretation is qualitative until scan rate is added or confirmed."
        )

    limitations.append(
        "AI uses computed metrics only; raw arrays are not sent."
    )
    return compact_text_list(limitations)


def render_interpretation_limitations(payload: dict[str, Any]) -> None:
    limitations = interpretation_limitations_from_payload(payload)
    if not limitations:
        return
    items = "".join(f"<li>{escape_html(item)}</li>" for item in limitations)
    st.markdown(
        f"""
        <div class="interpretation-panel">
            <h4>Interpretation limitations</h4>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_interpretation_metadata(saved_state: dict[str, Any], *, stale: bool) -> None:
    if not saved_state:
        return
    status_text = "Stale AI interpretation" if stale else "Current interpretation"
    payload_hash = str(saved_state.get("payload_hash") or saved_state.get("signature") or "")
    settings_hash = str(saved_state.get("analysis_settings_hash") or "")
    metadata_items = [
        f"Status: {status_text}",
        f"Generated: {saved_state.get('generated_at', 'Not available')}",
        f"Model: {saved_state.get('model', 'Not available')}",
        f"Payload version: {saved_state.get('payload_version', 'Not available')}",
    ]
    if payload_hash:
        metadata_items.append(f"Payload hash: {payload_hash[:12]}")
    if settings_hash:
        metadata_items.append(f"Settings hash: {settings_hash[:12]}")
    st.caption(" · ".join(metadata_items))


def researcher_review_label(value: Any) -> str:
    normalized = str(value).strip().lower()
    if value is True or normalized in {"true", "required", "yes", "1"}:
        return "Researcher review: Required before quantitative use"
    return "Researcher review: Recommended"


def render_structured_ai_interpretation(sections: dict[str, Any]) -> None:
    section_labels = [
        ("summary", "Summary"),
        ("key_interpretation", "Key electrochemical interpretation"),
        ("data_quality_and_limitations", "Data quality and limitations"),
        ("recommended_next_steps", "Recommended next steps"),
        ("suggested_lab_notebook_note", "Suggested lab notebook note"),
    ]
    for key, label in section_labels:
        value = sections.get(key)
        if value in {None, ""}:
            continue
        if isinstance(value, list):
            body = "<ul>" + "".join(f"<li>{escape_html(item)}</li>" for item in value) + "</ul>"
        else:
            body = f"<p>{escape_html(value)}</p>"
        st.markdown(
            f"""
            <div class="interpretation-panel">
                <h4>{escape_html(label)}</h4>
                {body}
            </div>
            """,
            unsafe_allow_html=True,
        )
    footer_parts = []
    if sections.get("confidence"):
        footer_parts.append(f"AI confidence: {sections.get('confidence')}")
    if "requires_researcher_review" in sections:
        footer_parts.append(researcher_review_label(sections.get("requires_researcher_review")))
    if footer_parts:
        st.caption(" · ".join(str(part) for part in footer_parts))


_AI_INTERPRETATION_BACKEND: Any = None


def load_ai_interpretation_backend() -> Any:
    global _AI_INTERPRETATION_BACKEND
    if _AI_INTERPRETATION_BACKEND is not None:
        return _AI_INTERPRETATION_BACKEND

    module_path = Path(__file__).with_name("ai_interpretation.py")
    spec = importlib.util.spec_from_file_location("voltscope_ai_interpretation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AI interpretation backend from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _AI_INTERPRETATION_BACKEND = module
    return module


def generate_detailed_ai_interpretation(
    payload: dict[str, Any],
    environ: Optional[dict[str, str]] = None,
    timeout_s: int = 45,
) -> AIInterpretationResponse:
    prompt = build_ai_interpretation_prompt(payload)
    try:
        backend = load_ai_interpretation_backend()
        streamlit_secrets = None if environ is not None else st.secrets
        result = backend.request_openai_interpretation(
            prompt,
            streamlit_secrets=streamlit_secrets,
            environ=environ,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return AIInterpretationResponse(
            content="",
            error=f"AI interpretation request failed: {exc}",
            configured=True,
            model="Not available",
        )

    return AIInterpretationResponse(
        content=str(getattr(result, "content", "") or ""),
        error=str(getattr(result, "error", "") or ""),
        configured=bool(getattr(result, "configured", False)),
        model=str(getattr(result, "model", "") or "Not available"),
    )


def ai_interpretation_store(
    project_workspace: Optional[dict[str, Any]],
    experiment_id: Optional[str],
) -> Optional[dict[str, Any]]:
    if project_workspace is None or not experiment_id:
        return None
    experiment = project_workspace.get("experiments", {}).get(experiment_id)
    if not isinstance(experiment, dict):
        return None
    return experiment.setdefault("ai_interpretations", {})


def render_interpretation_panel(
    text: str,
    key: str,
    ai_payload: Optional[dict[str, Any]] = None,
    project_workspace: Optional[dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    interpretation_id: Optional[str] = None,
) -> None:
    st.markdown(
        f"""
        <div class="interpretation-panel">
            <h4>Quick interpretation</h4>
            <p>{escape_html(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if ai_payload is None:
        st.button("Generate detailed AI interpretation", disabled=True, key=key)
        return

    with st.expander("AI interpretation payload", expanded=False):
        st.caption("Raw uploaded arrays are excluded. This is the exact computed payload sent to the AI backend.")
        st.json(to_jsonable(ai_payload))

    interpretation_id = interpretation_id or key
    signature = ai_interpretation_signature(ai_payload)
    store = ai_interpretation_store(project_workspace, experiment_id)
    session_key = f"ai_interpretation_state::{interpretation_id}"
    if store is not None:
        saved_state = store.get(interpretation_id, {})
    else:
        saved_state = st.session_state.get(session_key, {})

    stale = ai_interpretation_is_stale(saved_state, ai_payload)
    has_previous = bool(saved_state.get("content") or saved_state.get("error"))
    if stale:
        button_label = "Regenerate interpretation using current metrics"
    else:
        button_label = "Regenerate interpretation" if has_previous else "Generate detailed AI interpretation"

    render_interpretation_limitations(ai_payload)

    if stale:
        st.warning(
            "This interpretation may be stale because analysis settings, selected peaks, units, metadata, "
            "or processing options changed after it was generated. Regenerate the interpretation to update it."
        )
    elif has_previous:
        st.caption("Current interpretation")

    if st.button(button_label, key=key):
        response = generate_detailed_ai_interpretation(ai_payload)
        sections, structured = parse_ai_interpretation_sections(response.content)
        saved_state = {
            "signature": signature,
            "payload_hash": signature,
            "analysis_settings_hash": ai_analysis_settings_signature(ai_payload),
            "payload_version": ai_payload.get("payload_version", "Not available"),
            "generated_at": now_timestamp(),
            "content": response.content,
            "structured_sections": sections if structured else {},
            "structured_response": structured,
            "error": response.error,
            "backend_configured": response.configured,
            "model": response.model,
            "payload": to_jsonable(ai_payload),
        }
        if store is not None:
            store[interpretation_id] = saved_state
            if project_workspace is not None and experiment_id:
                project_workspace["experiments"][experiment_id]["updated_at"] = now_timestamp()
            save_project_workspaces()
        else:
            st.session_state[session_key] = saved_state
        stale = False

    if saved_state.get("content"):
        st.markdown("#### Stale AI interpretation" if stale else "#### Detailed AI interpretation")
        render_ai_interpretation_metadata(saved_state, stale=stale)
        sections = saved_state.get("structured_sections") or {}
        if sections:
            render_structured_ai_interpretation(sections)
        else:
            parsed_sections, structured = parse_ai_interpretation_sections(str(saved_state["content"]))
            if structured:
                render_structured_ai_interpretation(parsed_sections)
            else:
                st.markdown(str(saved_state["content"]))
    elif saved_state.get("error"):
        render_ai_interpretation_metadata(saved_state, stale=stale)
        st.info(str(saved_state["error"]))

    st.caption(
        "AI interpretation uses only computed VoltScope metrics, warnings, metadata flags, and experiment notes. "
        "It does not inspect raw uploaded data and should be reviewed by the researcher."
    )


def cv_interpretation_text(
    cv_behavior: Optional[CVBehaviorResult],
    metrics: dict[str, Optional[float]],
    unit_confidence: str,
    metadata_caution: bool = False,
) -> str:
    def with_metadata_caution(message: str) -> str:
        if not metadata_caution:
            return message
        return (
            message
            + " Because some experimental metadata are missing, treat mechanistic interpretation as qualitative "
            "unless conditions are confirmed."
        )

    behavior = cv_behavior.behavior if cv_behavior is not None else "reversible_like"
    label = cv_behavior.label if cv_behavior is not None else "reversible-like"
    if pending_current_metric(unit_confidence):
        return (
            "Potential-based peak positions can be reviewed, but current-dependent metrics are paused "
            "until the current unit is confirmed."
        )
    if behavior == "irreversible_oxidation_only":
        return (
            "This CV appears oxidation-dominant or irreversible under the selected peak settings. "
            "The anodic oxidation peak can be used as an oxidation feature, but a reliable cathodic return "
            "peak was not detected, so \u0394Ep, E\u00b0\u2032, and |Ipa/Ipc| are not applicable as reversible-pair "
            "metrics."
        )
    if behavior == "irreversible_reduction_only":
        return (
            "This CV appears reduction-dominant or irreversible under the selected peak settings. "
            "A reliable anodic return peak was not detected, so \u0394Ep, E\u00b0\u2032, and |Ipa/Ipc| "
            "should not be treated as reversible-pair metrics."
        )
    if behavior == "multiple_redox_couples":
        return (
            "Multiple redox couples were detected. The displayed metrics are calculated for the selected "
            "couple, so pair assignment should be reviewed before interpretation."
        )
    if behavior in {"noisy_ambiguous", "no_reliable_peaks"}:
        pair_metric_review = cv_pair_metrics_review_reason(
            metrics,
            cv_behavior.pair_confidence if cv_behavior is not None else None,
        )
        if pair_metric_review and pair_metric_review.startswith("Invalid"):
            return (
                "Candidate peaks were detected, but the selected pair is not valid for reversible-pair metrics. "
                "This CV is flagged as noisy or ambiguous, so review candidate peaks, smoothing, and baseline "
                "correction before using peak currents quantitatively."
            )
        return (
            "This CV is flagged as noisy or ambiguous. Peak selection may be sensitive to noise or baseline shape, "
            "so review candidate peaks, smoothing, and baseline correction before using peak currents quantitatively."
        )
    delta_ep = metrics.get("delta_ep_V")
    ratio = metrics.get("ipa_ipc_ratio")
    if delta_ep is not None and ratio is not None:
        delta_ep_mv = delta_ep * 1000
        if 45 <= delta_ep_mv <= 75:
            delta_context = "The peak separation is close to the ideal one-electron reversible value."
        elif delta_ep_mv > 90:
            delta_context = (
                "The peak separation is larger than the ideal one-electron reversible case and may reflect "
                "kinetic limitations, resistance, electrode condition, or scan-rate effects."
            )
        else:
            delta_context = "The peak separation should be interpreted alongside the experimental conditions."
        return with_metadata_caution(
            f"This CV appears {label}. The peak current ratio is {ratio:.4g}, and the peak separation "
            f"is {delta_ep_mv:.4g} mV. {delta_context} Interpret \u0394Ep alongside scan rate, electrolyte "
            "conditions, uncompensated resistance, and electrode preparation."
        )
    return with_metadata_caution(
        "The selected CV peaks are shown above. Review the peak table and candidate peaks before final interpretation."
    )


def lsv_interpretation_text(summary: Optional[dict[str, str]]) -> str:
    if not summary:
        return "LSV interpretation will appear after the selected file is analyzed."
    if summary.get("status") not in {"Passed", "Review needed"}:
        return summary.get("review_note") or "No reliable LSV onset was detected for this file."
    onset = summary.get("onset_potential", "not detected")
    direction = str(summary.get("direction", "LSV")).lower()
    reference = summary.get("reference_electrode", "the selected reference electrode")
    threshold = summary.get("threshold_magnitude") or summary.get("threshold", "the selected threshold")
    scan_rate = summary.get("scan_rate", "Not available")
    electrode_area = summary.get("electrode_area_metadata") or summary.get("electrode_area") or "Not detected"
    normalization_status = str(summary.get("normalization_status") or summary.get("normalization_note") or "").strip()
    smoothing_context = summary.get("onset_detection_signal", "Not available")
    metadata_incomplete = summary.get("metadata_completeness") in {"Incomplete", "Review recommended"}
    missing_metadata = str(summary.get("missing_metadata") or "")
    metadata_note = ""
    if metadata_incomplete:
        missing_parts = []
        if "scan rate" in missing_metadata:
            missing_parts.append("scan rate is unavailable")
        if "confirmed reference electrode" in missing_metadata:
            missing_parts.append("the reference electrode is currently defaulted")
        if "electrode area" in missing_metadata:
            missing_parts.append("electrode area metadata is missing")
        if missing_parts:
            metadata_note = (
                " The onset calculation passed, but "
                + " and ".join(missing_parts)
                + ", so add or confirm metadata before treating the record as publication-ready."
            )
    if "unit unconfirmed" in " ".join(str(value).lower() for value in summary.values()):
        return (
            "This LSV was parsed, but current/current-density units require confirmation. "
            "Current-dependent metrics may be incorrectly scaled until the unit is confirmed."
        )
    if summary.get("onset_potential", "Not detected") == "Not detected":
        return summary.get("review_note") or "No reliable sustained threshold crossing was detected."
    if direction == "cathodic":
        return (
            f"This LSV shows a cathodic onset at {onset} with a threshold of {threshold}. "
            "The onset was estimated using the selected threshold rule from a sustained threshold crossing. "
            f"Report it with the reference electrode ({reference}), scan rate ({scan_rate}), electrode area "
            f"({electrode_area}), onset-detection signal ({smoothing_context}), and normalization status"
            f"{': ' + normalization_status if normalization_status else ''}.{metadata_note}"
        )
    return (
        f"This LSV shows an anodic onset at {onset} with a threshold of {threshold}. "
        "The onset was estimated using the selected threshold rule from a sustained threshold crossing. "
        f"Report it with the reference electrode ({reference}), scan rate ({scan_rate}), electrode area "
        f"({electrode_area}), onset-detection signal ({smoothing_context}), and normalization status"
        f"{': ' + normalization_status if normalization_status else ''}.{metadata_note}"
    )


def compact_suggested_current_unit(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "Not needed":
        return "Not needed"
    return text.replace(" based on value magnitude", "").strip()


def render_unit_review_panel(
    *,
    current_col: str,
    current_col_label: str,
    suggested_unit: str,
    current_source: str = "Current",
    current_unit_warning: str,
    confirmed_current_unit_key: str,
    selectbox_key: str,
    button_key: str,
    confirmed_unit_type_key: Optional[str] = None,
    unit_confirmation_source_key: Optional[str] = None,
) -> None:
    suggested_display_unit = current_unit_label_for_summary(suggested_unit, current_source)
    suggested_label = display_unit_label(suggested_display_unit)
    explanation = current_unit_review_explanation(current_col_label, suggested_display_unit, current_unit_warning)
    reason = current_unit_review_reason(suggested_display_unit, current_unit_warning)
    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-card-header">
                <h4>Unit review required</h4>
                <span class="status-badge status-review-needed">Review needed</span>
            </div>
            <p>{escape_html(explanation)}</p>
            <p>Select the correct unit below, then apply it to scale current-dependent metrics.</p>
            <div class="analysis-grid">
                <div class="analysis-metric">
                    <span>Suggested unit</span>
                    <strong>{escape_html(suggested_label)}</strong>
                </div>
                <div class="analysis-metric">
                    <span>Reason</span>
                    <strong>{escape_html(reason)}</strong>
                </div>
                <div class="analysis-metric">
                    <span>Current column</span>
                    <strong title="{escape_html(current_col)}">{escape_html(current_col_label)}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    confirm_col1, confirm_col2 = st.columns([2, 1])
    confirm_unit_options = confirmation_unit_options(current_source)
    suggested_value = suggested_display_unit if suggested_display_unit in confirm_unit_options else confirm_unit_options[0]
    with confirm_col1:
        confirm_choice = st.selectbox(
            "Confirm current unit" if current_source == "Current" else "Confirm current-density unit",
            confirm_unit_options,
            index=confirm_unit_options.index(suggested_value),
            key=selectbox_key,
            format_func=display_unit_label,
        )
    with confirm_col2:
        st.write("")
        st.write("")
        if st.button("Apply unit", key=button_key):
            st.session_state[confirmed_current_unit_key] = confirm_choice
            if confirmed_unit_type_key:
                st.session_state[confirmed_unit_type_key] = confirmation_unit_type(current_source)
            if unit_confirmation_source_key:
                st.session_state[unit_confirmation_source_key] = "user"
            st.session_state["unit_confirmation_notice"] = (
                f"Unit confirmed as {display_unit_label(confirm_choice)}."
            )
            st.toast(f"Unit confirmed as {display_unit_label(confirm_choice)}.")
            st.rerun()


def render_parser_summary(
    parsed_datasets: list[ParsedDataset],
    experiment_type: str,
    key_prefix: str,
    display_current_unit: Optional[str] = None,
    current_source_override: Optional[str] = None,
    electrode_area_cm2: float = 1.0,
    current_input_unit_override: Optional[str] = None,
    heading: str = "File parsing details",
    show_heading: bool = True,
) -> pd.DataFrame:
    parser_summary = make_parser_summary(
        parsed_datasets,
        experiment_type,
        display_current_unit,
        current_source_override,
        electrode_area_cm2,
        current_input_unit_override,
    )
    if show_heading:
        st.subheader(heading)

    backend_warnings_by_file: dict[str, list[str]] = {dataset.filename: [] for dataset in parsed_datasets}
    global_warnings: list[str] = []
    for warning in cv_lsv_backend_warnings(parsed_datasets, experiment_type):
        matched = False
        for dataset in parsed_datasets:
            prefix = f"{dataset.filename}: "
            if warning.startswith(prefix):
                backend_warnings_by_file[dataset.filename].append(warning[len(prefix):])
                matched = True
                break
        if not matched:
            global_warnings.append(warning)

    advanced_fields = [
        ("Delimiter", "delimiter"),
        ("Metadata rows before header", "metadata_rows_before_header"),
        ("Header row", "header_row"),
        ("Rows skipped before numeric data", "rows_skipped"),
        ("Missing values", "missing_values"),
        ("Rows dropped", "rows_dropped"),
        ("Rows dropped reason", "rows_dropped_reason"),
        ("Rows dropped fraction", "rows_dropped_fraction"),
        ("Parser cleanup severity", "parser_cleanup_severity"),
        ("Unit conversion", "unit_conversion"),
        ("Scan rate estimate", "scan_rate_estimate"),
        ("Scan rate source", "scan_rate_source"),
        ("Scan rate note", "scan_rate_note"),
        ("Cycle count", "cycle_count"),
    ]

    for dataset in parsed_datasets:
        details = parser_summary_details(
            dataset,
            experiment_type,
            display_current_unit,
            current_source_override,
            electrode_area_cm2,
            current_input_unit_override,
        )
        magnitude_ack_key = current_magnitude_ack_key(key_prefix, dataset.filename)
        magnitude_warning_acknowledged = bool(st.session_state.get(magnitude_ack_key))
        combined_warnings = []
        if dataset.warnings:
            combined_warnings.extend(dataset.warnings)
        if details.get("parser_cleanup_notice"):
            combined_warnings.append(str(details["parser_cleanup_notice"]))
        combined_warnings.extend(backend_warnings_by_file.get(dataset.filename, []))
        if details.get("unit_header_warning"):
            combined_warnings.append(str(details["unit_header_warning"]))
        if details.get("current_unit_warning"):
            combined_warnings.append(str(details["current_unit_warning"]))
        if details.get("current_magnitude_warning") and not magnitude_warning_acknowledged:
            combined_warnings.append(str(details["current_magnitude_warning"]))
        combined_warnings = unique_messages([warning for warning in combined_warnings if warning])
        unit_review_required = details.get("current_unit_confidence") == "Low"
        if unit_review_required:
            review_warnings = ["Unit review required"]
            if details.get("parser_cleanup_notice"):
                review_warnings.append(str(details["parser_cleanup_notice"]))
            details["warnings"] = "; ".join(unique_messages(review_warnings))
        else:
            details["warnings"] = "; ".join(combined_warnings) if combined_warnings else "None"

        y_quantity = str(details.get("y_quantity") or "Current")
        quantity_for_label = "current-density" if y_quantity == "Current density" else "current"
        unit_lines = [
            ("Input potential unit", "V"),
            (
                f"Input {quantity_for_label} unit",
                "Ambiguous" if unit_review_required else details["raw_current_unit"],
            ),
            ("Analysis unit", "Pending confirmation" if unit_review_required else details["analysis_current_unit"]),
            ("Displayed as", "Pending confirmation" if unit_review_required else details["display_current_unit"]),
            ("Current unit confidence", details["current_unit_confidence"]),
        ]
        if details.get("suggested_current_unit") not in {None, "", "Not needed"}:
            unit_lines.append(("Suggested current unit", compact_suggested_current_unit(details["suggested_current_unit"])))
        if details.get("unit_header_audit"):
            unit_lines.append(("Unit audit", details["unit_header_audit"]))

        summary_cards = [
            (
                "Rows + warnings",
                [
                    ("Rows imported", details["rows_imported"]),
                    ("Warnings", details["warnings"]),
                ],
            ),
            (
                "Column mapping",
                [
                    ("Potential", details["potential_column"]),
                    ("Y-axis quantity", y_quantity),
                    (f"{y_quantity} column", details["current_column"]),
                ],
            ),
            (
                "Units",
                unit_lines,
            ),
            (
                "Ranges",
                [
                    ("Potential", details["potential_range"]),
                    (
                        y_quantity,
                        "Pending unit confirmation" if unit_review_required else details["current_range"],
                    ),
                ],
            ),
        ]
        metadata_lines = []
        if details.get("scan_rate_estimate") not in {None, "", "Not available"}:
            metadata_lines.append(("Scan rate", summarize_scan_rate_for_card(str(details["scan_rate_estimate"])) or details["scan_rate_estimate"]))
        if details.get("reference_electrode_metadata") not in {None, "", "Not detected"}:
            metadata_lines.append(("Reference electrode", details["reference_electrode_metadata"]))
        if details.get("electrode_area_metadata") not in {None, "", "Not detected"}:
            metadata_lines.append(("Electrode area", details["electrode_area_metadata"]))
        if metadata_lines:
            summary_cards.append(("Detected metadata", metadata_lines))

        metric_html = "\n".join(
            f"""
            <div class="analysis-metric">
                <span>{escape_html(label)}</span>
                <div class="metric-lines">
                    {''.join(
                        f'<div><b>{escape_html(str(line_label))}</b> {escape_html(str(line_value))}</div>'
                        for line_label, line_value in lines
                    )}
                </div>
            </div>
            """
            for label, lines in summary_cards
        )
        st.markdown(
            f"""
            <div class="analysis-card">
                <h4 title="{escape_html(details['file_name'])}">{escape_html(details['file_name'])}</h4>
                <div class="type-badge">Detected type: {escape_html(details['experiment_type'])}</div>
                <div class="analysis-grid">{metric_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if details.get("current_unit_warning") and not unit_review_required:
            st.warning(str(details["current_unit_warning"]))
        if details.get("current_magnitude_warning") and not unit_review_required:
            if magnitude_warning_acknowledged:
                st.info("Current magnitude warning acknowledged.")
            else:
                st.warning(str(details["current_magnitude_warning"]))
                if st.button("Acknowledge current magnitude warning", key=magnitude_ack_key):
                    st.session_state[magnitude_ack_key] = True
                    st.rerun()

        with st.expander(f"Details: {dataset.filename}"):
            advanced_rows = [
                {"Field": label, "Value": str(details[field])}
                for label, field in advanced_fields
            ]
            advanced_rows.extend(
                [
                    {"Field": "Detected reference electrode", "Value": str(details["reference_electrode_metadata"])},
                    {"Field": "Detected electrode area", "Value": str(details["electrode_area_metadata"])},
                    {"Field": "Raw potential column", "Value": str(details["raw_potential_column"])},
                    {"Field": f"Raw {y_quantity.lower()} column", "Value": str(details["raw_current_column"])},
                    {"Field": "Raw potential range", "Value": str(details["raw_potential_range"])},
                    {"Field": f"Original {y_quantity.lower()} values", "Value": str(details["original_current_values"])},
                    {"Field": f"Raw {y_quantity.lower()} range", "Value": str(details["raw_current_range"])},
                    {"Field": f"Analysis {y_quantity.lower()} range", "Value": str(details["analysis_current_range"])},
                    {"Field": f"Raw {y_quantity.lower()} unit", "Value": str(details["raw_current_unit"])},
                    {"Field": f"Analysis {y_quantity.lower()} unit", "Value": str(details["analysis_current_unit"])},
                    {"Field": f"Display {y_quantity.lower()} unit", "Value": str(details["display_current_unit"])},
                    {"Field": "Current unit confidence detail", "Value": str(details["current_unit_confidence_detail"])},
                    {"Field": "Suggested current unit", "Value": str(details["suggested_current_unit"])},
                    {"Field": "Unit header warning", "Value": str(details["unit_header_warning"] or "None")},
                    {"Field": "Unit header audit", "Value": str(details["unit_header_audit"] or "None")},
                    {"Field": "Current magnitude warning", "Value": str(details["current_magnitude_warning"] or "None")},
                    {"Field": "Current magnitude warning acknowledged", "Value": "Yes" if magnitude_warning_acknowledged else "No"},
                ]
            )
            st.dataframe(pd.DataFrame(advanced_rows), width="stretch", hide_index=True)
            st.markdown("**Data preview**")
            st.dataframe(dataframe_preview_for_display(dataset.dataframe).head(30), width="stretch", hide_index=True)

    for warning in global_warnings:
        st.warning(warning)

    return parser_summary


def render_saved_experiment(
    project_name: str,
    project_workspace: dict[str, Any],
    experiment_id: str,
    show_header: bool = True,
) -> None:
    experiment = project_workspace["experiments"][experiment_id]
    files = experiment.get("files", {})
    outputs = experiment.get("analysis_outputs", {})
    experiment_type = outputs.get("experiment_type") or experiment.get("experiment_type", "cv")
    saved_settings = outputs.get("settings", {})
    saved_display_units = saved_settings.get("display_units", {}) if isinstance(saved_settings, dict) else {}
    saved_current_source = (
        saved_settings.get("current_source")
        or saved_settings.get("lsv_current_source")
        or saved_display_units.get("current_source")
        or "Current"
    )
    saved_electrode_area = float(
        saved_settings.get("electrode_area_cm2")
        or saved_display_units.get("electrode_area_cm2")
        or 1.0
    )

    if show_header:
        st.markdown(
            f"""
            <div class="experiment-card">
                <h3>{escape_html(experiment.get("name", experiment_id))}</h3>
                <p>{escape_html(experiment_short_label(experiment_type))} · created {format_timestamp(experiment.get("created_at"))} · {len(files)} files</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if files:
        st.subheader("Saved files")
        file_rows = []
        fig = go.Figure()

        if experiment_type in {"cv", "lsv"}:
            current_display_options = ["uA", "mA", "A", "nA", "mA/cm^2", "uA/cm^2", "A/cm^2", "nA/cm^2"]
            default_display = saved_display_units.get("current") or saved_settings.get("current_display_unit") or "uA"
            if default_display not in current_display_options:
                default_display = "mA/cm^2" if saved_current_source == "Current density" else "uA"
            display_unit = st.selectbox(
                "Saved trace display unit",
                current_display_options,
                index=current_display_options.index(default_display),
                key=f"saved_display_unit::{project_name}::{experiment_id}",
            )
        elif experiment_type == "ca":
            display_unit = st.selectbox(
                "Saved current display unit",
                ["uA", "mA", "A", "nA"],
                key=f"saved_display_unit::{project_name}::{experiment_id}",
            )
        elif experiment_type == "cp":
            display_unit = st.selectbox(
                "Saved potential display unit",
                ["V", "mV"],
                key=f"saved_display_unit::{project_name}::{experiment_id}",
            )
        else:
            display_unit = st.selectbox(
                "Saved impedance display unit",
                ["ohm", "kohm", "Mohm"],
                key=f"saved_display_unit::{project_name}::{experiment_id}",
            )

        for filename, file_record in files.items():
            summary = file_record.get("summary", {})
            trace = file_record.get("trace", {})
            file_rows.append(
                {
                    "file": filename,
                    "sample": summary.get("sample_name", ""),
                    "type": experiment_short_label(summary.get("experiment_type", experiment_type)),
                    "points": summary.get("points", ""),
                    "warnings": "; ".join(summary.get("warnings", [])),
                }
            )

            sample_name = summary.get("sample_name") or filename
            if experiment_type in {"cv", "lsv"}:
                potential = trace.get("potential_v") or trace.get("potential") or []
                current = trace.get("raw_current_a") or trace.get("current") or []
                if potential and current:
                    fig.add_trace(
                        go.Scatter(
                            x=potential,
                            y=current_to_display(
                                np.array(current, dtype=float),
                                display_unit,
                                saved_current_source,
                                saved_electrode_area,
                            ),
                            mode="lines",
                            name=sample_name,
                        )
                    )
            elif experiment_type == "ca":
                time_s = trace.get("time") or []
                current = trace.get("current") or []
                if time_s and current:
                    fig.add_trace(
                        go.Scatter(
                            x=time_s,
                            y=current_from_amps(np.array(current, dtype=float), display_unit),
                            mode="lines",
                            name=sample_name,
                        )
                    )
            elif experiment_type == "cp":
                time_s = trace.get("time") or []
                potential = trace.get("potential") or []
                if time_s and potential:
                    fig.add_trace(
                        go.Scatter(
                            x=time_s,
                            y=convert_role_from_base(np.array(potential, dtype=float), "potential", display_unit),
                            mode="lines",
                            name=sample_name,
                        )
                    )
            elif experiment_type == "eis":
                zreal = trace.get("zreal_ohm") or []
                zimag = trace.get("zimag_ohm") or []
                if zreal and zimag:
                    fig.add_trace(
                        go.Scatter(
                            x=convert_role_from_base(np.array(zreal, dtype=float), "zreal", display_unit),
                            y=convert_role_from_base(-np.array(zimag, dtype=float), "zimag", display_unit),
                            mode="lines+markers",
                            name=sample_name,
                        )
                    )

        st.dataframe(pd.DataFrame(file_rows), width="stretch", hide_index=True)
        if fig.data:
            if experiment_type in {"cv", "lsv"}:
                fig.update_layout(xaxis_title="Potential / V", yaxis_title=current_axis_label(display_unit))
            elif experiment_type == "ca":
                fig.update_layout(xaxis_title="Time / s", yaxis_title=f"Current / {display_unit}")
            elif experiment_type == "cp":
                fig.update_layout(xaxis_title="Time / s", yaxis_title=f"Potential / {display_unit}")
            else:
                fig.update_layout(xaxis_title=f"Z real / {display_unit}", yaxis_title=f"-Z imaginary / {display_unit}")
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_layout(template="plotly_white", height=420, legend_title_text="Saved trace")
            st.plotly_chart(fig, width="stretch")

    if outputs.get("results") is not None:
        output_tabs = st.tabs(["Results", "Parser summary"])
        with output_tabs[0]:
            render_analysis_result_cards(outputs.get("results", []))
        with output_tabs[1]:
            st.dataframe(pd.DataFrame(outputs.get("parser_summary", [])), width="stretch", hide_index=True)
    else:
        output_tabs = st.tabs(["ESW", "Peak metrics", "Parser summary"])
        with output_tabs[0]:
            st.dataframe(pd.DataFrame(outputs.get("esw_summary", [])), width="stretch", hide_index=True)
        with output_tabs[1]:
            st.dataframe(pd.DataFrame(outputs.get("peak_metrics", [])), width="stretch", hide_index=True)
            compact_metrics = outputs.get("compact_metrics", [])
            if compact_metrics:
                st.dataframe(pd.DataFrame(compact_metrics), width="stretch", hide_index=True)
        with output_tabs[2]:
            st.dataframe(pd.DataFrame(outputs.get("parser_summary", [])), width="stretch", hide_index=True)

    render_experiment_notes(project_name, project_workspace, experiment_id)

def render_project_header(project_name: str, project_workspace: dict[str, Any]) -> None:
    experiments = project_workspace.get("experiments", {})
    st.markdown(
        f"""
        <section class="project-hero">
            <div class="eyebrow">Active project</div>
            <h1>{escape_html(project_name)}</h1>
            <p>{len(experiments)} experiments · created {format_timestamp(project_workspace.get("created_at"))}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_experiment_header(
    target: Any,
    experiment: dict[str, Any],
    experiment_id: str,
    type_label: str,
) -> None:
    target.markdown(
        f"""
        <div class="experiment-card">
            <h3>{escape_html(experiment.get("name", experiment_id))}</h3>
            <p>{escape_html(type_label)} · created {format_timestamp(experiment.get("created_at"))} · updated {format_timestamp(experiment.get("updated_at"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_dashboard(project_name: str, project_workspace: dict[str, Any]) -> None:
    experiments = project_workspace.get("experiments", {})
    render_create_experiment_form(project_name, project_workspace, key_prefix="dashboard")

    if experiments:
        render_project_experiment_cards(project_name, project_workspace)

def main() -> None:
    st.set_page_config(page_title="VoltScope AI", layout="wide")
    apply_app_theme()
    unit_notice = st.session_state.pop("unit_confirmation_notice", None)
    if unit_notice:
        st.toast(str(unit_notice))

    if "project_workspaces" not in st.session_state:
        st.session_state.project_workspaces = load_project_workspaces()
    st.session_state.project_workspaces = {
        project_name: normalize_project_workspace(project_name, workspace)
        for project_name, workspace in st.session_state.project_workspaces.items()
    }

    active_project_name = st.session_state.get("active_project_name")
    if not active_project_name:
        render_project_home()
        return

    project_workspace = get_project_workspace(active_project_name)
    active_experiment_id = render_project_sidebar(active_project_name, project_workspace)

    if not active_experiment_id:
        render_project_dashboard(active_project_name, project_workspace)
        return

    experiment_id = active_experiment_id
    experiment = project_workspace["experiments"][experiment_id]
    experiment_type = experiment.setdefault("experiment_type", "auto")
    if experiment_type not in EXPERIMENT_TYPES:
        experiment_type = "auto"
        experiment["experiment_type"] = experiment_type

    experiment_header_slot = st.empty()
    render_experiment_header(
        experiment_header_slot,
        experiment,
        experiment_id,
        experiment_header_type_label(effective_experiment_type(experiment)),
    )

    uploaded_files = st.file_uploader(
        "Data files",
        type=["csv", "txt", "tsv", "dat"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"upload::{active_project_name}::{experiment_id}",
    )

    parsed_datasets: list[ParsedDataset] = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                parsed_datasets.append(parse_electrochem_file(uploaded_file))
            except Exception as exc:
                st.error(f"Could not parse {uploaded_file.name}: {exc}")
    elif experiment.get("files"):
        parsed_datasets = restore_parsed_datasets_from_experiment(experiment)
        if not parsed_datasets:
            render_saved_experiment(active_project_name, project_workspace, experiment_id, show_header=False)
            return
    else:
        return

    if not parsed_datasets:
        st.stop()
        return

    detected_experiment_type, detection_reasons, detection_warnings = detect_uploaded_experiment_type(parsed_datasets)
    if experiment.get("experiment_type") != detected_experiment_type:
        experiment["experiment_type"] = detected_experiment_type
        experiment["updated_at"] = now_timestamp()
        save_project_workspaces()
    experiment_type = detected_experiment_type
    render_experiment_header(
        experiment_header_slot,
        experiment,
        experiment_id,
        experiment_header_type_label(experiment_type, detection_warnings),
    )

    st.caption(f"Detected data type: {experiment_label(experiment_type)}")
    for warning in detection_warnings:
        st.warning(warning)

    if experiment_type != "cv":
        render_generic_experiment_analysis(
            active_project_name,
            project_workspace,
            experiment_id,
            experiment_type,
            parsed_datasets,
        )
        return

    selected_filename = st.selectbox(
        "Analyzing file",
        [dataset.filename for dataset in parsed_datasets],
    )
    selected_dataset = next(dataset for dataset in parsed_datasets if dataset.filename == selected_filename)

    unit_review_container = st.container()
    cv_summary_container = st.container()
    cv_couple_selector_container = st.container()
    cv_plot_container = st.container()
    cv_metrics_container = st.container()
    peak_controls_expander = st.expander("Peak Selection & Baseline Correction", expanded=False)
    candidate_peaks_expander = st.expander("Candidate peaks / audit trail", expanded=False)
    data_details_expander = st.expander("Advanced details", expanded=False)
    esw_expander = st.expander("Advanced Analyses: Operational ESW", expanded=False)

    manual_potential_col = selected_dataset.detected_potential_col or selected_dataset.headers[0]
    manual_current_col = selected_dataset.detected_current_col or selected_dataset.headers[min(1, len(selected_dataset.headers) - 1)]

    with data_details_expander:
        render_detected_metadata(
            selected_dataset.metadata,
            f"manual_metadata::{active_project_name}::{experiment_id}::{selected_dataset.filename}",
        )

        st.divider()

        st.markdown("**Data Mapping & Units**")
        st.caption(
            f"Detected mapping: {manual_potential_col} -> Potential; {manual_current_col} -> Current."
        )
        column_mode = st.radio(
            "Column selection mode",
            ["Use detected columns per file", "Manually select columns"],
            horizontal=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            manual_potential_col = st.selectbox(
                "Potential column",
                selected_dataset.headers,
                index=selected_dataset.headers.index(manual_potential_col),
                disabled=column_mode == "Use detected columns per file",
            )
        with col2:
            manual_current_col = st.selectbox(
                "Current column",
                selected_dataset.headers,
                index=selected_dataset.headers.index(manual_current_col),
                disabled=column_mode == "Use detected columns per file",
            )

        unit_col1, unit_col2, unit_col3 = st.columns(3)
        with unit_col1:
            selected_potential_unit = st.selectbox("Potential input unit", ["Auto", "V", "mV"])
        with unit_col2:
            current_unit_options = ["Auto", "A", "mA", "uA", "nA", "pA"]
            confirmed_current_unit_key = (
                f"cv_confirmed_current_unit::{active_project_name}::{experiment_id}::"
                f"{selected_dataset.filename}::{manual_current_col}"
            )
            confirmed_current_unit = st.session_state.get(confirmed_current_unit_key)
            current_unit_index = (
                current_unit_options.index(confirmed_current_unit)
                if confirmed_current_unit in current_unit_options
                else 0
            )
            selected_current_unit = st.selectbox(
                "Current input unit",
                current_unit_options,
                index=current_unit_index,
                format_func=display_unit_label,
            )
        with unit_col3:
            y_axis_quantity = st.selectbox("Y-axis quantity", ["Current", "Current density"])

        source_default = 1 if current_source_guess_for_dataset(selected_dataset, manual_current_col) == "Current density" else 0
        current_source = st.selectbox(
            "Uploaded current column contains",
            ["Current", "Current density"],
            index=source_default,
            key=f"cv_current_source::{active_project_name}::{experiment_id}",
        )

        electrode_area_cm2 = 1.0
        if y_axis_quantity == "Current density":
            preview_current_unit = resolve_unit(
                selected_current_unit,
                selected_dataset,
                "current",
                manual_current_col,
                "A",
            )
            current_display_unit = current_density_unit_from_current_unit(preview_current_unit)
        else:
            current_display_unit = st.selectbox(
                "Current display unit",
                ["uA", "mA", "A", "nA"],
                format_func=display_unit_label,
            )

        area_required = (
            (current_source == "Current" and is_current_density_unit(current_display_unit))
            or (current_source == "Current density" and not is_current_density_unit(current_display_unit))
        )
        if area_required:
            electrode_area_cm2 = st.number_input(
                "Electrode surface area (cm\u00b2)",
                min_value=0.000001,
                value=1.0,
                step=0.1,
                format="%.6f",
                key=f"cv_electrode_area::{active_project_name}::{experiment_id}",
            )
        elif current_source == "Current density":
            st.info("The uploaded current column is already normalized, so electrode-area normalization is skipped.")

        st.divider()

        st.markdown("**Advanced Processing**")
        proc_col1, proc_col2, proc_col3 = st.columns(3)
        with proc_col1:
            smoothing_method = st.selectbox("Smoothing", ["None", "Moving average", "Savitzky-Golay"])
        with proc_col2:
            smoothing_window = 11
            if smoothing_method != "None":
                smoothing_window = st.slider("Smoothing window", min_value=3, max_value=51, value=11, step=2)
        with proc_col3:
            use_smoothed_for_peaks = False
            if smoothing_method != "None":
                use_smoothed_for_peaks = st.checkbox("Use smoothed data for peak detection", value=True)

        plot_col1, plot_col2, plot_col3 = st.columns(3)
        with plot_col1:
            reverse_x_axis = st.checkbox("Reverse x-axis", value=False)
        with plot_col2:
            show_markers = st.checkbox("Show data markers", value=False)
        with plot_col3:
            show_raw_with_smoothed = st.checkbox("Show raw trace when smoothed", value=True)

    current_input_unit_override = selected_current_unit if selected_current_unit != "Auto" else None
    current_unit_is_low_confidence = current_unit_is_ambiguous(
        selected_dataset,
        manual_current_col,
        selected_current_unit,
    )
    with unit_review_container:
        if current_unit_is_low_confidence and manual_current_col in selected_dataset.dataframe:
            suggested_current_unit = suggested_current_unit_for_values(
                selected_dataset.dataframe[manual_current_col].to_numpy(dtype=float)
            )
            unit_review_warning = current_unit_magnitude_warning(
                selected_dataset,
                manual_current_col,
                selected_current_unit,
                "A",
                current_source,
                electrode_area_cm2,
            )
            render_unit_review_panel(
                current_col=manual_current_col,
                current_col_label=user_facing_column_label(selected_dataset, manual_current_col),
                suggested_unit=suggested_current_unit,
                current_source=current_source,
                current_unit_warning=unit_review_warning,
                confirmed_current_unit_key=confirmed_current_unit_key,
                confirmed_unit_type_key=(
                    f"cv_confirmed_current_unit_type::{active_project_name}::{experiment_id}::"
                    f"{selected_dataset.filename}::{manual_current_col}"
                ),
                unit_confirmation_source_key=(
                    f"cv_unit_confirmation_source::{active_project_name}::{experiment_id}::"
                    f"{selected_dataset.filename}::{manual_current_col}"
                ),
                selectbox_key=(
                    f"cv_confirm_current_unit_choice::{active_project_name}::"
                    f"{experiment_id}::{selected_dataset.filename}::{manual_current_col}"
                ),
                button_key=(
                    f"cv_apply_current_unit::{active_project_name}::"
                    f"{experiment_id}::{selected_dataset.filename}::{manual_current_col}"
                ),
            )
        elif selected_current_unit != "Auto":
            st.success(f"Current unit confirmed by user: {display_unit_label(selected_current_unit)}")

    with data_details_expander:
        st.divider()
        parser_summary = render_parser_summary(
            parsed_datasets,
            "cv",
            key_prefix=f"cv_parser::{active_project_name}::{experiment_id}",
            display_current_unit=current_display_unit,
            current_source_override=current_source,
            electrode_area_cm2=electrode_area_cm2,
            current_input_unit_override=current_input_unit_override,
            heading="File parsing details",
        )

    analyzed: list[AnalyzedDataset] = []
    for dataset in parsed_datasets:
        if column_mode == "Use detected columns per file":
            potential_col = dataset.detected_potential_col
            current_col = dataset.detected_current_col
        else:
            potential_col = manual_potential_col if manual_potential_col in dataset.headers else None
            current_col = manual_current_col if manual_current_col in dataset.headers else None

        if potential_col is None:
            st.warning(f"Skipping {dataset.filename}: no potential column selected or detected.")
            continue
        if current_col is None:
            st.warning(f"Skipping {dataset.filename}: no current column selected or detected.")
            continue
        if potential_col == current_col:
            st.warning(f"Skipping {dataset.filename}: potential and current columns cannot be the same.")
            continue

        try:
            analyzed.append(
                normalize_dataset(
                    dataset,
                    potential_col,
                    current_col,
                    selected_potential_unit,
                    selected_current_unit,
                    current_source,
                    smoothing_method,
                    smoothing_window,
                )
            )
        except Exception as exc:
            st.warning(f"Skipping {dataset.filename}: {exc}")

    if not analyzed:
        st.error("No uploaded files could be analyzed with the selected settings.")
        st.stop()
        return

    for dataset in analyzed:
        for warning in dataset.warnings:
            st.warning(warning)

    detailed_dataset = next(
        (dataset for dataset in analyzed if dataset.filename == selected_filename),
        analyzed[0],
    )
    detailed_parsed_dataset = next(
        (dataset for dataset in parsed_datasets if dataset.filename == detailed_dataset.filename),
        selected_dataset,
    )
    detailed_parser_details = parser_summary_details(
        detailed_parsed_dataset,
        "cv",
        current_display_unit,
        current_source,
        electrode_area_cm2,
        selected_current_unit if selected_current_unit != "Auto" else None,
    )
    full_analysis_current_a = (
        detailed_dataset.smoothed_current_a
        if use_smoothed_for_peaks and detailed_dataset.smoothed_current_a is not None
        else detailed_dataset.raw_current_a
    )
    cycle_values = available_cycle_values(detailed_dataset.cycle_values)
    cycle_assignment_uncertain = len(cycle_values) > 1 and detailed_dataset.cycle_source == "inferred"
    selected_cycle_value: Optional[float] = None
    selected_cycle_label: Optional[str] = None
    selected_cycle_suffix = ""
    selected_cycle_summary_label: Optional[str] = None
    cycle_plot_mode = "Show all cycles overlay"

    with cv_summary_container:
        if len(cycle_values) > 1:
            cycle_labels = [cycle_display_label(value) for value in cycle_values]
            default_cycle_index = len(cycle_values) - 1
            cycle_select_key = f"cv_selected_cycle::{active_project_name}::{experiment_id}::{selected_filename}"
            if st.session_state.get(cycle_select_key) not in cycle_labels:
                st.session_state[cycle_select_key] = cycle_labels[default_cycle_index]
            selected_cycle_label = st.selectbox(
                "Analyze cycle",
                cycle_labels,
                index=default_cycle_index,
                key=cycle_select_key,
            )
            selected_cycle_value = cycle_values[cycle_labels.index(selected_cycle_label)]
            selected_cycle_summary_label = analyzed_cycle_label(selected_cycle_value, len(cycle_values))
            for message_level, cycle_message in cycle_assignment_messages(
                detailed_dataset,
                len(cycle_values),
                selected_cycle_summary_label,
            ):
                if message_level == "warning":
                    st.warning(cycle_message)
                else:
                    st.info(cycle_message)
            cycle_plot_mode = st.radio(
                "Plot cycles",
                ["Show all cycles overlay", "Show selected cycle only"],
                horizontal=True,
                key=f"cv_cycle_plot_mode::{active_project_name}::{experiment_id}::{selected_filename}",
            )
            selected_cycle_suffix = f" — {selected_cycle_label}"

    selected_cycle_mask = cycle_selection_mask(
        detailed_dataset.cycle_values,
        selected_cycle_value,
        len(detailed_dataset.potential_v),
    )
    selected_potential_v = detailed_dataset.potential_v[selected_cycle_mask]
    selected_raw_current_a = detailed_dataset.raw_current_a[selected_cycle_mask]
    analysis_current_a = full_analysis_current_a[selected_cycle_mask]

    if len(selected_potential_v) < 5:
        st.error("The selected cycle has too few points for CV analysis.")
        st.stop()
        return

    baseline_background_suspected = detect_sloped_baseline_background(selected_potential_v, analysis_current_a)

    with peak_controls_expander:
        st.subheader("Peak Selection & Baseline Correction")
        st.caption(
            "Method: split the CV into forward and reverse scans, ignore the first/last 5% of each scan, "
            "then select Epa/Epc from local extrema using scan direction, current sign, prominence, and noise."
        )
        current_display_values = current_to_display(
            analysis_current_a,
            current_display_unit,
            current_source,
            electrode_area_cm2,
        )
        current_range = float(np.nanmax(current_display_values) - np.nanmin(current_display_values))
        auto_peak_params = auto_tune_cv_peak_parameters(selected_potential_v, analysis_current_a)
        auto_prominence_display = float(
            current_to_display(
                np.array([auto_peak_params["min_prominence"]]),
                current_display_unit,
                current_source,
                electrode_area_cm2,
            )[0]
        )
        auto_min_abs_current_display = float(
            current_to_display(
                np.array([auto_peak_params["min_abs_current"]]),
                current_display_unit,
                current_source,
                electrode_area_cm2,
            )[0]
        )
        default_prominence = max(
            abs(auto_prominence_display),
            abs(current_to_display(np.array([1e-12]), current_display_unit, current_source, electrode_area_cm2)[0]),
        )

        auto_tune_peak_settings = st.checkbox(
            "Auto-tune peak selection parameters",
            value=True,
            key=f"cv_auto_peak_settings::{active_project_name}::{experiment_id}::{selected_filename}",
        )
        if auto_tune_peak_settings:
            st.caption(
                f"Auto-tuned for this CV: prominence {default_prominence:.4g} {display_unit_label(current_display_unit)}, "
                f"distance {int(auto_peak_params['min_distance'])} points, minimum absolute current 0 {display_unit_label(current_display_unit)}."
            )

        peak_setting_mode = "auto" if auto_tune_peak_settings else "manual"
        peak_setting_key_base = (
            f"{active_project_name}::{experiment_id}::{selected_filename}::"
            f"{current_display_unit}::{smoothing_method}::{use_smoothed_for_peaks}::{peak_setting_mode}"
        )
        peak_col1, peak_col2, peak_col3 = st.columns(3)
        with peak_col1:
            min_prominence_widget = st.number_input(
                f"Minimum peak prominence ({display_unit_label(current_display_unit)})",
                min_value=0.0,
                value=float(default_prominence),
                step=float(default_prominence / 2 if default_prominence > 0 else 1e-6),
                format="%.6e",
                disabled=auto_tune_peak_settings,
                key=f"cv_min_peak_prominence::{peak_setting_key_base}",
            )
        with peak_col2:
            min_distance_widget = st.number_input(
                "Minimum peak distance (points)",
                min_value=1,
                value=int(auto_peak_params["min_distance"]),
                step=1,
                disabled=auto_tune_peak_settings,
                key=f"cv_min_peak_distance::{peak_setting_key_base}",
            )
        with peak_col3:
            min_abs_current_widget = st.number_input(
                f"Minimum absolute current ({display_unit_label(current_display_unit)})",
                min_value=0.0,
                value=float(max(0.0, auto_min_abs_current_display)),
                step=float(default_prominence / 2 if default_prominence > 0 else 1e-6),
                format="%.6e",
                disabled=auto_tune_peak_settings,
                key=f"cv_min_abs_current::{peak_setting_key_base}",
            )

        if auto_tune_peak_settings:
            min_prominence_display = float(default_prominence)
            min_distance = int(auto_peak_params["min_distance"])
            min_abs_current_display = float(max(0.0, auto_min_abs_current_display))
        else:
            min_prominence_display = float(min_prominence_widget)
            min_distance = int(min_distance_widget)
            min_abs_current_display = float(min_abs_current_widget)

        min_prominence_a = current_display_value_to_base(
            min_prominence_display,
            current_display_unit,
            current_source,
            electrode_area_cm2,
        )
        min_abs_current_a = current_display_value_to_base(
            min_abs_current_display,
            current_display_unit,
            current_source,
            electrode_area_cm2,
        )
        peaks = detect_peaks(
            selected_potential_v,
            analysis_current_a,
            min_prominence_a,
            int(min_distance),
            min_abs_current_a,
        )
        all_cycle_peaks = peaks
        if len(cycle_values) > 1:
            all_cycle_peaks = detect_peaks(
                detailed_dataset.potential_v,
                full_analysis_current_a,
                min_prominence_a,
                int(min_distance),
                min_abs_current_a,
            )

        oxidation_peaks = [peak for peak in peaks if peak.peak_type == "oxidation"]
        reduction_peaks = [peak for peak in peaks if peak.peak_type == "reduction"]

        manual_peak_override = st.checkbox("Override selected Epa/Epc", value=False)
        selected_oxidation_peak: Optional[Peak] = None
        selected_reduction_peak: Optional[Peak] = None

        if manual_peak_override:
            manual_col1, manual_col2 = st.columns(2)
            with manual_col1:
                oxidation_idx = st.slider(
                    "Oxidation peak index",
                    min_value=0,
                    max_value=len(selected_potential_v) - 1,
                    value=int(np.nanargmax(analysis_current_a)),
                )
                selected_oxidation_peak = make_manual_peak(
                    "oxidation", oxidation_idx, selected_potential_v, analysis_current_a
                )
            with manual_col2:
                reduction_idx = st.slider(
                    "Reduction peak index",
                    min_value=0,
                    max_value=len(selected_potential_v) - 1,
                    value=int(np.nanargmin(analysis_current_a)),
                )
                selected_reduction_peak = make_manual_peak(
                    "reduction", reduction_idx, selected_potential_v, analysis_current_a
                )
        else:
            select_col1, select_col2 = st.columns(2)
            with select_col1:
                if oxidation_peaks:
                    labels = [
                        peak_label(peak, current_display_unit, electrode_area_cm2, current_source)
                        for peak in oxidation_peaks
                    ]
                    default_oxidation_idx = default_primary_peak_index(oxidation_peaks, "oxidation")
                    selected_label = st.selectbox("Oxidation peak", labels, index=default_oxidation_idx)
                    selected_oxidation_peak = oxidation_peaks[labels.index(selected_label)]
                else:
                    st.info("No oxidation peaks detected. Lower the prominence threshold or use manual selection.")
            with select_col2:
                if reduction_peaks:
                    labels = [
                        peak_label(peak, current_display_unit, electrode_area_cm2, current_source)
                        for peak in reduction_peaks
                    ]
                    default_reduction_idx = default_primary_peak_index(reduction_peaks, "reduction")
                    selected_label = st.selectbox("Reduction peak", labels, index=default_reduction_idx)
                    selected_reduction_peak = reduction_peaks[labels.index(selected_label)]
                else:
                    st.info("No reduction peaks detected. Lower the prominence threshold or use manual selection.")

        raw_selected_oxidation_peak = selected_oxidation_peak
        raw_selected_reduction_peak = selected_reduction_peak
        _behavior_rows, behavior_metrics = build_peak_metrics_rows(
            raw_selected_oxidation_peak,
            raw_selected_reduction_peak,
            None,
            analysis_current_a,
            current_display_unit,
            electrode_area_cm2,
            current_source,
        )
        cv_behavior = classify_cv_behavior(
            raw_selected_oxidation_peak,
            raw_selected_reduction_peak,
            behavior_metrics,
            selected_potential_v,
            peaks,
            baseline_background_suspected,
        )
        show_all_detected_couples = False
        if cv_behavior.behavior == "multiple_redox_couples" and cv_behavior.redox_couples:
            with cv_couple_selector_container:
                st.markdown("**Redox couple selection**")
                couple_labels = [redox_couple_option_label(couple) for couple in cv_behavior.redox_couples]
                selected_couple_label = st.selectbox(
                    "Analyze redox couple",
                    couple_labels,
                    index=cv_behavior.selected_couple_index or 0,
                    key=f"cv_selected_redox_couple::{active_project_name}::{experiment_id}::{selected_filename}",
                )
                selected_couple_index = couple_labels.index(selected_couple_label)
                cv_behavior = behavior_from_redox_couple(cv_behavior, selected_couple_index)
                couple_table = redox_couples_display_table(cv_behavior.redox_couples, selected_couple_index)

                def highlight_selected_couple(row: pd.Series) -> list[str]:
                    if row.get("Selected?") == "Selected":
                        return ["background-color: #fff7ed; font-weight: 600;" for _ in row]
                    return ["" for _ in row]

                with st.expander("Detected couples", expanded=True):
                    st.dataframe(
                        couple_table.style.hide(axis="index").apply(highlight_selected_couple, axis=1),
                        width="stretch",
                    )
                show_all_detected_couples = st.checkbox(
                    "Show all detected couples on plot",
                    value=False,
                    key=f"cv_show_redox_couples::{active_project_name}::{experiment_id}::{selected_filename}",
                )
        selected_oxidation_peak = cv_behavior.oxidation_peak
        selected_reduction_peak = cv_behavior.reduction_peak

        selected_primary_rows = []
        missing_primary_rows = {
            "Selected primary Epa": {
                "role": "Primary Epa",
                "status": (
                    "Not assigned: no reliable anodic return peak"
                    if cv_behavior.behavior == "irreversible_reduction_only"
                    else "Not selected"
                ),
                "confidence": "Low" if cv_behavior.behavior == "irreversible_reduction_only" else "",
            },
            "Selected primary Epc": {
                "role": "Primary Epc",
                "status": (
                    "Not assigned: no reliable cathodic return peak"
                    if cv_behavior.behavior == "irreversible_oxidation_only"
                    else "Not selected"
                ),
                "confidence": "Low" if cv_behavior.behavior == "irreversible_oxidation_only" else "",
            },
        }
        for role, peak in [("Selected primary Epa", selected_oxidation_peak), ("Selected primary Epc", selected_reduction_peak)]:
            if peak is None:
                missing_row = missing_primary_rows.get(role, {})
                selected_primary_rows.append(
                    {
                        "role": missing_row.get("role", role),
                        "status": missing_row.get("status", "Not selected"),
                        "potential_V": "\u2014",
                        f"current_{current_display_unit}": "\u2014",
                        "scan": "\u2014",
                        "confidence": missing_row.get("confidence", ""),
                    }
                )
                continue
            selected_primary_rows.append(
                {
                    "role": role,
                    "status": "Used for CV metrics",
                    "potential_V": peak.potential,
                    f"current_{current_display_unit}": current_to_display(
                        np.array([analysis_current_a[peak.index]]),
                        current_display_unit,
                        current_source,
                        electrode_area_cm2,
                    )[0],
                    "scan": scan_direction_label(peak.segment_direction),
                    "confidence": peak.confidence,
                }
            )
        rejected_peak_reasons = candidate_rejection_reasons(cv_behavior)
        for role, peak in [
            ("Rejected candidate Epa", cv_behavior.rejected_oxidation_peak),
            ("Rejected candidate Epc", cv_behavior.rejected_reduction_peak),
        ]:
            if peak is None:
                continue
            selected_primary_rows.append(
                {
                    "role": role,
                    "status": rejected_peak_reasons.get(peak.id, "Rejected as primary peak"),
                    "potential_V": peak.potential,
                    f"current_{current_display_unit}": current_to_display(
                        np.array([analysis_current_a[peak.index]]),
                        current_display_unit,
                        current_source,
                        electrode_area_cm2,
                    )[0],
                    "scan": scan_direction_label(peak.segment_direction),
                    "confidence": peak.confidence,
                }
            )

        st.markdown("**Selected Epa/Epc used for CV metrics**")
        render_cv_report_table(pd.DataFrame(selected_primary_rows))

        show_all_candidate_peaks = st.checkbox(
            "Show all candidate peaks on plot",
            value=False,
            key=f"cv_show_candidate_peaks::{active_project_name}::{experiment_id}::{selected_filename}",
        )

        st.subheader("Baseline correction")
        baseline_method = st.radio("Baseline method", ["None", "Linear"], horizontal=True)
        baseline_current: Optional[np.ndarray] = None
        anchor_a_idx = 0
        anchor_b_idx = len(selected_potential_v) - 1

        if baseline_method == "Linear":
            reference_peak = selected_oxidation_peak or selected_reduction_peak
            reference_idx = reference_peak.index if reference_peak else len(selected_potential_v) // 2
            default_span = max(5, len(selected_potential_v) // 10)
            default_a = max(0, reference_idx - default_span)
            default_b = min(len(selected_potential_v) - 1, reference_idx + default_span)

            anchor_col1, anchor_col2 = st.columns(2)
            with anchor_col1:
                anchor_a_idx = st.slider(
                    "Baseline anchor A index",
                    min_value=0,
                    max_value=len(selected_potential_v) - 1,
                    value=default_a,
                )
            with anchor_col2:
                anchor_b_idx = st.slider(
                    "Baseline anchor B index",
                    min_value=0,
                    max_value=len(selected_potential_v) - 1,
                    value=default_b,
                )

            baseline_current = calculate_linear_baseline(
                selected_potential_v,
                analysis_current_a,
                anchor_a_idx,
                anchor_b_idx,
            )
            if baseline_current is None:
                st.error("The two baseline anchors have the same potential. Move one anchor.")

            anchor_summary = pd.DataFrame(
                [
                    {
                        "anchor": "A",
                        "index": anchor_a_idx,
                        "potential_V": selected_potential_v[anchor_a_idx],
                        f"current_{current_display_unit}": current_to_display(
                            np.array([analysis_current_a[anchor_a_idx]]),
                            current_display_unit,
                            current_source,
                            electrode_area_cm2,
                        )[0],
                    },
                    {
                        "anchor": "B",
                        "index": anchor_b_idx,
                        "potential_V": selected_potential_v[anchor_b_idx],
                        f"current_{current_display_unit}": current_to_display(
                            np.array([analysis_current_a[anchor_b_idx]]),
                            current_display_unit,
                            current_source,
                            electrode_area_cm2,
                        )[0],
                    },
                ]
            )
            st.dataframe(anchor_summary, width="stretch", hide_index=True)

        if baseline_current is not None:
            corrected_current_for_peak_selection = analysis_current_a - baseline_current
            corrected_peaks = detect_peaks(
                selected_potential_v,
                corrected_current_for_peak_selection,
                min_prominence_a,
                int(min_distance),
                min_abs_current_a,
            )
            peaks = corrected_peaks
            if manual_peak_override:
                if raw_selected_oxidation_peak is not None:
                    raw_selected_oxidation_peak = make_manual_peak(
                        "oxidation",
                        raw_selected_oxidation_peak.index,
                        selected_potential_v,
                        corrected_current_for_peak_selection,
                    )
                if raw_selected_reduction_peak is not None:
                    raw_selected_reduction_peak = make_manual_peak(
                        "reduction",
                        raw_selected_reduction_peak.index,
                        selected_potential_v,
                        corrected_current_for_peak_selection,
                    )
            else:
                raw_selected_oxidation_peak, raw_selected_reduction_peak = select_default_primary_peaks(peaks)

            _behavior_rows, behavior_metrics = build_peak_metrics_rows(
                raw_selected_oxidation_peak,
                raw_selected_reduction_peak,
                baseline_current,
                analysis_current_a,
                current_display_unit,
                electrode_area_cm2,
                current_source,
            )
            cv_behavior = classify_cv_behavior(
                raw_selected_oxidation_peak,
                raw_selected_reduction_peak,
                behavior_metrics,
                selected_potential_v,
                peaks,
                baseline_background_suspected,
            )
            selected_oxidation_peak = cv_behavior.oxidation_peak
            selected_reduction_peak = cv_behavior.reduction_peak
            rejected_peak_reasons = candidate_rejection_reasons(cv_behavior)

    with candidate_peaks_expander:
        st.caption(candidate_peak_explanation(cv_behavior))
        candidate_display_filter = st.selectbox(
            "Candidate peak display",
            CANDIDATE_PEAK_FILTER_OPTIONS,
            index=CANDIDATE_PEAK_FILTER_OPTIONS.index("Top candidates"),
            key=f"cv_candidate_peak_display::{active_project_name}::{experiment_id}::{selected_filename}",
        )
        show_candidate_debug_columns = st.checkbox(
            "Show debug columns",
            value=False,
            key=f"cv_candidate_debug_columns::{active_project_name}::{experiment_id}::{selected_filename}",
        )
        include_all_cycle_candidates = False
        if len(cycle_values) > 1:
            include_all_cycle_candidates = st.checkbox(
                "Include candidate peaks from all cycles",
                value=False,
                key=f"cv_candidate_all_cycles::{active_project_name}::{experiment_id}::{selected_filename}",
            )
        candidate_source_peaks = all_cycle_peaks if include_all_cycle_candidates else peaks
        candidate_source_potential = detailed_dataset.potential_v if include_all_cycle_candidates else selected_potential_v
        candidate_display_label = display_unit_label(current_display_unit)
        selected_peak_roles = {}
        if selected_oxidation_peak is not None:
            selected_peak_roles[selected_oxidation_peak.id] = "Selected Epa"
        if selected_reduction_peak is not None:
            selected_peak_roles[selected_reduction_peak.id] = "Selected Epc"
        rejected_peak_roles = {}
        if cv_behavior.rejected_oxidation_peak is not None:
            rejected_peak_roles[cv_behavior.rejected_oxidation_peak.id] = "Rejected candidate Epa"
        if cv_behavior.rejected_reduction_peak is not None:
            rejected_peak_roles[cv_behavior.rejected_reduction_peak.id] = "Rejected candidate Epc"
        couple_peak_roles: dict[str, str] = {}
        couple_peak_ids: dict[str, str] = {}
        couple_peak_reasons: dict[str, str] = {}
        if cv_behavior.behavior == "multiple_redox_couples" and cv_behavior.redox_couples:
            selected_couple_index = cv_behavior.selected_couple_index or 0
            for idx, couple in enumerate(cv_behavior.redox_couples):
                is_selected_couple = idx == selected_couple_index
                ox_role = "Selected couple Epa" if is_selected_couple else "Candidate couple Epa"
                red_role = "Selected couple Epc" if is_selected_couple else "Candidate couple Epc"
                reason = (
                    "Used in selected redox-couple metrics."
                    if is_selected_couple
                    else f"Paired as {couple.id}; not selected for main metrics."
                )
                couple_peak_roles[couple.oxidation_peak.id] = ox_role
                couple_peak_roles[couple.reduction_peak.id] = red_role
                couple_peak_ids[couple.oxidation_peak.id] = couple.id
                couple_peak_ids[couple.reduction_peak.id] = couple.id
                couple_peak_reasons[couple.oxidation_peak.id] = reason
                couple_peak_reasons[couple.reduction_peak.id] = reason
        for message in candidate_peak_ambiguity_messages(
            candidate_source_peaks,
            selected_oxidation_peak,
            selected_reduction_peak,
        ):
            st.warning(message)

        def candidate_role_for_peak(peak: Peak) -> str:
            if peak.id in couple_peak_roles:
                return couple_peak_roles[peak.id]
            if peak.id in selected_peak_roles:
                return selected_peak_roles[peak.id]
            if peak.id in rejected_peak_roles:
                return rejected_peak_roles[peak.id]
            if cv_behavior.behavior == "multiple_redox_couples":
                if peak.peak_type == "oxidation":
                    return "Unpaired oxidation candidate"
                if peak.peak_type == "reduction":
                    return "Unpaired reduction candidate"
            return "Candidate"

        def candidate_reason_for_peak(peak: Peak) -> str:
            if peak.id in couple_peak_reasons:
                return couple_peak_reasons[peak.id]
            if peak.id in rejected_peak_reasons:
                return rejected_peak_reasons[peak.id]
            if cv_behavior.behavior == "multiple_redox_couples":
                return "No plausible unused partner assigned by the current pairing heuristic."
            return ""

        full_candidate_rows = []
        for peak in candidate_source_peaks:
            row = {
                "role": candidate_role_for_peak(peak),
                "couple_id": couple_peak_ids.get(peak.id, ""),
                "id": peak.id,
                "extremum": local_extremum_label(peak),
                "potential_V": peak.potential,
                f"current_{candidate_display_label}": current_to_display(
                    np.array([peak.raw_current]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                f"prominence_{candidate_display_label}": current_to_display(
                    np.array([peak.prominence]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                "scan": scan_direction_label(peak.segment_direction),
                "confidence": peak.confidence,
                "reason": candidate_reason_for_peak(peak),
            }
            if show_candidate_debug_columns:
                row = {"index": peak.index, **row}
            full_candidate_rows.append(row)

        displayed_candidate_peaks = filter_candidate_peaks_for_display(
            candidate_source_peaks,
            selected_oxidation_peak,
            selected_reduction_peak,
            candidate_display_filter,
            potential=candidate_source_potential,
        )
        displayed_candidate_rows = []
        for peak in displayed_candidate_peaks:
            row = {
                "role": candidate_role_for_peak(peak),
                "couple_id": couple_peak_ids.get(peak.id, ""),
                "id": peak.id,
                "extremum": local_extremum_label(peak),
                "potential_V": peak.potential,
                f"current_{candidate_display_label}": current_to_display(
                    np.array([peak.raw_current]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                f"prominence_{candidate_display_label}": current_to_display(
                    np.array([peak.prominence]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                "scan": scan_direction_label(peak.segment_direction),
                "confidence": peak.confidence,
                "reason": candidate_reason_for_peak(peak),
            }
            if show_candidate_debug_columns:
                row = {"index": peak.index, **row}
            displayed_candidate_rows.append(row)
        if displayed_candidate_rows:
            if candidate_display_filter == "Top candidates":
                if cv_behavior.behavior == "irreversible_oxidation_only":
                    candidate_caption = (
                        "Showing selected Epa plus top candidate extrema. No Epc was assigned. "
                        "Download full candidate table for all local extrema."
                    )
                elif cv_behavior.behavior == "irreversible_reduction_only":
                    candidate_caption = (
                        "Showing selected Epc plus top candidate extrema. No Epa was assigned. "
                        "Download full candidate table for all local extrema."
                    )
                elif cv_behavior.behavior == "multiple_redox_couples":
                    candidate_caption = (
                        "Showing selected redox couple plus top candidate extrema. "
                        "Download full candidate table for all local extrema."
                    )
                else:
                    candidate_caption = (
                        "Showing selected Epa/Epc plus top candidate extrema. "
                        "Download full candidate table for all local extrema."
                    )
                st.caption(candidate_caption)
            else:
                st.caption(f"Showing {len(displayed_candidate_rows)} of {len(full_candidate_rows)} candidate extrema.")
            render_cv_report_table(pd.DataFrame(displayed_candidate_rows))
        elif full_candidate_rows:
            st.info("No candidate extrema match the selected display filter.")
        else:
            st.info("No candidate peaks were detected with the current settings.")

        if full_candidate_rows:
            full_candidate_csv = pd.DataFrame(full_candidate_rows).to_csv(index=False)
            st.download_button(
                "Download full candidate peaks CSV",
                data=full_candidate_csv.encode("utf-8"),
                file_name=f"{Path(selected_filename).stem}_candidate_peaks.csv",
                mime="text/csv",
                key=f"download_cv_candidate_peaks::{active_project_name}::{experiment_id}::{selected_filename}",
                on_click="ignore",
            )


    peak_metric_rows, metrics = build_peak_metrics_rows(
        selected_oxidation_peak,
        selected_reduction_peak,
        baseline_current,
        analysis_current_a,
        current_display_unit,
        electrode_area_cm2,
        current_source,
    )
    per_cycle_metrics_df = pd.DataFrame()
    per_cycle_trend_messages: list[str] = []
    if len(cycle_values) > 1:
        per_cycle_metrics_df, per_cycle_trend_messages = build_per_cycle_cv_metrics(
            detailed_dataset.potential_v,
            full_analysis_current_a,
            detailed_dataset.cycle_values,
            cycle_values,
            selected_cycle_value,
            min_prominence_a,
            int(min_distance),
            min_abs_current_a,
            current_display_unit,
            electrode_area_cm2,
            current_source,
            metrics,
            selected_oxidation_peak,
            selected_reduction_peak,
            cv_behavior,
            auto_tune_peak_settings,
        )

    quality_messages = cv_analysis_quality_messages(
        selected_oxidation_peak,
        selected_reduction_peak,
        metrics,
        selected_potential_v,
        peaks,
        cv_behavior,
    )
    current_unit_review_needed = detailed_parser_details.get("current_unit_confidence") == "Low"
    parser_confidence = parser_confidence_from_details(detailed_parser_details)
    current_unit_warning = str(detailed_parser_details.get("current_unit_warning") or "")
    current_magnitude_ack = bool(
        st.session_state.get(
            current_magnitude_ack_key(
                f"cv_parser::{active_project_name}::{experiment_id}",
                detailed_parsed_dataset.filename,
            )
        )
    )
    current_magnitude_warning = (
        ""
        if current_magnitude_ack
        else str(detailed_parser_details.get("current_magnitude_warning") or "")
    )
    current_scale_warning = current_unit_warning or current_magnitude_warning
    current_scale_review_needed = current_unit_review_needed or bool(current_magnitude_warning)
    cv_peak_status_for_quality = cv_analysis_status(
        selected_oxidation_peak,
        selected_reduction_peak,
        metrics,
        selected_potential_v,
        cycle_assignment_uncertain,
        peaks,
        cv_behavior,
    )
    cv_overall_status_for_quality = overall_analysis_status(
        cv_peak_status_for_quality,
        str(detailed_parser_details.get("current_unit_confidence") or "High"),
        parser_confidence,
        current_scale_warning,
    )
    with cv_summary_container:
        cv_scan_mode_label = "Forward/reverse" if len(split_scan_segments(selected_potential_v)) >= 2 else "Single scan"
        st.markdown(
            f"<div class=\"analysis-file\">{escape_html(format_cv_run_context(current_display_unit, current_source, cv_scan_mode_label, detailed_parser_details.get('cycle_count'), selected_cycle_summary_label, cycle_assignment_uncertain))}</div>",
            unsafe_allow_html=True,
        )
        data_quality_items, data_quality_review = cv_data_quality_items(
            cv_overall_status_for_quality,
            current_unit_review_needed,
            quality_messages,
            parser_confidence,
            current_scale_review_needed,
            cv_behavior,
        )
        render_data_quality_strip(data_quality_items, data_quality_review)
        render_cv_analysis_summary(
            metrics,
            selected_oxidation_peak,
            selected_reduction_peak,
            current_display_unit,
            electrode_area_cm2,
            current_source,
            selected_potential_v,
            detailed_parser_details["scan_rate_estimate"],
            str(detailed_parser_details.get("scan_rate_source") or ""),
            detailed_parser_details["cycle_count"],
            selected_cycle_suffix,
            cycle_assignment_uncertain,
            selected_cycle_summary_label,
            None,
            None,
            str(detailed_parser_details.get("current_unit_confidence") or "High"),
            parser_confidence,
            current_scale_warning,
            peaks,
            cv_behavior,
        )
        render_metadata_status_notice(cv_metadata_summary_from_parser_details(detailed_parser_details, current_source))
        recommended_messages = list(quality_messages)
        if current_scale_review_needed and not current_unit_review_needed:
            recommended_messages.append(
                "Current-dependent metrics may be incorrectly scaled until current unit is confirmed."
            )
        render_recommended_actions(
            recommended_messages,
            key_prefix=f"cv_actions::{active_project_name}::{experiment_id}",
        )

    fig = go.Figure()
    trace_mode = "lines+markers" if show_markers else "lines"

    def add_cv_trace(
        x_values: np.ndarray,
        y_values: np.ndarray,
        name: str,
        opacity: float = 1.0,
        line_width: float = 2.0,
        line_dash: Optional[str] = None,
    ) -> None:
        line_style: dict[str, object] = {"width": line_width}
        if line_dash:
            line_style["dash"] = line_dash
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode=trace_mode,
                name=name,
                opacity=opacity,
                line=line_style,
                hovertemplate=(
                    f"Potential: %{{x:.4g}} V<br>{current_hover_label(current_display_unit)}: %{{y:.4g}} "
                    + display_unit_label(current_display_unit)
                    + "<extra></extra>"
                ),
            )
        )

    for dataset in analyzed:
        dataset_cycle_values = available_cycle_values(dataset.cycle_values)
        is_selected_file = dataset.filename == detailed_dataset.filename
        main_current_a = dataset.smoothed_current_a if dataset.smoothed_current_a is not None else dataset.raw_current_a

        if len(dataset_cycle_values) > 1:
            cycles_to_plot = dataset_cycle_values
            if is_selected_file and cycle_plot_mode == "Show selected cycle only" and selected_cycle_value is not None:
                cycles_to_plot = [selected_cycle_value]

            for cycle_value in cycles_to_plot:
                cycle_mask = cycle_selection_mask(dataset.cycle_values, cycle_value, len(dataset.potential_v))
                if not np.any(cycle_mask):
                    continue

                is_selected_cycle = (
                    is_selected_file
                    and selected_cycle_value is not None
                    and np.isclose(cycle_value, selected_cycle_value, rtol=0.0, atol=1e-9)
                )
                cycle_name = selected_cycle_trace_name(cycle_value, is_selected_cycle)
                trace_name = cycle_name if is_selected_file else f"{dataset.sample_name} {cycle_name}"
                opacity, line_width = selected_cycle_trace_style(is_selected_cycle)

                if dataset.smoothed_current_a is not None and show_raw_with_smoothed:
                    add_cv_trace(
                        dataset.potential_v[cycle_mask],
                        current_to_display(
                            dataset.raw_current_a[cycle_mask],
                            current_display_unit,
                            dataset.current_source,
                            electrode_area_cm2,
                        ),
                        f"{trace_name} raw",
                        opacity=0.22 if is_selected_cycle else 0.14,
                        line_width=1.2,
                        line_dash="dot",
                    )

                add_cv_trace(
                    dataset.potential_v[cycle_mask],
                    current_to_display(
                        main_current_a[cycle_mask],
                        current_display_unit,
                        dataset.current_source,
                        electrode_area_cm2,
                    ),
                    trace_name,
                    opacity=opacity,
                    line_width=line_width,
                )
        else:
            raw_display = current_to_display(
                dataset.raw_current_a,
                current_display_unit,
                dataset.current_source,
                electrode_area_cm2,
            )
            if dataset.smoothed_current_a is not None and show_raw_with_smoothed:
                add_cv_trace(
                    dataset.potential_v,
                    raw_display,
                    f"{dataset.sample_name} raw",
                    opacity=0.35,
                    line_width=1.4,
                )
            elif dataset.smoothed_current_a is None:
                add_cv_trace(dataset.potential_v, raw_display, dataset.sample_name)

            if dataset.smoothed_current_a is not None:
                smoothed_display = current_to_display(
                    dataset.smoothed_current_a,
                    current_display_unit,
                    dataset.current_source,
                    electrode_area_cm2,
                )
                add_cv_trace(dataset.potential_v, smoothed_display, f"{dataset.sample_name} smoothed")

    if show_all_candidate_peaks and candidate_source_peaks:
        candidate_marker_current = full_analysis_current_a if include_all_cycle_candidates else analysis_current_a
        oxidation_marker_peaks = [peak for peak in candidate_source_peaks if peak.peak_type == "oxidation"]
        reduction_marker_peaks = [peak for peak in candidate_source_peaks if peak.peak_type == "reduction"]
        for marker_peaks, symbol, name in [
            (oxidation_marker_peaks, "triangle-up", "Candidate oxidation peaks"),
            (reduction_marker_peaks, "triangle-down", "Candidate reduction peaks"),
        ]:
            if not marker_peaks:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[peak.potential for peak in marker_peaks],
                    y=current_to_display(
                        np.array([candidate_marker_current[peak.index] for peak in marker_peaks]),
                        current_display_unit,
                        current_source,
                        electrode_area_cm2,
                    ),
                    mode="markers",
                    name=name,
                    marker={"size": 10, "symbol": symbol},
                    hovertemplate=(
                        f"Potential: %{{x:.4g}} V<br>{current_hover_label(current_display_unit)}: %{{y:.4g}} "
                        + display_unit_label(current_display_unit)
                        + "<extra></extra>"
                    ),
                )
            )

    selected_peaks = [peak for peak in [selected_oxidation_peak, selected_reduction_peak] if peak]
    if selected_peaks:
        fig.add_trace(
            go.Scatter(
                x=[peak.potential for peak in selected_peaks],
                y=current_to_display(
                    np.array([analysis_current_a[peak.index] for peak in selected_peaks]),
                    current_display_unit,
                    current_source,
                    electrode_area_cm2,
                ),
                mode="markers+text",
                name=selected_peak_legend_label(cv_behavior),
                text=["Epa" if peak.peak_type == "oxidation" else "Epc" for peak in selected_peaks],
                textposition="top center",
                marker={"size": 14, "symbol": "star", "line": {"width": 1}},
            )
        )

    if (
        show_all_detected_couples
        and cv_behavior.behavior == "multiple_redox_couples"
        and cv_behavior.redox_couples
    ):
        selected_couple_index = cv_behavior.selected_couple_index or 0
        for idx, couple in enumerate(cv_behavior.redox_couples):
            if idx == selected_couple_index:
                continue
            couple_peaks = [couple.oxidation_peak, couple.reduction_peak]
            fig.add_trace(
                go.Scatter(
                    x=[peak.potential for peak in couple_peaks],
                    y=current_to_display(
                        np.array([analysis_current_a[peak.index] for peak in couple_peaks]),
                        current_display_unit,
                        current_source,
                        electrode_area_cm2,
                    ),
                    mode="markers+text",
                    name=couple.id,
                    text=["Epa", "Epc"],
                    textposition="bottom center",
                    marker={"size": 10, "symbol": "diamond-open", "line": {"width": 1}},
                )
            )

    if baseline_current is not None:
        fig.add_trace(
            go.Scatter(
                x=selected_potential_v,
                y=current_to_display(baseline_current, current_display_unit, current_source, electrode_area_cm2),
                mode="lines",
                name="Linear baseline",
                line={"dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[selected_potential_v[anchor_a_idx], selected_potential_v[anchor_b_idx]],
                y=current_to_display(
                    np.array([analysis_current_a[anchor_a_idx], analysis_current_a[anchor_b_idx]]),
                    current_display_unit,
                    current_source,
                    electrode_area_cm2,
                ),
                mode="markers",
                name="Baseline anchors",
                marker={"size": 12, "symbol": "x"},
            )
        )

    if reverse_x_axis:
        fig.update_xaxes(autorange="reversed")

    fig.update_layout(
        xaxis_title="Potential / V",
        yaxis_title=current_axis_label(current_display_unit),
        template="plotly_white",
        height=620,
        margin={"l": 72, "r": 28, "t": 24, "b": 72},
        legend_title_text="Trace",
    )

    with cv_plot_container:
        st.subheader("Interactive CV plot")
        cv_plot_quantity = "Current density" if current_source == "Current density" else "Current"
        st.caption(f"{cv_plot_quantity} vs potential")
        if current_unit_review_needed:
            st.info(
                "CV plot paused until current unit is confirmed. Potential-dependent peak positions "
                "were estimated, but current scale requires unit confirmation."
            )
        else:
            st.plotly_chart(fig, width="stretch")

            plot_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            st.download_button(
                "Download interactive plot as HTML",
                data=plot_html.encode("utf-8"),
                file_name="voltscope_cv_plot.html",
                mime="text/html",
                key=f"download_cv_plot::{active_project_name}::{experiment_id}::{selected_filename}",
                on_click="ignore",
            )
        cv_metadata_caution_needed = any(
            [
                detailed_parser_details.get("reference_electrode_metadata") == "Not detected",
                detailed_parser_details.get("electrode_area_metadata") == "Not detected",
                detailed_parser_details.get("scan_rate_estimate") == "Not available",
            ]
        )
        cv_quick_interpretation = cv_interpretation_text(
            cv_behavior,
            metrics,
            str(detailed_parser_details.get("current_unit_confidence") or "High"),
            cv_metadata_caution_needed,
        )
        cv_ai_warnings = compact_text_list(
            quality_messages
            + [
                current_scale_warning,
                "Cycle assignment was inferred from potential turning points and may be approximate."
                if cycle_assignment_uncertain
                else "",
            ]
        )
        render_interpretation_panel(
            cv_quick_interpretation,
            key=f"cv_ai_interpretation_placeholder::{active_project_name}::{experiment_id}",
            ai_payload=build_cv_ai_interpretation_payload(
                filename=selected_filename,
                quick_interpretation=cv_quick_interpretation,
                cv_behavior=cv_behavior,
                metrics=metrics,
                oxidation_peak=selected_oxidation_peak,
                reduction_peak=selected_reduction_peak,
                display_current_unit=current_display_unit,
                current_source=current_source,
                electrode_area_cm2=electrode_area_cm2,
                analysis_status=cv_overall_status_for_quality,
                parser_confidence=parser_confidence,
                unit_confidence=str(detailed_parser_details.get("current_unit_confidence") or "High"),
                peak_detection_confidence=cv_behavior.pair_confidence,
                warnings=cv_ai_warnings,
                scan_info={
                    "scan_mode": cv_scan_mode_label,
                    "scan_rate": detailed_parser_details.get("scan_rate_estimate"),
                    "scan_rate_source": detailed_parser_details.get("scan_rate_source"),
                    "cycle_count": detailed_parser_details.get("cycle_count"),
                    "analyzed_cycle": selected_cycle_summary_label,
                    "cycle_assignment_uncertain": cycle_assignment_uncertain,
                },
                metadata=detailed_dataset.metadata,
                notes=str(experiment.get("notes", "")),
            ),
            project_workspace=project_workspace,
            experiment_id=experiment_id,
            interpretation_id=f"cv::{selected_filename}",
        )

    with esw_expander:
        threshold_display = st.number_input(
            f"ESW threshold ({display_unit_label(current_display_unit)})",
            min_value=0.0,
            value=float(max(
                current_range * 0.1,
                current_to_display(np.array([1e-12]), current_display_unit, current_source, electrode_area_cm2)[0],
            )),
            step=float(max(
                current_range * 0.05,
                current_to_display(np.array([1e-12]), current_display_unit, current_source, electrode_area_cm2)[0],
            )),
            format="%.6e",
        )
        threshold_a = current_display_value_to_base(threshold_display, current_display_unit, current_source, electrode_area_cm2)
        esw_rows = []
        for dataset in analyzed:
            result = calculate_esw(dataset.potential_v, dataset.raw_current_a, threshold_a)
            esw_rows.append(
                {
                    "sample_name": dataset.sample_name,
                    "file_name": dataset.filename,
                    f"threshold_{current_display_unit}": threshold_display,
                    "cathodic_limit_V": result.cathodic_limit,
                    "anodic_limit_V": result.anodic_limit,
                    "operational_ESW_V": result.esw,
                }
            )
        esw_df = pd.DataFrame(esw_rows)
        st.dataframe(esw_df, width="stretch", hide_index=True)

    with cv_metrics_container:
        if not per_cycle_metrics_df.empty:
            st.subheader("Per-cycle CV metrics")
            st.caption(
                "Each row is analyzed independently using the current peak-selection settings. "
                "The selected cycle row is marked and should match the main CV Analysis Summary."
            )

            def highlight_selected_cycle(row: pd.Series) -> list[str]:
                if row.get("analyzed") == "Yes":
                    return ["background-color: #fff7ed; font-weight: 600;" for _ in row]
                return ["" for _ in row]

            st.dataframe(
                per_cycle_metrics_df.style.hide(axis="index").apply(highlight_selected_cycle, axis=1),
                width="stretch",
            )
            if per_cycle_trend_messages:
                st.markdown("**Trend summary**")
                for trend_message in per_cycle_trend_messages:
                    st.caption(trend_message)

        st.subheader("Selected peak audit")
        if peak_metric_rows:
            peak_metric_df = pd.DataFrame(peak_metric_rows)
            compact_metrics_df = metrics_table(
                metrics,
                current_display_unit,
                electrode_area_cm2,
                current_source,
                cv_behavior.behavior,
                cv_behavior.pair_confidence,
            )
            if current_scale_review_needed and not compact_metrics_df.empty:
                current_dependent_metrics = compact_metrics_df["metric"].isin(["Ipa", "Ipc", "|Ipa/Ipc|"])
                if current_unit_review_needed:
                    compact_metrics_df.loc[current_dependent_metrics, "value"] = "Pending unit confirmation"
                compact_metrics_df.loc[current_dependent_metrics, "method"] = (
                    compact_metrics_df.loc[current_dependent_metrics, "method"]
                    + "; current unit needs confirmation"
                )
            selected_peak_display_df = selected_peak_metrics_display_table(
                peak_metric_df,
                current_display_unit,
                baseline_current is not None,
                review_needed=(
                    cv_overall_status_for_quality != "Passed"
                    or cv_behavior.behavior in {"noisy_ambiguous", "no_reliable_peaks"}
                    or cv_pair_metrics_review_reason(metrics, cv_behavior.pair_confidence) is not None
                ),
            )
            if current_unit_review_needed and not selected_peak_display_df.empty:
                current_columns = [
                    column
                    for column in selected_peak_display_df.columns
                    if "current_" in str(column)
                ]
                for column in current_columns:
                    selected_peak_display_df[column] = "Pending unit confirmation"
            render_readable_table(selected_peak_display_df, class_name="readable-table report-table")
            with st.expander("Metric calculation details", expanded=False):
                render_cv_report_table(compact_metrics_df)
        else:
            peak_metric_df = pd.DataFrame()
            compact_metrics_df = pd.DataFrame()
            st.info("Select or manually define peaks to calculate Epa, Ipa, Epc, Ipc, \u0394Ep, and |Ipa/Ipc|.")

    export_buffer = io.StringIO()
    export_sections = []
    export_sections.append("# ESW summary\n" + esw_df.to_csv(index=False))
    if not per_cycle_metrics_df.empty:
        export_sections.append("# Per-cycle CV metrics\n" + per_cycle_metrics_df.to_csv(index=False))
    if peak_metric_rows:
        export_sections.append("# Peak metrics\n" + peak_metric_df.to_csv(index=False))
        export_sections.append("# Compact metrics\n" + compact_metrics_df.to_csv(index=False))
    export_buffer.write("\n".join(export_sections))

    st.download_button(
        "Download analysis results as CSV",
        data=export_buffer.getvalue().encode("utf-8"),
        file_name="voltscope_cv_analysis_results.csv",
        mime="text/csv",
        key=f"download_cv_analysis::{active_project_name}::{experiment_id}",
        on_click="ignore",
    )

    store_experiment_analysis(
        project_workspace,
        experiment_id,
        analyzed,
        parsed_datasets,
        parser_summary,
        esw_df,
        peak_metric_df,
        compact_metrics_df,
        {
            "selected_file": selected_filename,
            "column_mode": column_mode,
            "potential_unit": selected_potential_unit,
            "current_unit": selected_current_unit,
            "current_source": current_source,
            "current_display_unit": current_display_unit,
            "y_axis_quantity": y_axis_quantity,
            "electrode_area_cm2": electrode_area_cm2,
            "smoothing_method": smoothing_method,
            "smoothing_window": smoothing_window,
            "use_smoothed_for_peaks": use_smoothed_for_peaks,
            "peak_min_prominence": min_prominence_display,
            "peak_min_distance": int(min_distance),
            "peak_min_abs_current": min_abs_current_display,
            "baseline_method": baseline_method,
            "esw_threshold": threshold_display,
        },
    )
    render_experiment_notes(active_project_name, project_workspace, experiment_id)


def running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        return get_script_run_ctx() is not None
    finally:
        logger.setLevel(previous_level)


if __name__ == "__main__":
    if running_under_streamlit():
        main()
    else:
        print("Run VoltScope AI with: streamlit run app/streamlit_app.py")
