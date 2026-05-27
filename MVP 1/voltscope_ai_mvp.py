"""
VoltScope AI MVP
----------------
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
import json
import re
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


def metadata_rows_for_display(metadata: dict[str, str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    parsed_rows: list[dict[str, str]] = []
    unparsed_rows: list[dict[str, str]] = []
    for key, value in metadata.items():
        if key.startswith("metadata_line_"):
            line_number = key.rsplit("_", 1)[-1]
            unparsed_rows.append({"Line": line_number, "Text": value})
        else:
            parsed_rows.append({"Field": key, "Value": value})
    return parsed_rows, unparsed_rows


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
        "user": "user-selected",
    }
    source_text = source_labels.get(source, source)
    return f"{reference_electrode} ({source_text})"


def reference_electrode_default_note(reference_electrode: Optional[str], source: str) -> str:
    if reference_electrode and source == "default":
        return f"Reference electrode was not found in the file; using default {reference_electrode}."
    return ""


def find_table_bounds(lines: list[str], delimiter: str) -> tuple[int, int, list[str], bool]:
    for idx, line in enumerate(lines):
        tokens = split_line(line, delimiter)
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

    raise ValueError("Could not find a numeric data table in the uploaded file.")


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

    suggested_unit = suggested_current_unit_for_values(current_values)
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
        return (
            "Current unit may be wrong. The current column has no unit label and values are "
            f"unusually large if interpreted as {display_unit_label(interpreted_current_unit)}."
        )
    return ""


def current_unit_review_explanation(current_col_label: str, suggested_unit: str, current_unit_warning: str) -> str:
    if current_unit_warning:
        return (
            "The current column has no unit in the header. If interpreted as A, "
            "the current values are unusually large for a typical CV/LSV."
        )
    if suggested_unit == "A":
        return (
            "The current column has no unit in the header. Based on value magnitude, "
            "A is the most likely input unit. Please confirm before using current-dependent metrics."
        )
    return (
        f"The current column '{current_col_label}' has no unit in the header. Based on value magnitude, "
        f"{display_unit_label(suggested_unit)} is the most likely input unit. "
        "Please confirm before using current-dependent metrics."
    )


def current_unit_review_reason(suggested_unit: str, current_unit_warning: str) -> str:
    if current_unit_warning:
        return "Suggested based on value magnitude; interpreting as A gives unusually large currents."
    if suggested_unit == "A":
        return "Suggested based on value magnitude; A gives microamp-scale currents."
    return f"Suggested based on value magnitude; {display_unit_label(suggested_unit)} gives a plausible current scale."


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
) -> str:
    if selected_current_unit == "Auto":
        return ""
    if current_header_has_explicit_unit(current_col):
        return ""
    return (
        "Column units were missing from header. Current unit was confirmed by user as "
        f"{display_unit_label(selected_current_unit)}."
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


def confidence_rank(confidence: str) -> int:
    ranks = {
        "low": 0,
        "medium": 1,
        "medium-high": 2,
        "high": 3,
        "manual": 4,
    }
    return ranks.get(confidence, 0)


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
    expected_direction = expected_peak_direction(peak_type)
    correct_segment = expected_direction is not None and segment_direction == expected_direction
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
        return "high"
    if score >= 5:
        return "medium-high"
    if score >= 3:
        return "medium"
    return "low"


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
    return filtered


def default_primary_peak_index(peaks: list[Peak], peak_type: str) -> int:
    expected_direction = expected_peak_direction(peak_type)

    def sort_key(index: int) -> tuple[bool, int, float, float]:
        peak = peaks[index]
        if peak_type == "oxidation":
            peak_height = float(peak.raw_current)
        else:
            peak_height = float(-peak.raw_current)
        return (
            peak.segment_direction == expected_direction,
            confidence_rank(peak.confidence),
            peak_height,
            float(peak.prominence),
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
                "method": (
                    "Maximum current on forward scan"
                    if metric_name == "oxidation"
                    else "Minimum current on reverse scan"
                ),
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


def metrics_table(
    metrics: dict[str, Optional[float]],
    display_unit: str,
    electrode_area_cm2: float = 1.0,
    current_source: str = "Current",
) -> pd.DataFrame:
    rows = []
    display_label = display_unit_label(display_unit)
    if metrics["epa_V"] is not None:
        rows.append({"metric": "Epa", "value": f"{metrics['epa_V']:.4g} V", "method": "Maximum current on forward scan"})
    if metrics["ipa_A"] is not None:
        ipa_display = current_to_display(np.array([metrics["ipa_A"]]), display_unit, current_source, electrode_area_cm2)[0]
        rows.append({"metric": "Ipa", "value": f"{ipa_display:.4g} {display_label}", "method": "Current at Epa"})
    if metrics["epc_V"] is not None:
        rows.append({"metric": "Epc", "value": f"{metrics['epc_V']:.4g} V", "method": "Minimum current on reverse scan"})
    if metrics["ipc_A"] is not None:
        ipc_display = current_to_display(np.array([metrics["ipc_A"]]), display_unit, current_source, electrode_area_cm2)[0]
        rows.append({"metric": "Ipc", "value": f"{ipc_display:.4g} {display_label}", "method": "Current at Epc"})
    if metrics["delta_ep_V"] is not None:
        rows.append({"metric": "\u0394Ep", "value": f"{metrics['delta_ep_V'] * 1000:.4g} mV", "method": "Epa - Epc"})
    if metrics["epa_V"] is not None and metrics["epc_V"] is not None:
        formal_potential = (metrics["epa_V"] + metrics["epc_V"]) / 2
        rows.append({"metric": "E\u00b0\u2032", "value": f"{formal_potential:.4g} V", "method": "(Epa + Epc) / 2"})
    if metrics["ipa_ipc_ratio"] is not None:
        rows.append({"metric": "|Ipa/Ipc|", "value": f"{metrics['ipa_ipc_ratio']:.4g}", "method": "abs(Ipa/Ipc)"})
    return pd.DataFrame(rows)


def selected_peak_metrics_display_table(
    peak_metric_df: pd.DataFrame,
    display_unit: str,
    baseline_active: bool,
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

    return display_df[[column for column in columns if column in display_df.columns]]


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


def cv_analysis_quality_messages(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    metrics: Optional[dict[str, Optional[float]]] = None,
    potential: Optional[np.ndarray] = None,
) -> list[str]:
    messages = []
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
    return messages


def cv_analysis_status(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    metrics: dict[str, Optional[float]],
    potential: np.ndarray,
    cycle_assignment_uncertain: bool = False,
) -> str:
    if oxidation_peak is None and reduction_peak is None:
        return "Failed"
    if oxidation_peak is None or reduction_peak is None:
        return "Incomplete"
    if cycle_assignment_uncertain:
        return "Review recommended"
    if cv_analysis_quality_messages(oxidation_peak, reduction_peak, metrics, potential):
        return "Review recommended"
    return "Passed"


def parser_confidence_from_details(details: dict[str, Any]) -> str:
    if details.get("potential_column") in {None, "Not detected"}:
        return "Low"
    if details.get("current_column") in {None, "Not detected"}:
        return "Low"
    warnings = str(details.get("warnings", "None")).strip()
    if warnings and warnings != "None":
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
) -> str:
    status = cv_analysis_status(oxidation_peak, reduction_peak, metrics, potential, cycle_assignment_uncertain)
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
        else:
            cycle_min_prominence = min_prominence_a
            cycle_min_distance = int(min_distance)
            cycle_min_abs_current = min_abs_current_a
            if auto_tune_each_cycle:
                cycle_params = auto_tune_cv_peak_parameters(cycle_potential, cycle_current)
                cycle_min_prominence = float(cycle_params["min_prominence"])
                cycle_min_distance = int(cycle_params["min_distance"])
                cycle_min_abs_current = float(cycle_params["min_abs_current"])

            peaks = detect_peaks(
                cycle_potential,
                cycle_current,
                cycle_min_prominence,
                cycle_min_distance,
                cycle_min_abs_current,
            )
            oxidation_peak, reduction_peak = select_default_primary_peaks(peaks)
            _peak_rows, metrics = build_peak_metrics_rows(
                oxidation_peak,
                reduction_peak,
                None,
                cycle_current,
                display_unit,
                electrode_area_cm2,
                current_source,
            )

        epa = metrics.get("epa_V")
        epc = metrics.get("epc_V")
        ipa = metrics.get("ipa_A")
        ipc = metrics.get("ipc_A")
        delta_ep = metrics.get("delta_ep_V")
        formal_potential = (epa + epc) / 2 if epa is not None and epc is not None else None
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
                "\u0394Ep (mV)": "Not detected" if delta_ep is None else f"{delta_ep * 1000:.4g}",
                "E\u00b0\u2032 (V)": "Not detected" if formal_potential is None else f"{formal_potential:.4g}",
                "|Ipa/Ipc|": (
                    "Not detected"
                    if metrics.get("ipa_ipc_ratio") is None
                    else f"{metrics['ipa_ipc_ratio']:.4g}"
                ),
                "analysis quality": cv_analysis_quality_label(
                    oxidation_peak,
                    reduction_peak,
                    metrics,
                    cycle_potential,
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
        formatted = f"{value:.0f}" if abs(value) >= 10 else f"{value:.3g}"
        return f"{formatted} mV/s"
    return scan_rate_estimate


def render_cv_analysis_summary(
    metrics: dict[str, Optional[float]],
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    display_unit: str,
    electrode_area_cm2: float,
    current_source: str,
    potential: np.ndarray,
    scan_rate_estimate: str = "Not available",
    cycle_count: str = "Not detected",
    title_suffix: str = "",
    cycle_assignment_uncertain: bool = False,
    analyzed_cycle_label: Optional[str] = None,
    analysis_status_override: Optional[str] = None,
    analysis_quality_override: Optional[str] = None,
    unit_confidence: str = "High",
    parser_confidence: str = "High",
    current_unit_warning: str = "",
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
    scan_rate_label = summarize_scan_rate_for_card(scan_rate_estimate)
    cycle_count_label = None
    if cycle_count and not str(cycle_count).startswith("Not detected"):
        cycle_count_label = str(cycle_count)

    confidence_values = [
        peak.confidence for peak in [oxidation_peak, reduction_peak] if peak is not None
    ]
    if oxidation_peak is None or reduction_peak is None:
        confidence_label = "Incomplete"
    elif len(confidence_values) == 2 and all(value == "high" for value in confidence_values):
        confidence_label = "High"
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

    title = f"CV Analysis Summary{title_suffix}"
    summary_items = [
        ("Analysis status", status),
        ("Peak detection confidence", confidence_label),
        ("Unit confidence", unit_confidence),
        ("Parser confidence", parser_confidence),
        ("Epa", format_potential_metric(epa)),
        ("Ipa", ipa_value),
        ("Epc", format_potential_metric(epc)),
        ("Ipc", ipc_value),
        ("\u0394Ep", "Not detected" if delta_ep_mv is None else f"{delta_ep_mv:.4g} mV"),
        ("E\u00b0\u2032", format_potential_metric(formal_potential)),
        ("|Ipa/Ipc|", ratio_value),
        ("Overall quality", analysis_quality_override or status),
        ("Scan mode", scan_mode),
    ]
    if switching_potential:
        summary_items.append(("Switching potential", switching_potential))
    if scan_rate_label:
        summary_items.append(("Scan rate", scan_rate_label))
    if analyzed_cycle_label:
        summary_items.append(("Analyzed cycle", analyzed_cycle_label))
    elif cycle_count_label:
        summary_items.append(("Cycle count", cycle_count_label))

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

    direction = lsv_scan_direction(potential_v)
    onset_potential = None
    onset_direction = direction
    if threshold_a is not None and threshold_a > 0:
        result = calculate_esw(potential_v, current_values, threshold_a)
        onset_potential, onset_direction = select_lsv_onset(result, direction)

    status = "Passed" if onset_potential is not None else "Review needed"
    review_note = ""
    if onset_potential is None:
        review_note = "No threshold crossing was detected in the uploaded potential range. Review the threshold or uploaded data."
        method_note = review_note
    else:
        signed_threshold = -threshold_a if onset_direction == "cathodic" else threshold_a
        threshold_note = format_lsv_threshold(signed_threshold, current_unit, current_source, electrode_area_cm2)
        reference_note = f" Reported vs {reference_electrode}." if reference_electrode else ""
        method_note = (
            f"Onset = linearly interpolated potential where {onset_direction} "
            f"{current_label} first crosses {threshold_note}.{reference_note}"
        )
    if unit_review_needed:
        status = "Review needed"
        review_note = "Current-dependent metrics may be incorrectly scaled until current unit is confirmed."

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
    if parsed_dataset is not None:
        potential_col = record.get("columns", {}).get("potential")
        scan_rate_estimate = scan_rate_estimate_summary(parsed_dataset, potential_col)
        scan_rate = summarize_scan_rate_for_card(scan_rate_estimate) or "Not available"
        if scan_rate == "Not available":
            scan_rate_note = scan_rate_unavailable_reason(parsed_dataset, potential_col)

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

    return {
        "status": status,
        "review_note": review_note,
        "onset_label": onset_label,
        "onset_potential": format_lsv_potential(onset_potential, reference_electrode),
        "onset_method": f"{method_quantity} threshold crossing",
        "threshold": format_lsv_threshold(threshold_a, current_unit, current_source, electrode_area_cm2),
        "threshold_source": threshold_source,
        "threshold_rule": threshold_rule,
        "method_note": method_note,
        "direction": summary_direction.title(),
        "primary_current_label": primary_current_label,
        "primary_current_value": format_lsv_current_value(primary_current, current_unit, current_source, electrode_area_cm2),
        "primary_potential_label": primary_potential_label,
        "primary_potential": format_lsv_potential(primary_potential, reference_electrode),
        "max_label": f"Max {current_label}",
        "max_value": format_lsv_current_value(max_current, current_unit, current_source, electrode_area_cm2),
        "max_potential": format_lsv_potential(max_potential, reference_electrode),
        "min_label": f"Min {current_label}",
        "min_value": format_lsv_current_value(min_current, current_unit, current_source, electrode_area_cm2),
        "min_potential": format_lsv_potential(min_potential, reference_electrode),
        "potential_range": format_fixed_range(finite_potential, "V", 3),
        "scan_rate": scan_rate,
        "scan_rate_note": scan_rate_note,
        "reference_electrode": reference_electrode_source_label(reference_electrode, reference_electrode_source),
        "reference_note": reference_electrode_default_note(reference_electrode, reference_electrode_source),
    }


def render_lsv_analysis_summary(summary: dict[str, str]) -> None:
    status = summary.get("status", "Review needed")
    status_class = status_badge_class(status)
    summary_items = [
        ("Analysis status", status),
        (summary.get("onset_label", "Onset potential"), summary.get("onset_potential", "Not detected")),
        ("Onset method", summary.get("onset_method", "Threshold crossing")),
        ("Threshold", summary.get("threshold", "Not available")),
        ("Threshold source", summary.get("threshold_source", "User-defined")),
        ("Threshold rule", summary.get("threshold_rule", "User-defined threshold.")),
        ("Direction", summary.get("direction", "Not available")),
        (summary.get("primary_current_label", "Max current"), summary.get("primary_current_value", "Not available")),
        (summary.get("primary_potential_label", "Potential at max current"), summary.get("primary_potential", "Not available")),
        ("Potential range", summary.get("potential_range", "Not available")),
        ("Scan rate", summary.get("scan_rate", "Not available")),
        ("Reference electrode", summary.get("reference_electrode", "Not selected")),
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
                <h4>LSV Analysis Summary</h4>
                <span class="status-badge status-{escape_html(status_class)}">{escape_html(status)}</span>
            </div>
            <div class="analysis-grid">{metric_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if summary.get("method_note"):
        st.caption(summary["method_note"])
    if summary.get("reference_note"):
        st.caption(summary["reference_note"])
    if summary.get("scan_rate_note"):
        st.info(summary["scan_rate_note"])
    if summary.get("review_note"):
        st.warning(summary["review_note"])


def format_lsv_display_state(
    display_units: dict[str, Any],
    threshold_a: Optional[float],
    reference_electrode: Optional[str],
    electrode_area_metadata: Optional[str] = None,
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
    else:
        threshold_text = format_lsv_threshold(
            threshold_a,
            display_units.get("current", "mA/cm^2"),
            current_source,
            electrode_area_cm2,
        )
    reference_text = reference_electrode or "Not selected"

    if current_source == "Current density":
        summary = (
            "Display: Current density from uploaded column, "
            f"threshold = {threshold_text}, reference = {reference_text}."
        )
        if electrode_area_metadata:
            summary += (
                f" Electrode area metadata: {electrode_area_metadata}. "
                "No additional normalization applied."
            )
        return summary

    if display_quantity == "Current":
        return (
            f"Display: Current, threshold = {threshold_text}, "
            f"reference = {reference_text}."
        )

    area_text = f"{electrode_area_cm2:.1f}" if electrode_area_cm2 == round(electrode_area_cm2) else f"{electrode_area_cm2:g}"
    return (
        f"Display: {display_quantity}, area = {area_text} cm\u00b2, "
        f"threshold = {threshold_text}, reference = {reference_text}."
    )


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

    direction = lsv_scan_direction(potential_v)
    result = calculate_esw(potential_v, current_values, threshold_a) if threshold_a and threshold_a > 0 else ESWResult(None, None, None)
    onset_potential, onset_direction = select_lsv_onset(result, direction)
    onset_direction = onset_direction or direction
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
    if parsed_dataset is not None:
        potential_col = record.get("columns", {}).get("potential")
        scan_rate_estimate = scan_rate_estimate_summary(parsed_dataset, potential_col)
        scan_rate = summarize_scan_rate_for_card(scan_rate_estimate) or "Not available"
        if scan_rate == "Not available":
            scan_rate_note = scan_rate_unavailable_reason(parsed_dataset, potential_col)

    direction_method = "Potential increases from start to end." if direction == "anodic" else "Potential decreases from start to end."
    threshold_method = threshold_rule
    crossing_method = f"Linear interpolation where {current_label} crosses the selected threshold."
    rows = [
        {
            "Metric": onset_label,
            "Value": format_lsv_potential(onset_potential, reference_electrode),
            "Method": crossing_method,
        },
        {
            "Metric": "Threshold",
            "Value": format_lsv_threshold(threshold_a, current_unit, current_source, electrode_area_cm2),
            "Method": threshold_method,
        },
        {
            "Metric": "Threshold source",
            "Value": threshold_source,
            "Method": threshold_rule,
        },
        {
            "Metric": "Onset method",
            "Value": "Current-density threshold crossing" if is_density else "Current threshold crossing",
            "Method": "Uses the selected threshold and scan direction to select the relevant crossing.",
        },
        {
            "Metric": "Direction",
            "Value": onset_direction.title(),
            "Method": direction_method,
        },
        {
            "Metric": f"Max {current_label_title}",
            "Value": format_lsv_current_value(max_current, current_unit, current_source, electrode_area_cm2),
            "Method": f"Maximum {current_label} in the normalized LSV trace.",
        },
        {
            "Metric": f"Potential at max {current_label_title}",
            "Value": format_lsv_potential(max_potential, reference_electrode),
            "Method": f"Potential value at the maximum {current_label}.",
        },
        {
            "Metric": f"Min {current_label_title}",
            "Value": format_lsv_current_value(min_current, current_unit, current_source, electrode_area_cm2),
            "Method": f"Minimum {current_label} in the normalized LSV trace.",
        },
        {
            "Metric": f"Potential at min {current_label_title}",
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
            "Method": scan_rate_note or "Estimated from dE/dt when time and potential columns are available.",
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

    direction = lsv_scan_direction(potential_v)
    result = calculate_esw(potential_v, current_values, threshold_a)
    onset_potential, onset_direction = select_lsv_onset(result, direction)
    if onset_potential is None or onset_direction is None:
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
    return {
        "onset_potential_v": float(onset_potential),
        "onset_potential_display": float(potential_display),
        "threshold_display": float(current_display),
        "direction": onset_direction,
        "label": onset_label,
        "hover": (
            f"{onset_label}<br>"
            f"Threshold: {current_display:.4g} {display_unit_label(current_unit)}"
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
        detected_source = current_source_guess_for_column(current_col)
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
        excluded = {potential_col} if potential_col else set()
        time_col = detect_column_by_role(dataset.dataframe, "time", excluded=excluded)
    if not time_col or not potential_col or time_col not in dataset.dataframe or potential_col not in dataset.dataframe:
        return "Not available"

    time_values = dataset.dataframe[time_col].to_numpy(dtype=float)
    potential_values = dataset.dataframe[potential_col].to_numpy(dtype=float)
    time_unit = resolve_role_unit_from_values("Auto", dataset, "time", time_col, time_values)
    potential_unit = resolve_role_unit_from_values("Auto", dataset, "potential", potential_col, potential_values)
    time_s = convert_role_to_base(time_values, "time", time_unit)
    potential_v = convert_role_to_base(potential_values, "potential", potential_unit)
    scan_rate = representative_scan_rate_v_s(time_s, potential_v)
    if scan_rate is None:
        return "Not available"
    return f"{format_summary_number(scan_rate)} V/s ({format_summary_number(scan_rate * 1000)} mV/s)"


def scan_rate_unavailable_reason(dataset: ParsedDataset, potential_col: Optional[str]) -> str:
    time_col = dataset.detected_time_col
    if not time_col:
        excluded = {potential_col} if potential_col else set()
        time_col = detect_column_by_role(dataset.dataframe, "time", excluded=excluded)
    if not time_col:
        return "Scan rate could not be estimated because no time column was found."
    if not potential_col or time_col not in dataset.dataframe or potential_col not in dataset.dataframe:
        return "Scan rate could not be estimated because the required time or potential column was unavailable."
    return "Scan rate could not be estimated from the available time and potential values."


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
    detected_current_source = current_source_guess_for_column(current_col)
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
            suggested_current_unit = (
                f"{display_unit_label(suggested_current_unit_for_values(current_values))} "
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
    )
    warnings = "; ".join(dataset.warnings) if dataset.warnings else "None"
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
    scan_rate_note = ""
    if scan_rate_estimate == "Not available":
        scan_rate_note = scan_rate_unavailable_reason(dataset, potential_col)

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
        "unit_conversion": parser_unit_conversion_summary(potential_unit, current_unit, current_source, display_current_unit)
        if potential_col and current_col
        else "Not available",
        "scan_rate_estimate": scan_rate_estimate,
        "scan_rate_note": scan_rate_note,
        "cycle_count": cycle_count_summary(dataset, potential_col),
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
            "cycle_count": details["cycle_count"],
            "warnings": "; ".join(dataset.warnings),
        }
        for role in roles:
            row[f"detected_{role}"] = detected.get(role)
            column = detected.get(role)
            if column and role in {"potential", "current"}:
                values = dataset.dataframe[column].to_numpy(dtype=float)
                row[f"{role}_unit"] = resolve_role_unit_from_values("Auto", dataset, role, column, values)
        if experiment_type in {"cv", "lsv"} and detected.get("current"):
            row["current_source_guess"] = (
                "Current density" if header_indicates_current_density(detected["current"]) else "Current"
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
                current_source_guesses.append(
                    "Current density" if header_indicates_current_density(current_col) else "Current"
                )
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
            row.update(
                {
                    "potential_min_V": float(np.nanmin(potential_v)),
                    "potential_max_V": float(np.nanmax(potential_v)),
                    f"{current_summary_key}_min": float(np.nanmin(current_values)),
                    f"{current_summary_key}_max": float(np.nanmax(current_values)),
                }
            )
            if lsv_threshold_a is not None and lsv_threshold_a > 0:
                result = calculate_esw(potential_v, current_values, lsv_threshold_a)
                direction = lsv_scan_direction(potential_v)
                onset_potential, onset_direction = select_lsv_onset(result, direction)
                row.update(
                    {
                        f"threshold_{lsv_threshold_unit or 'display'}": lsv_threshold_display,
                        "onset_direction": onset_direction or direction,
                        "onset_potential_V": onset_potential,
                        "operational_ESW_V": result.esw,
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
    st.markdown("**Detected metadata**")
    parsed_rows, unparsed_rows = metadata_rows_for_display(metadata)

    if parsed_rows:
        st.table(pd.DataFrame(parsed_rows))
    elif unparsed_rows:
        st.info("No structured metadata fields were detected.")
    else:
        st.write("No metadata rows detected before the numeric table.")

    if unparsed_rows:
        with st.expander("Unparsed metadata lines", expanded=False):
            st.table(pd.DataFrame(unparsed_rows))

    if metadata:
        with st.expander("Advanced/debug raw metadata", expanded=False):
            st.json(metadata)

    st.text_area(
        "Manual file metadata",
        placeholder=(
            "Add sample name, electrode, electrolyte, concentration, scan rate, instrument, "
            "reference electrode, or other file-specific metadata."
        ),
        key=manual_metadata_key,
    )


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

    parser_summary = render_parser_summary(
        parsed_datasets,
        experiment_type,
        key_prefix=f"generic_parser::{project_name}::{experiment_id}",
        display_current_unit=parser_display_current_unit,
        current_source_override=parser_current_source,
        electrode_area_cm2=parser_electrode_area,
    )
    lsv_summary_container = st.container() if is_lsv else None
    lsv_plot_container = st.container() if is_lsv else None
    lsv_metrics_container = st.container() if is_lsv else None
    lsv_controls_container = st.container() if is_lsv else None
    lsv_mapping_container = st.container() if is_lsv else None
    lsv_details_container = st.container() if is_lsv else None
    lsv_controls_expander = (
        lsv_controls_container.expander("LSV Analysis & Display Controls", expanded=False)
        if lsv_controls_container is not None
        else nullcontext()
    )
    mapping_context = (
        lsv_mapping_container.expander("Data Mapping & Units", expanded=False)
        if lsv_mapping_container is not None
        else nullcontext()
    )

    with mapping_context:
        selected_filename = st.selectbox(
            "Preview / detailed analysis file",
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
                selected_units[role] = st.selectbox(
                    f"{role_label(role)} input unit",
                    unit_options_for_role(role),
                    key=f"generic_unit::{project_name}::{experiment_id}::{role}",
                )

    if is_lsv and lsv_details_container is not None:
        with lsv_details_container.expander("Detected Metadata / Data & Processing Details", expanded=False):
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

            current_source_default = 1 if current_col_for_units and header_indicates_current_density(current_col_for_units) else 0
            current_source_key = (
                f"lsv_current_source::{project_name}::{experiment_id}::"
                f"{selected_filename}::{current_col_for_units or 'current'}"
            )
            if current_source_default == 1 and st.session_state.get(current_source_key) != "Current density":
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
                st.info("The uploaded current column is already normalized, so electrode-area normalization is skipped.")
                detected_area_metadata = electrode_area_metadata_summary(selected_dataset.metadata)
                if detected_area_metadata:
                    st.info(
                        f"Electrode area metadata: {detected_area_metadata}. "
                        "No additional normalization applied."
                    )
            elif uses_current_density_axis:
                display_units["electrode_area_cm2"] = st.number_input(
                    "Electrode surface area (cm^2)",
                    min_value=0.000001,
                    value=1.0,
                    step=0.1,
                    format="%.6f",
                    key=f"lsv_electrode_area::{project_name}::{experiment_id}",
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
                st.markdown(
                    f"<div class=\"analysis-file\">{escape_html(format_lsv_display_state(display_units, lsv_threshold_a, reference_electrode, selected_metadata_area))}</div>",
                    unsafe_allow_html=True,
                )
                render_lsv_analysis_summary(
                    build_lsv_analysis_summary(
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
                )

    plot_context = lsv_plot_container if lsv_plot_container is not None else nullcontext()
    with plot_context:
        st.subheader(f"Interactive {experiment_short_label(experiment_type)} plot")
        fig = go.Figure()
        for record in records:
            add_generic_trace(fig, record, experiment_type, display_units, show_markers)

        if experiment_type == "lsv" and lsv_threshold_display is not None and lsv_threshold_display > 0:
            if display_units.get("y_axis") in {"Current", "Current density"}:
                fig.add_hline(
                    y=lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=f"+{lsv_threshold_display:g} {display_unit_label(lsv_threshold_unit)}",
                )
                fig.add_hline(
                    y=-lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=f"-{lsv_threshold_display:g} {display_unit_label(lsv_threshold_unit)}",
                )
            elif display_units.get("x_axis") in {"Current", "Current density"}:
                fig.add_vline(
                    x=lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=f"+{lsv_threshold_display:g} {display_unit_label(lsv_threshold_unit)}",
                )
                fig.add_vline(
                    x=-lsv_threshold_display,
                    line_dash="dot",
                    annotation_text=f"-{lsv_threshold_display:g} {display_unit_label(lsv_threshold_unit)}",
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
            data=plot_html,
            file_name=f"voltscope_{experiment_type}_plot.html",
            mime="text/html",
        )

    results_df = build_generic_results(
        records,
        experiment_type,
        lsv_threshold_a=lsv_threshold_a,
        lsv_threshold_display=lsv_threshold_display,
        lsv_threshold_unit=lsv_threshold_unit,
        lsv_current_source=display_units.get("current_source", "Current"),
    )
    metrics_context = lsv_metrics_container if lsv_metrics_container is not None else nullcontext()
    with metrics_context:
        st.subheader("Detailed LSV Metrics" if is_lsv else "Analysis results")
        selected_metrics_df = None
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
            st.table(selected_metrics_df)
            with st.expander("Raw/internal results", expanded=False):
                st.dataframe(results_df, width="stretch", hide_index=True)
        else:
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
            data=export_buffer.getvalue(),
            file_name=f"voltscope_{experiment_type}_analysis_results.csv",
            mime="text/csv",
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
        placeholder="Record observations, electrode prep, electrolyte details, interpretation, or follow-up steps.",
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
        return " A/cm^2"
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
        return f" {key.replace('threshold_', '')}"
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
    current_unit_warning: str,
    confirmed_current_unit_key: str,
    selectbox_key: str,
    button_key: str,
) -> None:
    suggested_label = display_unit_label(suggested_unit)
    explanation = current_unit_review_explanation(current_col_label, suggested_unit, current_unit_warning)
    reason = current_unit_review_reason(suggested_unit, current_unit_warning)
    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-card-header">
                <h4>Unit review required</h4>
                <span class="status-badge status-review-needed">Review needed</span>
            </div>
            <p>{escape_html(explanation)}</p>
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
    confirm_unit_options = ["A", "mA", "uA", "nA", "pA"]
    suggested_value = suggested_unit if suggested_unit in confirm_unit_options else "uA"
    with confirm_col1:
        confirm_choice = st.selectbox(
            "Confirm current unit",
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
            st.rerun()


def render_parser_summary(
    parsed_datasets: list[ParsedDataset],
    experiment_type: str,
    key_prefix: str,
    display_current_unit: Optional[str] = None,
    current_source_override: Optional[str] = None,
    electrode_area_cm2: float = 1.0,
    current_input_unit_override: Optional[str] = None,
) -> pd.DataFrame:
    parser_summary = make_parser_summary(
        parsed_datasets,
        experiment_type,
        display_current_unit,
        current_source_override,
        electrode_area_cm2,
        current_input_unit_override,
    )
    st.subheader("Parser summary")

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
        ("Unit conversion", "unit_conversion"),
        ("Scan rate estimate", "scan_rate_estimate"),
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
        combined_warnings.extend(backend_warnings_by_file.get(dataset.filename, []))
        if details.get("unit_header_warning"):
            combined_warnings.append(str(details["unit_header_warning"]))
        if details.get("current_unit_warning"):
            combined_warnings.append(str(details["current_unit_warning"]))
        if details.get("current_magnitude_warning") and not magnitude_warning_acknowledged:
            combined_warnings.append(str(details["current_magnitude_warning"]))
        unit_review_required = details.get("current_unit_confidence") == "Low"
        details["warnings"] = (
            "Unit review required"
            if unit_review_required
            else "; ".join(combined_warnings) if combined_warnings else "None"
        )

        if unit_review_required:
            current_unit_summary = "Ambiguous"
        else:
            current_unit_summary = str(details["raw_current_unit"])
        if not unit_review_required and details["analysis_current_unit"] != details["raw_current_unit"]:
            current_unit_summary += f" \u2192 analysis {details['analysis_current_unit']}"
        if not unit_review_required and details["display_current_unit"] != details["analysis_current_unit"]:
            current_unit_summary += f" \u2192 displayed as {details['display_current_unit']}"

        y_quantity = str(details.get("y_quantity") or "Current")
        unit_lines = [
            ("Potential", "V"),
            (f"{y_quantity} unit", current_unit_summary),
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
            st.table(pd.DataFrame(advanced_rows))
            st.markdown("**Data preview**")
            st.dataframe(dataframe_preview_for_display(dataset.dataframe).head(30), width="stretch")

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
    st.set_page_config(page_title="VoltScope AI MVP", layout="wide")
    apply_app_theme()

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

    parser_summary_container = st.container()

    selected_filename = st.selectbox(
        "Preview / detailed analysis file",
        [dataset.filename for dataset in parsed_datasets],
    )
    selected_dataset = next(dataset for dataset in parsed_datasets if dataset.filename == selected_filename)

    cv_summary_container = st.container()
    cv_plot_container = st.container()
    cv_metrics_container = st.container()
    peak_controls_expander = st.expander("Peak Selection & Baseline Correction", expanded=False)
    candidate_peaks_expander = st.expander("Detected candidate peaks", expanded=False)
    data_details_expander = st.expander("Data & Processing Details", expanded=False)
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
            )
        with unit_col3:
            y_axis_quantity = st.selectbox("Y-axis quantity", ["Current", "Current density"])

        source_default = 1 if header_indicates_current_density(manual_current_col) else 0
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
            current_display_unit = st.selectbox("Current display unit", ["uA", "mA", "A", "nA"])

        area_required = (
            (current_source == "Current" and is_current_density_unit(current_display_unit))
            or (current_source == "Current density" and not is_current_density_unit(current_display_unit))
        )
        if area_required:
            electrode_area_cm2 = st.number_input(
                "Electrode surface area (cm^2)",
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

    with parser_summary_container:
        current_input_unit_override = selected_current_unit if selected_current_unit != "Auto" else None
        current_unit_is_low_confidence = current_unit_is_ambiguous(
            selected_dataset,
            manual_current_col,
            selected_current_unit,
        )
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
                current_unit_warning=unit_review_warning,
                confirmed_current_unit_key=confirmed_current_unit_key,
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

        parser_summary = render_parser_summary(
            parsed_datasets,
            "cv",
            key_prefix=f"cv_parser::{active_project_name}::{experiment_id}",
            display_current_unit=current_display_unit,
            current_source_override=current_source,
            electrode_area_cm2=electrode_area_cm2,
            current_input_unit_override=current_input_unit_override,
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

    with peak_controls_expander:
        st.subheader("Peak Selection & Baseline Correction")
        st.caption(
            "Method: split the CV into forward and reverse scans, ignore the first/last 5% of each scan, "
            "then select oxidation from the increasing scan maximum and reduction from the decreasing scan minimum."
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
                f"Auto-tuned for this CV: prominence {default_prominence:.4g} {current_display_unit}, "
                f"distance {int(auto_peak_params['min_distance'])} points, minimum absolute current 0 {current_display_unit}."
            )

        peak_setting_mode = "auto" if auto_tune_peak_settings else "manual"
        peak_setting_key_base = (
            f"{active_project_name}::{experiment_id}::{selected_filename}::"
            f"{current_display_unit}::{smoothing_method}::{use_smoothed_for_peaks}::{peak_setting_mode}"
        )
        peak_col1, peak_col2, peak_col3 = st.columns(3)
        with peak_col1:
            min_prominence_widget = st.number_input(
                f"Minimum peak prominence ({current_display_unit})",
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
                f"Minimum absolute current ({current_display_unit})",
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

        selected_primary_rows = []
        for role, peak in [("Selected primary Epa", selected_oxidation_peak), ("Selected primary Epc", selected_reduction_peak)]:
            if peak is None:
                selected_primary_rows.append(
                    {
                        "role": role,
                        "status": "Not selected",
                        "potential_V": None,
                        f"current_{current_display_unit}": None,
                        "scan": None,
                        "confidence": None,
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

        st.markdown("**Selected Epa/Epc used for CV metrics**")
        st.dataframe(pd.DataFrame(selected_primary_rows), width="stretch")

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
            st.dataframe(anchor_summary, width="stretch")

    with candidate_peaks_expander:
        st.caption(
            "Candidate peaks are local extrema retained for transparency. Only the selected Epa/Epc pair is used for CV metrics."
        )
        candidate_display_filter = st.selectbox(
            "Candidate peak display",
            CANDIDATE_PEAK_FILTER_OPTIONS,
            index=CANDIDATE_PEAK_FILTER_OPTIONS.index("Top candidates"),
            key=f"cv_candidate_peak_display::{active_project_name}::{experiment_id}::{selected_filename}",
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
        selected_peak_roles = {}
        if selected_oxidation_peak is not None:
            selected_peak_roles[selected_oxidation_peak.id] = "Selected Epa"
        if selected_reduction_peak is not None:
            selected_peak_roles[selected_reduction_peak.id] = "Selected Epc"

        full_candidate_rows = [
            {
                "role": selected_peak_roles.get(peak.id, "Candidate"),
                "id": peak.id,
                "extremum": local_extremum_label(peak),
                "index": peak.index,
                "potential_V": peak.potential,
                f"current_{current_display_unit}": current_to_display(
                    np.array([peak.raw_current]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                f"prominence_{current_display_unit}": current_to_display(
                    np.array([peak.prominence]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                "scan": scan_direction_label(peak.segment_direction),
                "confidence": peak.confidence,
            }
            for peak in candidate_source_peaks
        ]
        displayed_candidate_peaks = filter_candidate_peaks_for_display(
            candidate_source_peaks,
            selected_oxidation_peak,
            selected_reduction_peak,
            candidate_display_filter,
            potential=candidate_source_potential,
        )
        displayed_candidate_rows = [
            {
                "role": selected_peak_roles.get(peak.id, "Candidate"),
                "id": peak.id,
                "extremum": local_extremum_label(peak),
                "index": peak.index,
                "potential_V": peak.potential,
                f"current_{current_display_unit}": current_to_display(
                    np.array([peak.raw_current]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                f"prominence_{current_display_unit}": current_to_display(
                    np.array([peak.prominence]), current_display_unit, current_source, electrode_area_cm2
                )[0],
                "scan": scan_direction_label(peak.segment_direction),
                "confidence": peak.confidence,
            }
            for peak in displayed_candidate_peaks
        ]
        if displayed_candidate_rows:
            if candidate_display_filter == "Top candidates":
                st.caption(
                    "Showing selected Epa/Epc plus top candidate extrema. "
                    "Download full candidate table for all local extrema."
                )
            else:
                st.caption(f"Showing {len(displayed_candidate_rows)} of {len(full_candidate_rows)} candidate extrema.")
            st.table(pd.DataFrame(displayed_candidate_rows))
        elif full_candidate_rows:
            st.info("No candidate extrema match the selected display filter.")
        else:
            st.info("No candidate peaks were detected with the current settings.")

        if full_candidate_rows:
            full_candidate_csv = pd.DataFrame(full_candidate_rows).to_csv(index=False)
            st.download_button(
                "Download full candidate peaks CSV",
                data=full_candidate_csv,
                file_name=f"{Path(selected_filename).stem}_candidate_peaks.csv",
                mime="text/csv",
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
            auto_tune_peak_settings,
        )

    quality_messages = cv_analysis_quality_messages(
        selected_oxidation_peak,
        selected_reduction_peak,
        metrics,
        selected_potential_v,
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
    with cv_summary_container:
        for message in quality_messages:
            st.warning(message)
        if current_scale_review_needed and not current_unit_review_needed:
            st.warning(
                "Current-dependent metrics may be incorrectly scaled until current unit is confirmed."
            )

        render_cv_analysis_summary(
            metrics,
            selected_oxidation_peak,
            selected_reduction_peak,
            current_display_unit,
            electrode_area_cm2,
            current_source,
            selected_potential_v,
            detailed_parser_details["scan_rate_estimate"],
            detailed_parser_details["cycle_count"],
            selected_cycle_suffix,
            cycle_assignment_uncertain,
            selected_cycle_summary_label,
            None,
            None,
            str(detailed_parser_details.get("current_unit_confidence") or "High"),
            parser_confidence,
            current_scale_warning,
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
                    + current_display_unit
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
                        + current_display_unit
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
                name="Selected Epa/Epc",
                text=["Epa" if peak.peak_type == "oxidation" else "Epc" for peak in selected_peaks],
                textposition="top center",
                marker={"size": 14, "symbol": "star", "line": {"width": 1}},
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
        legend_title_text="Trace",
    )

    with cv_plot_container:
        st.subheader("Interactive CV plot")
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
                data=plot_html,
                file_name="voltscope_cv_plot.html",
                mime="text/html",
            )

    with esw_expander:
        threshold_display = st.number_input(
            f"ESW threshold ({current_display_unit})",
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
        st.dataframe(esw_df, width="stretch")

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

            st.table(
                per_cycle_metrics_df.style.apply(highlight_selected_cycle, axis=1),
            )
            if per_cycle_trend_messages:
                st.markdown("**Trend summary**")
                for trend_message in per_cycle_trend_messages:
                    st.caption(trend_message)

        st.subheader("Detailed CV metrics")
        if peak_metric_rows:
            peak_metric_df = pd.DataFrame(peak_metric_rows)
            compact_metrics_df = metrics_table(metrics, current_display_unit, electrode_area_cm2, current_source)
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
            )
            if current_unit_review_needed and not selected_peak_display_df.empty:
                current_columns = [
                    column
                    for column in selected_peak_display_df.columns
                    if "current_" in str(column)
                ]
                for column in current_columns:
                    selected_peak_display_df[column] = "Pending unit confirmation"
            st.dataframe(selected_peak_display_df, width="stretch", hide_index=True)
            st.table(compact_metrics_df)
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
        data=export_buffer.getvalue(),
        file_name="voltscope_cv_analysis_results.csv",
        mime="text/csv",
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


if __name__ == "__main__":
    main()
