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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


POTENTIAL_UNITS_TO_V = {"V": 1.0, "mV": 1e-3}
CURRENT_UNITS_TO_A = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9}
TIME_UNITS_TO_S = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}
FREQUENCY_UNITS_TO_HZ = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}
IMPEDANCE_UNITS_TO_OHM = {"ohm": 1.0, "kohm": 1e3, "Mohm": 1e6}
PHASE_UNITS_TO_DEG = {"deg": 1.0, "rad": 180.0 / np.pi}
WORKSPACE_STORE_PATH = Path(__file__).with_name("voltscope_project_workspaces.json")


EXPERIMENT_TYPES = {
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
EXPERIMENT_TYPE_LABELS = [config["label"] for config in EXPERIMENT_TYPES.values()]
EXPERIMENT_TYPE_BY_LABEL = {config["label"]: key for key, config in EXPERIMENT_TYPES.items()}


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
    confidence: str


@dataclass
class AnalyzedDataset:
    filename: str
    sample_name: str
    potential_col: str
    current_col: str
    potential_unit: str
    current_unit: str
    potential_v: np.ndarray
    raw_current_a: np.ndarray
    smoothed_current_a: Optional[np.ndarray]
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


def detect_delimiter(lines: list[str]) -> str:
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


def detect_unit_from_header(header: str, role: str) -> Optional[str]:
    label = normalize_label(header)

    if role == "potential":
        if "mv" in label:
            return "mV"
        if "/v" in label or "(v)" in label or label in {"v", "e", "voltage"}:
            return "V"
        return None

    if role == "current":
        if "na" in label:
            return "nA"
        if "ua" in label:
            return "uA"
        if "ma" in label:
            return "mA"
        if "/a" in label or "(a)" in label or label in {"a", "i", "current"}:
            return "A"
        return None

    if role == "time":
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


def score_potential_column(header: str, values: np.ndarray) -> float:
    label = normalize_label(header)
    finite = values[np.isfinite(values)]
    score = 0.0

    if any(keyword in label for keyword in ["potential", "voltage", "ewe", "e/v", "potential/v"]):
        score += 7
    if label in {"e", "v", "x", "potentialv", "ewe/v"}:
        score += 4
    if "time" in label or "current" in label or label in {"i", "a"}:
        score -= 5
    if detect_unit_from_header(header, "potential"):
        score += 2

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

    if any(keyword in label for keyword in ["current", "i/a", "current/a", "amp", "amps"]):
        score += 7
    if label in {"i", "y", "a", "ma", "ua", "na"}:
        score += 4
    if "potential" in label or "voltage" in label or label in {"e", "v"}:
        score -= 5
    if detect_unit_from_header(header, "current"):
        score += 2

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
    for line in lines[data_start_idx:]:
        tokens = split_line(line, delimiter)
        if len(tokens) < 2:
            continue

        if len(tokens) > len(headers):
            extra_headers = [f"Column {idx + 1}" for idx in range(len(headers), len(tokens))]
            headers = deduplicate_headers(headers + extra_headers)
        elif len(tokens) < len(headers):
            tokens = tokens + [""] * (len(headers) - len(tokens))

        values = [parse_number(token) for token in tokens[: len(headers)]]
        if np.isfinite(values).sum() >= 2:
            rows.append(values)

    if not rows:
        raise ValueError("Could not parse numeric rows from the data table.")

    df = pd.DataFrame(rows, columns=headers)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    headers = list(df.columns)

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


def convert_potential_to_volts(values: np.ndarray, unit: str) -> np.ndarray:
    return values * POTENTIAL_UNITS_TO_V.get(unit, 1.0)


def convert_current_to_amps(values: np.ndarray, unit: str) -> np.ndarray:
    return values * CURRENT_UNITS_TO_A.get(unit, 1.0)


CURRENT_DENSITY_UNITS_TO_A_PER_CM2 = {
    "A/cm^2": 1.0,
    "mA/cm^2": 1e-3,
    "uA/cm^2": 1e-6,
    "nA/cm^2": 1e-9,
}
CURRENT_DENSITY_UNIT = "mA/cm^2"
REFERENCE_ELECTRODE_OPTIONS = [
    "Ag/AgCl",
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


def current_from_amps(
    values: np.ndarray,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
) -> np.ndarray:
    if is_current_density_unit(display_unit):
        return current_density_from_base(values, display_unit, "Current", electrode_area_cm2)
    return values / CURRENT_UNITS_TO_A.get(display_unit, 1.0)


def current_value_to_amps(
    value: float,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
) -> float:
    if is_current_density_unit(display_unit):
        return current_density_value_to_base(value, display_unit, "Current", electrode_area_cm2)
    return value * CURRENT_UNITS_TO_A.get(display_unit, 1.0)


def current_axis_label(display_unit: str) -> str:
    if is_current_density_unit(display_unit):
        return f"Current density ({display_unit})"
    return f"Current ({display_unit})"


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
        return ["Auto", "A", "mA", "uA", "nA"]
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
        return ["uA", "mA", "A", "nA"]
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
        warnings.append(f"{filename}: too few numeric rows for reliable CV analysis.")
        return warnings

    potential_range = float(np.nanmax(potential_v[finite_mask]) - np.nanmin(potential_v[finite_mask]))
    current_range = float(np.nanmax(current_a[finite_mask]) - np.nanmin(current_a[finite_mask]))
    if potential_range == 0:
        warnings.append(f"{filename}: potential values are constant.")
    if current_range == 0:
        warnings.append(f"{filename}: current values are constant.")

    turning_points = detect_potential_turning_points(potential_v[finite_mask])
    if len(turning_points) > 1:
        warnings.append(
            f"{filename}: multiple potential turning points detected; cycle splitting is approximate."
        )

    return warnings


def normalize_dataset(
    dataset: ParsedDataset,
    potential_col: str,
    current_col: str,
    selected_potential_unit: str,
    selected_current_unit: str,
    smoothing_method: str,
    smoothing_window: int,
) -> AnalyzedDataset:
    working = dataset.dataframe[[potential_col, current_col]].dropna().copy()
    potential_values = working[potential_col].to_numpy(dtype=float)
    current_values = working[current_col].to_numpy(dtype=float)

    potential_unit = resolve_unit(selected_potential_unit, dataset, "potential", potential_col, "V")
    current_unit = resolve_unit(selected_current_unit, dataset, "current", current_col, "A")

    potential_v = convert_potential_to_volts(potential_values, potential_unit)
    raw_current_a = convert_current_to_amps(current_values, current_unit)

    finite_mask = np.isfinite(potential_v) & np.isfinite(raw_current_a)
    potential_v = potential_v[finite_mask]
    raw_current_a = raw_current_a[finite_mask]

    smoothed_current_a = None
    if smoothing_method != "None" and len(raw_current_a) >= 5:
        smoothed_current_a = smooth_current(raw_current_a, smoothing_method, smoothing_window)

    warnings = validate_cv_data(dataset.filename, potential_v, raw_current_a)

    return AnalyzedDataset(
        filename=dataset.filename,
        sample_name=dataset.filename.rsplit(".", 1)[0],
        potential_col=potential_col,
        current_col=current_col,
        potential_unit=potential_unit,
        current_unit=current_unit,
        potential_v=potential_v,
        raw_current_a=raw_current_a,
        smoothed_current_a=smoothed_current_a,
        metadata=dataset.metadata,
        warnings=dataset.warnings + warnings,
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

    for segment_index, (start, end, _direction) in enumerate(segments, start=1):
        for idx in range(start + 1, end - 1):
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
                            confidence=confidence_from_prominence(prominence, min_prominence),
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
                            confidence=confidence_from_prominence(prominence, min_prominence),
                        )
                    )

    filtered = enforce_min_distance(candidate_peaks, min_distance)
    type_counts = {"oxidation": 0, "reduction": 0}
    for peak in filtered:
        type_counts[peak.peak_type] += 1
        prefix = "ox" if peak.peak_type == "oxidation" else "red"
        peak.id = f"{prefix}_{type_counts[peak.peak_type]}"
    return filtered


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


def peak_label(peak: Peak, display_unit: str, electrode_area_cm2: float = 1.0) -> str:
    current_display = current_from_amps(np.array([peak.raw_current]), display_unit, electrode_area_cm2)[0]
    prominence_display = current_from_amps(np.array([peak.prominence]), display_unit, electrode_area_cm2)[0]
    return (
        f"{peak.id}: {peak.peak_type}, E={peak.potential:.4g} V, "
        f"I={current_display:.4g} {display_unit}, prom={prominence_display:.4g} {display_unit}"
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
        confidence="manual",
    )


def build_peak_metrics_rows(
    oxidation_peak: Optional[Peak],
    reduction_peak: Optional[Peak],
    baseline_current: Optional[np.ndarray],
    analysis_current: np.ndarray,
    display_unit: str,
    electrode_area_cm2: float = 1.0,
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
        baseline_at_peak = 0.0 if baseline_current is None else float(baseline_current[peak.index])
        raw_current = float(analysis_current[peak.index])
        corrected_current = raw_current - baseline_at_peak
        raw_display = current_from_amps(np.array([raw_current]), display_unit, electrode_area_cm2)[0]
        baseline_display = current_from_amps(np.array([baseline_at_peak]), display_unit, electrode_area_cm2)[0]
        corrected_display = current_from_amps(np.array([corrected_current]), display_unit, electrode_area_cm2)[0]

        rows.append(
            {
                "peak": metric_name,
                "potential_V": peak.potential,
                f"raw_current_{display_unit}": raw_display,
                f"baseline_current_{display_unit}": baseline_display,
                f"corrected_current_{display_unit}": corrected_display,
                "confidence": peak.confidence,
            }
        )

        if metric_name == "oxidation":
            metrics["epa_V"] = peak.potential
            metrics["ipa_A"] = corrected_current
        else:
            metrics["epc_V"] = peak.potential
            metrics["ipc_A"] = corrected_current

    if metrics["epa_V"] is not None and metrics["epc_V"] is not None:
        metrics["delta_ep_V"] = abs(metrics["epa_V"] - metrics["epc_V"])
    if metrics["ipa_A"] is not None and metrics["ipc_A"] not in {None, 0}:
        metrics["ipa_ipc_ratio"] = abs(metrics["ipa_A"] / metrics["ipc_A"])

    return rows, metrics


def metrics_table(
    metrics: dict[str, Optional[float]],
    display_unit: str,
    electrode_area_cm2: float = 1.0,
) -> pd.DataFrame:
    rows = []
    if metrics["epa_V"] is not None:
        rows.append({"metric": "Epa", "value": f"{metrics['epa_V']:.4g} V"})
    if metrics["ipa_A"] is not None:
        ipa_display = current_from_amps(np.array([metrics["ipa_A"]]), display_unit, electrode_area_cm2)[0]
        rows.append({"metric": "ipa", "value": f"{ipa_display:.4g} {display_unit}"})
    if metrics["epc_V"] is not None:
        rows.append({"metric": "Epc", "value": f"{metrics['epc_V']:.4g} V"})
    if metrics["ipc_A"] is not None:
        ipc_display = current_from_amps(np.array([metrics["ipc_A"]]), display_unit, electrode_area_cm2)[0]
        rows.append({"metric": "ipc", "value": f"{ipc_display:.4g} {display_unit}"})
    if metrics["delta_ep_V"] is not None:
        rows.append({"metric": "Delta Ep", "value": f"{metrics['delta_ep_V']:.4g} V"})
    if metrics["ipa_ipc_ratio"] is not None:
        rows.append({"metric": "ipa/ipc", "value": f"{metrics['ipa_ipc_ratio']:.4g}"})
    return pd.DataFrame(rows)



# ---------- Multi-experiment analysis helpers ----------


def experiment_label(experiment_type: str) -> str:
    return EXPERIMENT_TYPES.get(experiment_type, EXPERIMENT_TYPES["cv"])["label"]


def experiment_short_label(experiment_type: str) -> str:
    return EXPERIMENT_TYPES.get(experiment_type, EXPERIMENT_TYPES["cv"])["short_label"]


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


def make_parser_summary(parsed_datasets: list[ParsedDataset], experiment_type: str) -> pd.DataFrame:
    rows = []
    roles = required_roles_for_experiment(experiment_type) + optional_roles_for_experiment(experiment_type)
    for dataset in parsed_datasets:
        detected = detect_experiment_columns(dataset, experiment_type)
        row = {
            "file": dataset.filename,
            "rows": len(dataset.dataframe),
            "columns": len(dataset.headers),
            "delimiter": dataset.delimiter,
            "metadata_fields": len(dataset.metadata),
            "warnings": "; ".join(dataset.warnings),
        }
        for role in roles:
            row[f"detected_{role}"] = detected.get(role)
        rows.append(row)
    return pd.DataFrame(rows)


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
        unit = resolve_role_unit(selected_units.get(role, "Auto"), dataset, role, column)
        values = working[column].to_numpy(dtype=float)
        normalized_data[role] = convert_role_to_base(values, role, unit)
        columns[role] = column
        units[role] = unit

    mask = finite_pair_mask(*normalized_data.values())
    normalized_data = {role: values[mask] for role, values in normalized_data.items()}
    if len(next(iter(normalized_data.values()))) < 5:
        warnings.append("Too few numeric rows for reliable analysis.")

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
                row.update(
                    {
                        f"threshold_{lsv_threshold_unit or 'display'}": lsv_threshold_display,
                        "cathodic_limit_V": result.cathodic_limit,
                        "anodic_limit_V": result.anodic_limit,
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


def render_generic_experiment_analysis(
    project_name: str,
    project_workspace: dict[str, Any],
    experiment_id: str,
    experiment_type: str,
    parsed_datasets: list[ParsedDataset],
) -> None:
    parser_summary = make_parser_summary(parsed_datasets, experiment_type)
    st.subheader("Parser summary")
    st.dataframe(parser_summary, width="stretch", hide_index=True)

    selected_filename = st.selectbox(
        "Preview / detailed analysis file",
        [dataset.filename for dataset in parsed_datasets],
        key=f"generic_preview::{project_name}::{experiment_id}",
    )
    selected_dataset = next(dataset for dataset in parsed_datasets if dataset.filename == selected_filename)
    selected_detected = detect_experiment_columns(selected_dataset, experiment_type)

    with st.expander("Raw parsed data preview", expanded=True):
        st.dataframe(dataframe_preview_for_display(selected_dataset.dataframe).head(30), width="stretch")

    with st.expander("Detected metadata"):
        if selected_dataset.metadata:
            st.json(selected_dataset.metadata)
        else:
            st.write("No metadata rows detected before the numeric table.")
            st.text_area(
                "Manual metadata / notes",
                placeholder="Type any sample details, instrument settings, electrolyte notes, or context for this file.",
                key=f"manual_metadata::{project_name}::{experiment_id}::{selected_dataset.filename}",
            )

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

    st.subheader("Display controls")
    display_units: dict[str, str] = {}
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
        potential_display_unit = resolve_role_unit(
            selected_units.get("potential", "Auto"),
            selected_dataset,
            "potential",
            potential_col_for_units,
        )
        if potential_display_unit not in display_unit_options_for_role("potential"):
            potential_display_unit = "V"
        current_unit_for_density = resolve_role_unit(
            selected_units.get("current", "Auto"),
            selected_dataset,
            "current",
            current_col_for_units,
        )

        axis_options = ["Potential", "Current density"]
        display_units["current_source"] = st.selectbox(
            "Uploaded current column contains",
            ["Current", "Current density"],
            key=f"lsv_current_source::{project_name}::{experiment_id}",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            display_units["x_axis"] = st.selectbox(
                "X-axis",
                axis_options,
                index=0,
                key=f"lsv_x_axis::{project_name}::{experiment_id}",
            )
        with col2:
            display_units["y_axis"] = st.selectbox(
                "Y-axis",
                axis_options,
                index=1,
                key=f"lsv_y_axis::{project_name}::{experiment_id}",
            )
        with col3:
            show_markers = st.checkbox("Show data markers", value=False, key=f"generic_markers::{project_name}::{experiment_id}")

        if display_units["x_axis"] == display_units["y_axis"]:
            st.warning("Choose different quantities for the X-axis and Y-axis.")

        display_units["potential"] = potential_display_unit
        display_units["current"] = current_density_unit_from_current_unit(current_unit_for_density)
        if display_units["current_source"] == "Current density":
            display_units["electrode_area_cm2"] = 1.0
            st.info("The uploaded current column is already normalized, so electrode-area normalization is skipped.")
        else:
            display_units["electrode_area_cm2"] = st.number_input(
                "Electrode surface area (cm^2)",
                min_value=0.000001,
                value=1.0,
                step=0.1,
                format="%.6f",
                key=f"lsv_electrode_area::{project_name}::{experiment_id}",
            )
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
        records.append(record)

    if not records:
        st.error("No uploaded files could be analyzed with the selected experiment type and column settings.")
        st.stop()
        return

    for record in records:
        for warning in record.get("warnings", []):
            st.warning(f"{record['filename']}: {warning}")

    lsv_threshold_display = None
    lsv_threshold_a = None
    lsv_threshold_unit = None
    reference_electrode = None
    if experiment_type == "lsv":
        y_unit = display_units["current"]
        electrode_area_cm2 = float(display_units.get("electrode_area_cm2", 1.0))
        current_source = display_units.get("current_source", "Current")
        all_lsv_y = np.concatenate(
            [
                np.abs(current_density_from_base(record["data"]["current"], y_unit, current_source, electrode_area_cm2))
                for record in records
            ]
        )
        finite_lsv_y = all_lsv_y[np.isfinite(all_lsv_y)]
        default_threshold = float(np.nanmax(finite_lsv_y) * 0.1) if finite_lsv_y.size else 0.5
        if default_threshold <= 0:
            default_threshold = 0.5
        lsv_threshold_unit = y_unit
        lsv_threshold_display = st.number_input(
            f"Current density threshold ({y_unit})",
            min_value=0.0,
            value=default_threshold,
            step=max(default_threshold * 0.1, 1e-9),
            format="%.6e",
            key=f"lsv_threshold::{project_name}::{experiment_id}",
        )
        lsv_threshold_a = current_density_value_to_base(lsv_threshold_display, y_unit, current_source, electrode_area_cm2)
        reference_electrode = st.selectbox(
            "Reference electrode",
            REFERENCE_ELECTRODE_OPTIONS,
            key=f"lsv_reference_electrode::{project_name}::{experiment_id}",
        )
        if reference_electrode == "Other":
            custom_reference = st.text_input(
                "Reference electrode name",
                key=f"lsv_reference_electrode_custom::{project_name}::{experiment_id}",
            ).strip()
            reference_electrode = custom_reference or "Other"

    st.subheader(f"Interactive {experiment_short_label(experiment_type)} plot")
    fig = go.Figure()
    for record in records:
        add_generic_trace(fig, record, experiment_type, display_units, show_markers)

    if experiment_type == "lsv" and lsv_threshold_display is not None and lsv_threshold_display > 0:
        if display_units.get("y_axis") == "Current density":
            fig.add_hline(
                y=lsv_threshold_display,
                line_dash="dot",
                annotation_text=f"+{lsv_threshold_display:g} {lsv_threshold_unit}",
            )
            fig.add_hline(
                y=-lsv_threshold_display,
                line_dash="dot",
                annotation_text=f"-{lsv_threshold_display:g} {lsv_threshold_unit}",
            )
        elif display_units.get("x_axis") == "Current density":
            fig.add_vline(
                x=lsv_threshold_display,
                line_dash="dot",
                annotation_text=f"+{lsv_threshold_display:g} {lsv_threshold_unit}",
            )
            fig.add_vline(
                x=-lsv_threshold_display,
                line_dash="dot",
                annotation_text=f"-{lsv_threshold_display:g} {lsv_threshold_unit}",
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

    st.subheader("Analysis results")
    results_df = build_generic_results(
        records,
        experiment_type,
        lsv_threshold_a=lsv_threshold_a,
        lsv_threshold_display=lsv_threshold_display,
        lsv_threshold_unit=lsv_threshold_unit,
        lsv_current_source=display_units.get("current_source", "Current"),
    )
    st.dataframe(results_df, width="stretch", hide_index=True)

    export_buffer = io.StringIO()
    export_buffer.write("# Parser summary\n")
    export_buffer.write(parser_summary.to_csv(index=False))
    export_buffer.write("\n# Analysis results\n")
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
            "lsv_current_source": display_units.get("current_source", "Current"),
            "reference_electrode": reference_electrode,
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
            experiment.setdefault("experiment_type", "cv")
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
    experiment_type: str = "cv",
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
    return {
        "filename": dataset.filename,
        "sample_name": dataset.sample_name,
        "potential_col": dataset.potential_col,
        "current_col": dataset.current_col,
        "potential_unit": dataset.potential_unit,
        "current_unit": dataset.current_unit,
        "points": int(len(dataset.potential_v)),
        "potential_min_v": float(np.min(potential)) if potential.size else None,
        "potential_max_v": float(np.max(potential)) if potential.size else None,
        "current_min_a": float(np.min(current)) if current.size else None,
        "current_max_a": float(np.max(current)) if current.size else None,
        "warnings": dataset.warnings,
    }


def serialize_analyzed_dataset(dataset: AnalyzedDataset) -> dict[str, Any]:
    return {
        "summary": summarize_analysis(dataset),
        "metadata": dataset.metadata,
        "warnings": dataset.warnings,
        "trace": {
            "potential_v": dataset.potential_v.tolist(),
            "raw_current_a": dataset.raw_current_a.tolist(),
            "smoothed_current_a": (
                dataset.smoothed_current_a.tolist() if dataset.smoothed_current_a is not None else None
            ),
        },
    }


def store_experiment_analysis(
    project_workspace: dict[str, Any],
    experiment_id: str,
    analyzed: list[AnalyzedDataset],
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
    for dataset in analyzed:
        files[dataset.filename] = serialize_analyzed_dataset(dataset)

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

                action_cols = st.columns([0.68, 0.32])
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
                        "Open experiment",
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
        }
        .analysis-card .analysis-file {
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 0.8rem;
            overflow-wrap: anywhere;
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
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .analysis-metric strong {
            display: block;
            font-size: 1rem;
            margin-top: 0.15rem;
            overflow-wrap: anywhere;
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
        "<div class='workspace-card'><h3>Create experiment</h3><p>Choose a data type first, then upload files into that experiment.</p></div>",
        unsafe_allow_html=True,
    )
    default_name = f"Experiment {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    with st.form(key=f"{key_prefix}_create_experiment_form::{project_name}"):
        experiment_name = st.text_input(
            "Experiment name",
            value=default_name,
            key=f"{key_prefix}_new_experiment::{project_name}",
        )
        experiment_type_label = st.selectbox(
            "Data type",
            EXPERIMENT_TYPE_LABELS,
            key=f"{key_prefix}_new_experiment_type::{project_name}",
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
            EXPERIMENT_TYPE_BY_LABEL[experiment_type_label],
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
            display_unit = st.selectbox(
                "Saved trace display unit",
                ["uA", "mA", "A", "nA"],
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
                            y=current_from_amps(np.array(current, dtype=float), display_unit),
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
                fig.update_layout(xaxis_title="Potential / V", yaxis_title=f"Current / {display_unit}")
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
    experiment_type = experiment.setdefault("experiment_type", "cv")
    if experiment_type not in EXPERIMENT_TYPES:
        experiment_type = "cv"
        experiment["experiment_type"] = experiment_type
    st.markdown(
        f"""
        <div class="experiment-card">
            <h3>{escape_html(experiment.get("name", experiment_id))}</h3>
            <p>{escape_html(experiment_short_label(experiment_type))} · created {format_timestamp(experiment.get("created_at"))} · updated {format_timestamp(experiment.get("updated_at"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Data files",
        type=["csv", "txt", "tsv", "dat"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"upload::{active_project_name}::{experiment_id}::{experiment_type}",
    )

    if not uploaded_files:
        if experiment.get("files"):
            render_saved_experiment(active_project_name, project_workspace, experiment_id, show_header=False)
        return

    parsed_datasets: list[ParsedDataset] = []
    for uploaded_file in uploaded_files:
        try:
            parsed_datasets.append(parse_electrochem_file(uploaded_file))
        except Exception as exc:
            st.error(f"Could not parse {uploaded_file.name}: {exc}")

    if not parsed_datasets:
        st.stop()
        return

    if experiment_type != "cv":
        render_generic_experiment_analysis(
            active_project_name,
            project_workspace,
            experiment_id,
            experiment_type,
            parsed_datasets,
        )
        return

    st.subheader("Parser summary")
    parser_summary = pd.DataFrame(
        [
            {
                "file": dataset.filename,
                "rows": len(dataset.dataframe),
                "columns": len(dataset.headers),
                "delimiter": dataset.delimiter,
                "detected_potential": dataset.detected_potential_col,
                "detected_current": dataset.detected_current_col,
                "metadata_fields": len(dataset.metadata),
                "warnings": "; ".join(dataset.warnings),
            }
            for dataset in parsed_datasets
        ]
    )
    st.dataframe(parser_summary, width="stretch")

    selected_filename = st.selectbox(
        "Preview / detailed analysis file",
        [dataset.filename for dataset in parsed_datasets],
    )
    selected_dataset = next(dataset for dataset in parsed_datasets if dataset.filename == selected_filename)

    with st.expander("Raw parsed data preview", expanded=True):
        st.dataframe(dataframe_preview_for_display(selected_dataset.dataframe).head(30), width="stretch")

    with st.expander("Detected metadata"):
        if selected_dataset.metadata:
            st.json(selected_dataset.metadata)
        else:
            st.write("No metadata rows detected before the numeric table.")
            st.text_area(
                "Manual metadata / notes",
                placeholder="Type any sample details, instrument settings, electrolyte notes, or context for this file.",
                key=f"manual_metadata::{active_project_name}::{experiment_id}::{selected_dataset.filename}",
            )

    st.subheader("Column and unit controls")
    column_mode = st.radio(
        "Column selection mode",
        ["Use detected columns per file", "Manually select columns"],
        horizontal=True,
    )

    manual_potential_col = selected_dataset.detected_potential_col or selected_dataset.headers[0]
    manual_current_col = selected_dataset.detected_current_col or selected_dataset.headers[min(1, len(selected_dataset.headers) - 1)]

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
        selected_current_unit = st.selectbox("Current input unit", ["Auto", "A", "mA", "uA", "nA"])
    with unit_col3:
        y_axis_quantity = st.selectbox("Y-axis quantity", ["Current", "Current density"])

    electrode_area_cm2 = 1.0
    if y_axis_quantity == "Current density":
        current_display_unit = CURRENT_DENSITY_UNIT
        electrode_area_cm2 = st.number_input(
            "Electrode surface area (cm^2)",
            min_value=0.000001,
            value=1.0,
            step=0.1,
            format="%.6f",
            key=f"cv_electrode_area::{active_project_name}::{experiment_id}",
        )
    else:
        current_display_unit = st.selectbox("Current display unit", ["uA", "mA", "A", "nA"])

    st.subheader("Processing controls")
    proc_col1, proc_col2, proc_col3 = st.columns(3)
    with proc_col1:
        smoothing_method = st.selectbox("Smoothing", ["None", "Moving average", "Savitzky-Golay"])
    with proc_col2:
        smoothing_window = st.slider("Smoothing window", min_value=3, max_value=51, value=11, step=2)
    with proc_col3:
        use_smoothed_for_peaks = st.checkbox("Use smoothed data for peak detection", value=True)

    plot_col1, plot_col2, plot_col3 = st.columns(3)
    with plot_col1:
        reverse_x_axis = st.checkbox("Reverse x-axis", value=False)
    with plot_col2:
        show_markers = st.checkbox("Show data markers", value=False)
    with plot_col3:
        show_raw_with_smoothed = st.checkbox("Show raw trace when smoothed", value=True)

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
    analysis_current_a = (
        detailed_dataset.smoothed_current_a
        if use_smoothed_for_peaks and detailed_dataset.smoothed_current_a is not None
        else detailed_dataset.raw_current_a
    )

    st.subheader("Peak detection")
    current_display_values = current_from_amps(
        analysis_current_a,
        current_display_unit,
        electrode_area_cm2,
    )
    current_range = float(np.nanmax(current_display_values) - np.nanmin(current_display_values))
    default_prominence = max(
        current_range * 0.05,
        current_from_amps(np.array([1e-12]), current_display_unit, electrode_area_cm2)[0],
    )

    peak_col1, peak_col2, peak_col3 = st.columns(3)
    with peak_col1:
        min_prominence_display = st.number_input(
            f"Minimum peak prominence ({current_display_unit})",
            min_value=0.0,
            value=float(default_prominence),
            step=float(default_prominence / 2 if default_prominence > 0 else 1e-6),
            format="%.6e",
        )
    with peak_col2:
        min_distance = st.number_input("Minimum peak distance (points)", min_value=1, value=10, step=1)
    with peak_col3:
        min_abs_current_display = st.number_input(
            f"Minimum absolute current ({current_display_unit})",
            min_value=0.0,
            value=0.0,
            step=float(default_prominence / 2 if default_prominence > 0 else 1e-6),
            format="%.6e",
        )

    peaks = detect_peaks(
        detailed_dataset.potential_v,
        analysis_current_a,
        current_value_to_amps(min_prominence_display, current_display_unit, electrode_area_cm2),
        int(min_distance),
        current_value_to_amps(min_abs_current_display, current_display_unit, electrode_area_cm2),
    )

    peak_rows = [
        {
            "id": peak.id,
            "type": peak.peak_type,
            "index": peak.index,
            "potential_V": peak.potential,
            f"current_{current_display_unit}": current_from_amps(
                np.array([peak.raw_current]), current_display_unit, electrode_area_cm2
            )[0],
            f"prominence_{current_display_unit}": current_from_amps(
                np.array([peak.prominence]), current_display_unit, electrode_area_cm2
            )[0],
            "segment": peak.segment_index,
            "confidence": peak.confidence,
        }
        for peak in peaks
    ]
    st.dataframe(pd.DataFrame(peak_rows), width="stretch")

    oxidation_peaks = [peak for peak in peaks if peak.peak_type == "oxidation"]
    reduction_peaks = [peak for peak in peaks if peak.peak_type == "reduction"]

    manual_peak_override = st.checkbox("Manually select peak points by index", value=False)
    selected_oxidation_peak: Optional[Peak] = None
    selected_reduction_peak: Optional[Peak] = None

    if manual_peak_override:
        manual_col1, manual_col2 = st.columns(2)
        with manual_col1:
            oxidation_idx = st.slider(
                "Oxidation peak index",
                min_value=0,
                max_value=len(detailed_dataset.potential_v) - 1,
                value=int(np.nanargmax(analysis_current_a)),
            )
            selected_oxidation_peak = make_manual_peak(
                "oxidation", oxidation_idx, detailed_dataset.potential_v, analysis_current_a
            )
        with manual_col2:
            reduction_idx = st.slider(
                "Reduction peak index",
                min_value=0,
                max_value=len(detailed_dataset.potential_v) - 1,
                value=int(np.nanargmin(analysis_current_a)),
            )
            selected_reduction_peak = make_manual_peak(
                "reduction", reduction_idx, detailed_dataset.potential_v, analysis_current_a
            )
    else:
        select_col1, select_col2 = st.columns(2)
        with select_col1:
            if oxidation_peaks:
                labels = [peak_label(peak, current_display_unit, electrode_area_cm2) for peak in oxidation_peaks]
                selected_label = st.selectbox("Oxidation peak", labels)
                selected_oxidation_peak = oxidation_peaks[labels.index(selected_label)]
            else:
                st.info("No oxidation peaks detected. Lower the prominence threshold or use manual selection.")
        with select_col2:
            if reduction_peaks:
                labels = [peak_label(peak, current_display_unit, electrode_area_cm2) for peak in reduction_peaks]
                selected_label = st.selectbox("Reduction peak", labels)
                selected_reduction_peak = reduction_peaks[labels.index(selected_label)]
            else:
                st.info("No reduction peaks detected. Lower the prominence threshold or use manual selection.")

    st.subheader("Baseline correction")
    baseline_method = st.radio("Baseline method", ["None", "Linear"], horizontal=True)
    baseline_current: Optional[np.ndarray] = None
    anchor_a_idx = 0
    anchor_b_idx = len(detailed_dataset.potential_v) - 1

    if baseline_method == "Linear":
        reference_peak = selected_oxidation_peak or selected_reduction_peak
        reference_idx = reference_peak.index if reference_peak else len(detailed_dataset.potential_v) // 2
        default_span = max(5, len(detailed_dataset.potential_v) // 10)
        default_a = max(0, reference_idx - default_span)
        default_b = min(len(detailed_dataset.potential_v) - 1, reference_idx + default_span)

        anchor_col1, anchor_col2 = st.columns(2)
        with anchor_col1:
            anchor_a_idx = st.slider(
                "Baseline anchor A index",
                min_value=0,
                max_value=len(detailed_dataset.potential_v) - 1,
                value=default_a,
            )
        with anchor_col2:
            anchor_b_idx = st.slider(
                "Baseline anchor B index",
                min_value=0,
                max_value=len(detailed_dataset.potential_v) - 1,
                value=default_b,
            )

        baseline_current = calculate_linear_baseline(
            detailed_dataset.potential_v,
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
                    "potential_V": detailed_dataset.potential_v[anchor_a_idx],
                    f"current_{current_display_unit}": current_from_amps(
                        np.array([analysis_current_a[anchor_a_idx]]), current_display_unit, electrode_area_cm2
                    )[0],
                },
                {
                    "anchor": "B",
                    "index": anchor_b_idx,
                    "potential_V": detailed_dataset.potential_v[anchor_b_idx],
                    f"current_{current_display_unit}": current_from_amps(
                        np.array([analysis_current_a[anchor_b_idx]]), current_display_unit, electrode_area_cm2
                    )[0],
                },
            ]
        )
        st.dataframe(anchor_summary, width="stretch")

    fig = go.Figure()
    trace_mode = "lines+markers" if show_markers else "lines"

    for dataset in analyzed:
        raw_display = current_from_amps(dataset.raw_current_a, current_display_unit, electrode_area_cm2)
        if dataset.smoothed_current_a is not None and show_raw_with_smoothed:
            fig.add_trace(
                go.Scatter(
                    x=dataset.potential_v,
                    y=raw_display,
                    mode="lines",
                    name=f"{dataset.sample_name} raw",
                    opacity=0.35,
                    hovertemplate=(
                        f"Potential: %{{x:.4g}} V<br>{current_hover_label(current_display_unit)}: %{{y:.4g}} "
                        + current_display_unit
                        + "<extra></extra>"
                    ),
                )
            )
        elif dataset.smoothed_current_a is None:
            fig.add_trace(
                go.Scatter(
                    x=dataset.potential_v,
                    y=raw_display,
                    mode=trace_mode,
                    name=dataset.sample_name,
                    hovertemplate=(
                        f"Potential: %{{x:.4g}} V<br>{current_hover_label(current_display_unit)}: %{{y:.4g}} "
                        + current_display_unit
                        + "<extra></extra>"
                    ),
                )
            )

        if dataset.smoothed_current_a is not None:
            smoothed_display = current_from_amps(dataset.smoothed_current_a, current_display_unit, electrode_area_cm2)
            fig.add_trace(
                go.Scatter(
                    x=dataset.potential_v,
                    y=smoothed_display,
                    mode=trace_mode,
                    name=f"{dataset.sample_name} smoothed",
                    hovertemplate=(
                        f"Potential: %{{x:.4g}} V<br>{current_hover_label(current_display_unit)}: %{{y:.4g}} "
                        + current_display_unit
                        + "<extra></extra>"
                    ),
                )
            )

    if peaks:
        oxidation_marker_peaks = [peak for peak in peaks if peak.peak_type == "oxidation"]
        reduction_marker_peaks = [peak for peak in peaks if peak.peak_type == "reduction"]
        for marker_peaks, symbol, name in [
            (oxidation_marker_peaks, "triangle-up", "Detected oxidation peaks"),
            (reduction_marker_peaks, "triangle-down", "Detected reduction peaks"),
        ]:
            if not marker_peaks:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[peak.potential for peak in marker_peaks],
                    y=current_from_amps(
                        np.array([analysis_current_a[peak.index] for peak in marker_peaks]),
                        current_display_unit,
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
                y=current_from_amps(
                    np.array([analysis_current_a[peak.index] for peak in selected_peaks]),
                    current_display_unit,
                    electrode_area_cm2,
                ),
                mode="markers",
                name="Selected peaks",
                marker={"size": 14, "symbol": "star", "line": {"width": 1}},
            )
        )

    if baseline_current is not None:
        fig.add_trace(
            go.Scatter(
                x=detailed_dataset.potential_v,
                y=current_from_amps(baseline_current, current_display_unit, electrode_area_cm2),
                mode="lines",
                name="Linear baseline",
                line={"dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[detailed_dataset.potential_v[anchor_a_idx], detailed_dataset.potential_v[anchor_b_idx]],
                y=current_from_amps(
                    np.array([analysis_current_a[anchor_a_idx], analysis_current_a[anchor_b_idx]]),
                    current_display_unit,
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

    st.subheader("Interactive CV plot")
    st.plotly_chart(fig, width="stretch")

    plot_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    st.download_button(
        "Download interactive plot as HTML",
        data=plot_html,
        file_name="voltscope_cv_plot.html",
        mime="text/html",
    )

    st.subheader("Operational ESW summary")
    threshold_display = st.number_input(
        f"ESW threshold ({current_display_unit})",
        min_value=0.0,
        value=float(max(
            current_range * 0.1,
            current_from_amps(np.array([1e-12]), current_display_unit, electrode_area_cm2)[0],
        )),
        step=float(max(
            current_range * 0.05,
            current_from_amps(np.array([1e-12]), current_display_unit, electrode_area_cm2)[0],
        )),
        format="%.6e",
    )
    threshold_a = current_value_to_amps(threshold_display, current_display_unit, electrode_area_cm2)
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

    st.subheader("Peak metrics")
    peak_metric_rows, metrics = build_peak_metrics_rows(
        selected_oxidation_peak,
        selected_reduction_peak,
        baseline_current,
        analysis_current_a,
        current_display_unit,
        electrode_area_cm2,
    )
    if peak_metric_rows:
        peak_metric_df = pd.DataFrame(peak_metric_rows)
        compact_metrics_df = metrics_table(metrics, current_display_unit, electrode_area_cm2)
        st.dataframe(peak_metric_df, width="stretch")
        st.dataframe(compact_metrics_df, width="stretch")
    else:
        peak_metric_df = pd.DataFrame()
        compact_metrics_df = pd.DataFrame()
        st.info("Select or manually define peaks to calculate Epa, ipa, Epc, ipc, Delta Ep, and ipa/ipc.")

    export_buffer = io.StringIO()
    export_sections = []
    export_sections.append("# ESW summary\n" + esw_df.to_csv(index=False))
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
        parser_summary,
        esw_df,
        peak_metric_df,
        compact_metrics_df,
        {
            "selected_file": selected_filename,
            "column_mode": column_mode,
            "potential_unit": selected_potential_unit,
            "current_unit": selected_current_unit,
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
