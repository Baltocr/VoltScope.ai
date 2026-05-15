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
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


POTENTIAL_UNITS_TO_V = {"V": 1.0, "mV": 1e-3}
CURRENT_UNITS_TO_A = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9}


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
        potential_unit = detect_unit_from_header(column, "potential")
        current_unit = detect_unit_from_header(column, "current")
        if potential_unit:
            detected_units[f"potential:{column}"] = potential_unit
        if current_unit:
            detected_units[f"current:{column}"] = current_unit

    warnings = []
    if not has_header:
        warnings.append("No header row was detected; generated generic column names.")
    if potential_col is None:
        warnings.append("Potential column could not be detected automatically.")
    if current_col is None:
        warnings.append("Current column could not be detected automatically.")

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


def current_from_amps(values: np.ndarray, display_unit: str) -> np.ndarray:
    return values / CURRENT_UNITS_TO_A.get(display_unit, 1.0)


def current_value_to_amps(value: float, display_unit: str) -> float:
    return value * CURRENT_UNITS_TO_A.get(display_unit, 1.0)


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


def peak_label(peak: Peak, display_unit: str) -> str:
    current_display = current_from_amps(np.array([peak.raw_current]), display_unit)[0]
    prominence_display = current_from_amps(np.array([peak.prominence]), display_unit)[0]
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
        raw_display = current_from_amps(np.array([raw_current]), display_unit)[0]
        baseline_display = current_from_amps(np.array([baseline_at_peak]), display_unit)[0]
        corrected_display = current_from_amps(np.array([corrected_current]), display_unit)[0]

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


def metrics_table(metrics: dict[str, Optional[float]], display_unit: str) -> pd.DataFrame:
    rows = []
    if metrics["epa_V"] is not None:
        rows.append({"metric": "Epa", "value": f"{metrics['epa_V']:.4g} V"})
    if metrics["ipa_A"] is not None:
        ipa_display = current_from_amps(np.array([metrics["ipa_A"]]), display_unit)[0]
        rows.append({"metric": "ipa", "value": f"{ipa_display:.4g} {display_unit}"})
    if metrics["epc_V"] is not None:
        rows.append({"metric": "Epc", "value": f"{metrics['epc_V']:.4g} V"})
    if metrics["ipc_A"] is not None:
        ipc_display = current_from_amps(np.array([metrics["ipc_A"]]), display_unit)[0]
        rows.append({"metric": "ipc", "value": f"{ipc_display:.4g} {display_unit}"})
    if metrics["delta_ep_V"] is not None:
        rows.append({"metric": "Delta Ep", "value": f"{metrics['delta_ep_V']:.4g} V"})
    if metrics["ipa_ipc_ratio"] is not None:
        rows.append({"metric": "ipa/ipc", "value": f"{metrics['ipa_ipc_ratio']:.4g}"})
    return pd.DataFrame(rows)


# ---------- Streamlit UI ----------


def get_project_workspace(project_name: str) -> dict[str, Any]:
    if "project_workspaces" not in st.session_state:
        st.session_state.project_workspaces = {}

    workspaces = st.session_state.project_workspaces
    if project_name not in workspaces:
        workspaces[project_name] = {"analyses": {}, "notes": {}}

    return workspaces[project_name]


def store_project_analyses(project_workspace: dict[str, Any], analyzed: list[AnalyzedDataset]) -> None:
    analyses = project_workspace.setdefault("analyses", {})
    for dataset in analyzed:
        analyses[dataset.filename] = dataset


def main() -> None:
    st.set_page_config(page_title="VoltScope AI MVP", layout="wide")

    st.title("VoltScope AI MVP")
    st.caption("Cyclic voltammetry upload, plotting, peak detection, and baseline correction")

    with st.sidebar:
        st.header("Project workspace")
        project_name_input = st.text_input("Project name", value="Untitled project")
        project_name = project_name_input.strip() or "Untitled project"
        project_workspace = get_project_workspace(project_name)
        stored_files = sorted(project_workspace.get("analyses", {}).keys())
        st.caption(f"Active project: {project_name}")
        if stored_files:
            st.write(f"Stored analyses: {len(stored_files)}")
            st.dataframe(pd.DataFrame({"file": stored_files}), use_container_width=True, hide_index=True)
        else:
            st.write("No analyses stored in this project yet.")

    uploaded_files = st.file_uploader(
        "Upload CV data files",
        type=["csv", "txt", "tsv", "dat"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more .csv, .txt, .tsv, or .dat files to begin.")
        st.stop()
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
    st.dataframe(parser_summary, use_container_width=True)

    selected_filename = st.selectbox(
        "Preview / detailed analysis file",
        [dataset.filename for dataset in parsed_datasets],
    )
    selected_dataset = next(dataset for dataset in parsed_datasets if dataset.filename == selected_filename)

    with st.expander("Raw parsed data preview", expanded=True):
        st.dataframe(selected_dataset.dataframe.head(30), use_container_width=True)

    with st.expander("Detected metadata"):
        if selected_dataset.metadata:
            st.json(selected_dataset.metadata)
        else:
            st.write("No metadata rows detected before the numeric table.")

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

    store_project_analyses(project_workspace, analyzed)

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
    current_display_values = current_from_amps(analysis_current_a, current_display_unit)
    current_range = float(np.nanmax(current_display_values) - np.nanmin(current_display_values))
    default_prominence = max(current_range * 0.05, 1e-12 / CURRENT_UNITS_TO_A[current_display_unit])

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
        current_value_to_amps(min_prominence_display, current_display_unit),
        int(min_distance),
        current_value_to_amps(min_abs_current_display, current_display_unit),
    )

    peak_rows = [
        {
            "id": peak.id,
            "type": peak.peak_type,
            "index": peak.index,
            "potential_V": peak.potential,
            f"current_{current_display_unit}": current_from_amps(
                np.array([peak.raw_current]), current_display_unit
            )[0],
            f"prominence_{current_display_unit}": current_from_amps(
                np.array([peak.prominence]), current_display_unit
            )[0],
            "segment": peak.segment_index,
            "confidence": peak.confidence,
        }
        for peak in peaks
    ]
    st.dataframe(pd.DataFrame(peak_rows), use_container_width=True)

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
                labels = [peak_label(peak, current_display_unit) for peak in oxidation_peaks]
                selected_label = st.selectbox("Oxidation peak", labels)
                selected_oxidation_peak = oxidation_peaks[labels.index(selected_label)]
            else:
                st.info("No oxidation peaks detected. Lower the prominence threshold or use manual selection.")
        with select_col2:
            if reduction_peaks:
                labels = [peak_label(peak, current_display_unit) for peak in reduction_peaks]
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
                        np.array([analysis_current_a[anchor_a_idx]]), current_display_unit
                    )[0],
                },
                {
                    "anchor": "B",
                    "index": anchor_b_idx,
                    "potential_V": detailed_dataset.potential_v[anchor_b_idx],
                    f"current_{current_display_unit}": current_from_amps(
                        np.array([analysis_current_a[anchor_b_idx]]), current_display_unit
                    )[0],
                },
            ]
        )
        st.dataframe(anchor_summary, use_container_width=True)

    fig = go.Figure()
    trace_mode = "lines+markers" if show_markers else "lines"

    for dataset in analyzed:
        raw_display = current_from_amps(dataset.raw_current_a, current_display_unit)
        if dataset.smoothed_current_a is not None and show_raw_with_smoothed:
            fig.add_trace(
                go.Scatter(
                    x=dataset.potential_v,
                    y=raw_display,
                    mode="lines",
                    name=f"{dataset.sample_name} raw",
                    opacity=0.35,
                    hovertemplate="Potential: %{x:.4g} V<br>Current: %{y:.4g} "
                    + current_display_unit
                    + "<extra></extra>",
                )
            )
        elif dataset.smoothed_current_a is None:
            fig.add_trace(
                go.Scatter(
                    x=dataset.potential_v,
                    y=raw_display,
                    mode=trace_mode,
                    name=dataset.sample_name,
                    hovertemplate="Potential: %{x:.4g} V<br>Current: %{y:.4g} "
                    + current_display_unit
                    + "<extra></extra>",
                )
            )

        if dataset.smoothed_current_a is not None:
            smoothed_display = current_from_amps(dataset.smoothed_current_a, current_display_unit)
            fig.add_trace(
                go.Scatter(
                    x=dataset.potential_v,
                    y=smoothed_display,
                    mode=trace_mode,
                    name=f"{dataset.sample_name} smoothed",
                    hovertemplate="Potential: %{x:.4g} V<br>Current: %{y:.4g} "
                    + current_display_unit
                    + "<extra></extra>",
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
                    ),
                    mode="markers",
                    name=name,
                    marker={"size": 10, "symbol": symbol},
                    hovertemplate="Potential: %{x:.4g} V<br>Current: %{y:.4g} "
                    + current_display_unit
                    + "<extra></extra>",
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
                y=current_from_amps(baseline_current, current_display_unit),
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
        yaxis_title=f"Current / {current_display_unit}",
        template="plotly_white",
        height=620,
        legend_title_text="Trace",
    )

    st.subheader("Interactive CV plot")
    st.plotly_chart(fig, use_container_width=True)

    plot_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    st.download_button(
        "Download interactive plot as HTML",
        data=plot_html,
        file_name="voltscope_cv_plot.html",
        mime="text/html",
    )

    st.subheader("Operational ESW summary")
    threshold_display = st.number_input(
        f"ESW threshold current ({current_display_unit})",
        min_value=0.0,
        value=float(max(current_range * 0.1, 1e-12 / CURRENT_UNITS_TO_A[current_display_unit])),
        step=float(max(current_range * 0.05, 1e-12 / CURRENT_UNITS_TO_A[current_display_unit])),
        format="%.6e",
    )
    threshold_a = current_value_to_amps(threshold_display, current_display_unit)
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
    st.dataframe(esw_df, use_container_width=True)

    st.subheader("Peak metrics")
    peak_metric_rows, metrics = build_peak_metrics_rows(
        selected_oxidation_peak,
        selected_reduction_peak,
        baseline_current,
        analysis_current_a,
        current_display_unit,
    )
    if peak_metric_rows:
        peak_metric_df = pd.DataFrame(peak_metric_rows)
        st.dataframe(peak_metric_df, use_container_width=True)
        st.dataframe(metrics_table(metrics, current_display_unit), use_container_width=True)
    else:
        peak_metric_df = pd.DataFrame()
        st.info("Select or manually define peaks to calculate Epa, ipa, Epc, ipc, Delta Ep, and ipa/ipc.")

    export_buffer = io.StringIO()
    export_sections = []
    export_sections.append("# ESW summary\n" + esw_df.to_csv(index=False))
    if peak_metric_rows:
        export_sections.append("# Peak metrics\n" + peak_metric_df.to_csv(index=False))
        export_sections.append("# Compact metrics\n" + metrics_table(metrics, current_display_unit).to_csv(index=False))
    export_buffer.write("\n".join(export_sections))

    st.download_button(
        "Download analysis results as CSV",
        data=export_buffer.getvalue(),
        file_name="voltscope_cv_analysis_results.csv",
        mime="text/csv",
    )

    st.subheader("Project notebook")
    st.caption(f"Notes for {selected_filename} in {project_name}")
    notes = project_workspace.setdefault("notes", {})
    note_key = f"project_note::{project_name}::{selected_filename}"
    if note_key not in st.session_state:
        st.session_state[note_key] = notes.get(selected_filename, "")
    notes[selected_filename] = st.text_area(
        "File notes",
        key=note_key,
        height=180,
        placeholder="Record observations, electrolyte details, electrode prep, follow-up steps, or interpretation notes.",
    )


if __name__ == "__main__":
    main()
