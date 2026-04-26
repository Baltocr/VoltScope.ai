"""
VoltScope AI MVP
----------------
A simple Streamlit app for electrochemistry data analysis.

Current features:
- Upload CSV data
- Select potential and current/current-density columns
- Plot current density vs potential
- Choose current-density threshold
- Estimate cathodic and anodic threshold-crossing potentials
- Calculate operational electrochemical stability window (ESW)
- Generate a short interpretation paragraph

Run locally:
    pip install streamlit pandas numpy plotly
    streamlit run voltscope_ai_mvp.py
"""

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class ESWResult:
    cathodic_limit: Optional[float]
    anodic_limit: Optional[float]
    esw: Optional[float]


def find_threshold_crossing(
    potential: np.ndarray,
    current_density: np.ndarray,
    threshold: float,
    direction: str,
) -> Optional[float]:
    """
    Estimate the potential where current density crosses a threshold.

    Parameters
    ----------
    potential:
        Array of potential values.
    current_density:
        Array of current density values.
    threshold:
        Positive threshold value in mA/cm^2.
    direction:
        "anodic" for +threshold or "cathodic" for -threshold.

    Returns
    -------
    Estimated potential at threshold crossing, or None if no crossing found.
    """
    if direction not in {"anodic", "cathodic"}:
        raise ValueError("direction must be 'anodic' or 'cathodic'")

    target = threshold if direction == "anodic" else -threshold

    # Sort by potential so interpolation behaves more predictably.
    order = np.argsort(potential)
    x = potential[order]
    y = current_density[order]

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

        crosses_target = (y1 - target) * (y2 - target) < 0
        if not crosses_target:
            continue

        crossing = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        crossings.append(float(crossing))

    if crossings:
        if direction == "anodic":
            return min(crossings)
        return max(crossings)

    mask = y >= target if direction == "anodic" else y <= target
    if not np.any(mask):
        return None

    threshold_points = x[mask]
    if direction == "anodic":
        return float(np.min(threshold_points))
    return float(np.max(threshold_points))


def calculate_esw(
    potential: np.ndarray,
    current_density: np.ndarray,
    threshold: float,
) -> ESWResult:
    cathodic = find_threshold_crossing(
        potential, current_density, threshold, direction="cathodic"
    )
    anodic = find_threshold_crossing(
        potential, current_density, threshold, direction="anodic"
    )

    if cathodic is None or anodic is None:
        return ESWResult(cathodic, anodic, None)

    return ESWResult(cathodic, anodic, anodic - cathodic)


def build_interpretation(sample_name: str, threshold: float, result: ESWResult) -> str:
    if result.esw is None:
        return (
            f"Using +/-{threshold:g} mA/cm^2 as the operational current-density threshold, "
            f"the full electrochemical stability window could not be determined for {sample_name} "
            f"because one or both threshold crossings were not detected in the uploaded data range."
        )

    return (
        f"Using +/-{threshold:g} mA/cm^2 as the operational current-density threshold, "
        f"{sample_name} shows a cathodic limit of {result.cathodic_limit:.3f} V and an anodic "
        f"limit of {result.anodic_limit:.3f} V, giving an operational electrochemical stability "
        f"window of {result.esw:.3f} V under these experimental conditions. Because this value is "
        f"defined by a selected current threshold, it should be interpreted as an operational ESW "
        f"rather than an absolute thermodynamic limit."
    )


def main() -> None:
    st.set_page_config(page_title="VoltScope AI MVP", layout="wide")

    st.title("VoltScope AI MVP")
    st.caption("Electrochemistry data analysis + research communication assistant")

    st.markdown(
        "Upload a CSV file containing potential and current/current-density data. "
        "This MVP estimates the operational ESW using a user-defined current-density threshold."
    )

    uploaded_file = st.file_uploader("Upload electrochemistry CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        st.stop()
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        st.stop()
        return

    st.subheader("Raw data preview")
    st.dataframe(df.head(20), use_container_width=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        st.error("The CSV must contain at least two numeric columns.")
        st.stop()
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        potential_col = st.selectbox("Potential column", numeric_columns)

    with col2:
        current_col = st.selectbox("Current or current-density column", numeric_columns)

    with col3:
        sample_name = st.text_input("Sample name", value="Sample 1")

    st.subheader("Current conversion")
    current_is_density = st.radio(
        "Is your selected current column already normalized as current density?",
        ["Yes, already mA/cm^2", "No, convert current to mA/cm^2"],
        horizontal=True,
    )

    working = df[[potential_col, current_col]].dropna().copy()
    potential = working[potential_col].to_numpy(dtype=float)
    current_values = working[current_col].to_numpy(dtype=float)

    if current_is_density == "Yes, already mA/cm^2":
        current_density = current_values
        y_label = "Current density (mA/cm^2)"
    else:
        area = st.number_input(
            "Electrode geometric area (cm^2)",
            min_value=0.000001,
            value=1.0,
            step=0.1,
            format="%.6f",
        )
        current_unit = st.selectbox("Current unit in uploaded file", ["mA", "A", "uA"])

        if current_unit == "A":
            current_mA = current_values * 1000.0
        elif current_unit == "uA":
            current_mA = current_values / 1000.0
        else:
            current_mA = current_values

        current_density = current_mA / area
        y_label = "Current density (mA/cm^2)"

    threshold = st.number_input(
        "Threshold current density (mA/cm^2)",
        min_value=0.000001,
        value=0.5,
        step=0.1,
        format="%.6f",
    )

    result = calculate_esw(potential, current_density, threshold)

    st.subheader("Plot")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=potential,
            y=current_density,
            mode="lines+markers",
            name=sample_name,
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"+{threshold:g} mA/cm^2")
    fig.add_hline(y=-threshold, line_dash="dash", annotation_text=f"-{threshold:g} mA/cm^2")

    if result.anodic_limit is not None:
        fig.add_vline(
            x=result.anodic_limit,
            line_dash="dot",
            annotation_text=f"Anodic: {result.anodic_limit:.3f} V",
        )

    if result.cathodic_limit is not None:
        fig.add_vline(
            x=result.cathodic_limit,
            line_dash="dot",
            annotation_text=f"Cathodic: {result.cathodic_limit:.3f} V",
        )

    fig.update_layout(
        xaxis_title=f"Potential ({potential_col})",
        yaxis_title=y_label,
        template="plotly_white",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Calculated ESW")
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Cathodic limit",
        "Not found" if result.cathodic_limit is None else f"{result.cathodic_limit:.3f} V",
    )
    metric_cols[1].metric(
        "Anodic limit",
        "Not found" if result.anodic_limit is None else f"{result.anodic_limit:.3f} V",
    )
    metric_cols[2].metric(
        "Operational ESW",
        "Not found" if result.esw is None else f"{result.esw:.3f} V",
    )

    st.subheader("Generated interpretation")
    interpretation = build_interpretation(sample_name, threshold, result)
    st.write(interpretation)

    export_df = pd.DataFrame(
        {
            "sample_name": [sample_name],
            "threshold_mA_cm2": [threshold],
            "cathodic_limit_V": [result.cathodic_limit],
            "anodic_limit_V": [result.anodic_limit],
            "operational_ESW_V": [result.esw],
            "interpretation": [interpretation],
        }
    )

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    st.download_button(
        "Download ESW results as CSV",
        data=csv_buffer.getvalue(),
        file_name="voltscope_esw_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
