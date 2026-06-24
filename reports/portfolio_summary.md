# VoltScope AI Portfolio Summary

## 2-3 Sentence Project Summary

VoltScope AI is a Streamlit research-software prototype for parsing, auditing, plotting, and interpreting cyclic voltammetry (CV) and linear sweep voltammetry (LSV) data. I built it to connect my biochemistry background with chemical/electrochemical engineering workflows, using Python, data analysis, and AI-assisted interpretation to make electrochemical results more transparent and reproducible.

## GitHub / LinkedIn Project Description

VoltScope AI analyzes messy electrochemical data exports by detecting columns, metadata, units, CV peaks, LSV onset potentials, and data-quality issues. The app emphasizes research-grade transparency: it flags ambiguous units, missing metadata, noisy traces, invalid reversible-pair metrics, and operational threshold assumptions before results are treated as final. Optional AI interpretation is grounded in computed VoltScope metrics and warnings rather than raw uploaded arrays.

## Technical Skills Demonstrated

- Scientific Python: NumPy, pandas, Streamlit, Plotly, regression testing
- Electrochemical analysis: CV peak metrics, reversible/irreversible behavior classification, multi-cycle CV handling, LSV onset detection
- Data quality and parsing: delimiter/header detection, metadata extraction, unit conversion, corrupted-row cleanup, missing-metadata flags
- Research UX: parser summaries, audit tables, review-needed states, unit-confirmation workflow, current-density normalization checks
- AI-assisted research tooling: OpenAI integration using structured metric payloads, stale-output detection, metric-validity guardrails
- Software engineering: modular repository organization, tests for edge cases, secret handling, reproducible demo data

## Resume Project Section

**VoltScope AI | Electrochemical data analysis prototype**

- Built a Streamlit research-software prototype that parses messy CV/LSV potentiostat exports, normalizes units, detects metadata, plots traces, and computes electrochemical metrics with data-quality review flags.
- Implemented CV peak classification for reversible-like, noisy/ambiguous, irreversible, multi-cycle, and multi-couple traces, plus LSV sustained-threshold onset detection with reference-electrode, scan-rate, and current-density context.
- Added AI-assisted interpretation from structured computed metrics and warnings, excluding raw uploaded arrays and preserving researcher review for uncertain or invalid metrics.
- Created regression tests for clean, noisy, irreversible, unit-ambiguous, metadata-rich, and corrupted-file cases to stabilize scientific behavior.

## One-Line Resume Bullet

- Built VoltScope AI, a Python/Streamlit electrochemical analysis prototype for parsing messy CV/LSV data, computing peak/onset metrics, flagging unit/metadata uncertainty, and generating auditable AI-assisted interpretation summaries.

## Grad School Application Narrative

VoltScope AI represents my effort to bridge biochemistry training with electrochemical engineering and scientific computing. I designed the project around practical research problems: real instrument exports are messy, units and metadata are often incomplete, and electrochemical interpretation depends on knowing when a metric is valid versus when it needs review. By combining parser design, numerical analysis, interactive visualization, regression tests, and AI-assisted documentation, the project shows how computational tools can improve reliability and transparency in electrochemical workflows.

## Longer Project Description

VoltScope AI turns raw CV and LSV files into audited plots, quantitative metrics, and cautious interpretation summaries. For CV, it reports Epa, Ipa, Epc, Ipc, ΔEp, E°′, |Ipa/Ipc|, and behavior classification when appropriate, while suppressing reversible-pair metrics for noisy or irreversible traces. For LSV, it reports operational onset potential, threshold source, onset reliability, max anodic/cathodic current or current density, scan rate, reference electrode, and normalization status. The app uses review flags and deterministic limitations to distinguish parser success from research readiness.

## Why This Project Matters

Electrochemical conclusions can change when units, reference electrodes, scan rates, electrode area, baseline drift, or noisy candidate peaks are mishandled. VoltScope AI demonstrates a research workflow where software does not simply output numbers; it documents assumptions, warns about uncertainty, and supports a researcher in deciding whether results are ready for interpretation. This connects directly to electrochemistry, chemical engineering, biochemistry, and AI-assisted scientific workflows.
