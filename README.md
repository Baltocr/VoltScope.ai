# VoltScope AI

VoltScope AI is a Streamlit-based research-software prototype for parsing, auditing, visualizing, and interpreting cyclic voltammetry (CV) and linear sweep voltammetry (LSV) data.

This project reflects my transition from biochemistry into chemical and electrochemical engineering: it combines electrochemical domain knowledge with Python, scientific data analysis, uncertainty-aware workflows, and AI-assisted research documentation.

## Why This Project Exists

Electrochemical measurements are often limited not by a lack of data, but by messy data handling. Potentiostat exports may contain metadata before the table, inconsistent delimiters, missing units, ambiguous current-density normalization, noisy extrema, or corrupted numeric rows. In CV and LSV workflows, these details can change peak currents, onset potentials, current-density values, and conclusions about reversibility or stability.

VoltScope AI is designed as a serious research-software prototype for making those assumptions visible. It parses real-world-style files, computes electrochemical metrics, flags uncertainty, and helps produce auditable interpretation notes without replacing researcher judgment.

## What the App Does

- Uploads electrochemical files in `.csv`, `.txt`, `.tsv`, or `.dat` format
- Detects delimiters, metadata rows, headers, numeric table boundaries, and usable columns
- Maps potential/current/current-density columns and normalizes units for analysis
- Requires unit confirmation when current or current-density units are ambiguous
- Plots interactive CV and LSV traces with selected peak/onset annotations
- Computes CV peak metrics and LSV onset metrics
- Surfaces parser cleanup, missing metadata, invalid metrics, noisy traces, and review-needed conditions
- Generates deterministic quick interpretations and optional OpenAI-assisted detailed interpretations from structured analysis payloads

## Supported Electrochemical Techniques

- **Cyclic voltammetry (CV)**
- **Linear sweep voltammetry (LSV)**

The portfolio version is intentionally focused on CV and LSV rather than trying to cover every electrochemical technique superficially.

## Demo Cases

Four curated demo files are included in [`sample_data/`](sample_data):

- [`cv_clean_reversible.csv`](sample_data/cv_clean_reversible.csv): clean reversible-like CV with clear anodic and cathodic peaks
- [`cv_noisy_ambiguous.csv`](sample_data/cv_noisy_ambiguous.csv): noisy CV where peak assignment and reversible-pair metrics require review
- [`cv_oxidation_only.csv`](sample_data/cv_oxidation_only.csv): irreversible oxidation-only CV with no reliable cathodic return peak
- [`lsv_anodic_onset_with_metadata.txt`](sample_data/lsv_anodic_onset_with_metadata.txt): LSV with reference electrode, scan-rate, electrode-area metadata, and anodic onset detection

## CV Analysis

For CV files, VoltScope AI reports:

- **Epa**: anodic peak potential
- **Ipa**: anodic peak current
- **Epc**: cathodic peak potential
- **Ipc**: cathodic peak current
- **ΔEp**: peak separation when a valid Epa/Epc pair exists
- **E°′**: formal potential estimate, `(Epa + Epc) / 2`, when valid
- **|Ipa/Ipc|**: peak current ratio when valid
- CV behavior classification:
  - reversible-like
  - quasi-reversible / review needed
  - noisy / ambiguous
  - irreversible oxidation-only
  - irreversible reduction-only
  - multiple redox couples
  - no reliable peaks
- Multi-cycle analysis when cycle structure is present or inferred

VoltScope AI does not force reversible-pair metrics when the data do not support them. If a trace is oxidation-only, reduction-only, noisy, or has an invalid selected pair, ΔEp, E°′, and |Ipa/Ipc| are marked as not applicable, invalid, tentative, or review-needed rather than displayed as final quantitative results.

## LSV Analysis

For LSV files, VoltScope AI reports:

- Operational onset potential from a selected threshold rule
- Threshold value, sign, source, and method
- Anodic or cathodic direction
- Maximum anodic current/current density or maximum cathodic current/current density, depending on the trace
- Potential at limiting current/current density
- Onset reliability based on sustained threshold crossing
- Reference electrode, scan rate, and electrode area when available
- Whether the uploaded signal is raw current, already current density, or normalized by VoltScope

LSV onset is treated as an operational threshold-crossing value, not a universal physical constant. The app reports the threshold, reference electrode, scan rate, smoothing/onset-detection method, and normalization state so the result can be interpreted in context.

## Data Quality and Uncertainty Handling

VoltScope AI separates successful computation from research readiness. A file can parse and plot successfully while still requiring review.

The app flags:

- Missing or ambiguous current/current-density units
- Suspicious current magnitudes after unit conversion
- Missing reference electrode, scan rate, or electrode area metadata
- Noisy CV traces with competing candidate extrema
- Invalid reversible-pair metrics, including Epa lower than Epc
- Irreversible or one-sided CV behavior where return peaks are not reliable
- LSV threshold crossings caused by scan boundaries or isolated noise spikes
- Recoverable parser cleanup, such as dropped non-numeric rows inside a data table

The goal is not to hide imperfect data. The goal is to make limitations explicit before a researcher uses the numbers.

## AI Interpretation System

The optional AI interpretation system uses the OpenAI API only when configured by the user. It receives a structured VoltScope payload containing computed metrics, metric-validity flags, metadata status, parser warnings, quality flags, selected audit fields, and user notes.

The AI does **not** receive raw uploaded data arrays. It is instructed to avoid unsupported claims, avoid interpreting invalid or pending metrics as final, and include a lab-notebook-style note. AI output is best treated as a draft interpretation aid, not as proof of mechanism, kinetics, diffusion behavior, or publication-ready conclusions.

## Repository Layout

```text
voltscope-ai/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── parsing/
│   ├── analysis/
│   ├── ai/
│   ├── plotting/
│   └── utils/
├── sample_data/
├── tests/
├── reports/
│   └── demo_outputs/
├── MVP 1/
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

The current Streamlit implementation remains in `MVP 1/voltscope_ai_mvp.py` for compatibility with existing saved workspaces. The portfolio-facing entrypoint is `app/streamlit_app.py`, with reusable package structure in `src/` for continued refactoring.

## Run Locally

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app/streamlit_app.py
```

The compatibility entrypoint also works:

```bash
streamlit run "MVP 1/voltscope_ai_mvp.py"
```

## OpenAI API Key

AI interpretation is optional. The app never hard-codes API keys and first checks Streamlit secrets, then the `OPENAI_API_KEY` environment variable.

Recommended local Streamlit setup:

```bash
mkdir -p .streamlit
```

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Alternative shell setup:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

`.env.example` documents the expected variable names, but the app does not require committing or exposing a `.env` file. Local secret files are ignored by git.

## Tests

Run the regression suite with:

```bash
python -m pytest
```

You can also run the same tests through `unittest`:

```bash
python -m unittest discover tests
```

The tests cover parser behavior, unit conversion, CV peak classification, LSV onset detection, metadata handling, unit confirmation, corrupted-row cleanup, and the curated portfolio demo cases. If `pytest` is not found, activate the virtual environment and reinstall dependencies with `pip install -r requirements.txt`.

## Limitations

- This is a research-software prototype, not validated analytical instrumentation software.
- Synthetic demo files are designed for reproducible testing and demonstration, not as universal electrochemistry benchmarks.
- CV classification uses heuristics and should be reviewed for noisy, baseline-sensitive, irreversible, multi-cycle, or multi-couple traces.
- LSV onset values are operational threshold-crossing estimates and should be reported with threshold, reference electrode, scan rate, normalization, and smoothing context.
- AI interpretation should be reviewed by the researcher and should not be treated as mechanistic proof.

## Future Work

- Continue extracting reusable parser, analysis, plotting, and AI code from the compatibility Streamlit module into `src/`
- Add a command-line or package API for batch processing
- Expand metadata schemas for common potentiostat exports
- Add more experimentally validated benchmark datasets
- Improve baseline correction, manual peak selection, and onset audit workflows
- Add report export templates for lab notebooks, posters, or research summaries

## Portfolio Framing

VoltScope AI demonstrates scientific Python, electrochemical data analysis, research-grade uncertainty handling, interactive visualization, regression testing, and cautious AI-assisted interpretation. It is intended as a portfolio project for graduate study and technical roles at the intersection of biochemistry, chemical/electrochemical engineering, and scientific computing.
