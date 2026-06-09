## Paper for Progressive Frequency-conditioned FiLM

This directory contains the LaTeX source for the paper:

**Progressive Frequency-conditioned FiLM for iTransformer-based Long-term Time Series Forecasting**

The paper corresponds to the model implementation in `models/MVAR_iTransformer_v32.py`.

### Directory Structure

- `main.tex`: Main LaTeX file (XeLaTeX)
- `sections/`: Chapter files
  - `01_introduction.tex` — Introduction and contributions
  - `02_related_work.tex` — Related work (deep forecasting, frequency analysis, conditional modulation, non-stationarity, scientific regression)
  - `03_method.tex` — Method (RevIN, Multi-Resolution Spectrum, Spectrum Compressor, Layer-wise Progressive FiLM, Learnable Gamma)
  - `04_experiments.tex` — Experiments (datasets, baselines, main results, ablation, sensitivity)
  - `05_conclusion.tex` — Conclusion and future work
  - `A_appendix.tex` — Appendix (notation, complexity analysis, degradation verification)
- `references.bib`: Bibliography (replace placeholder entries marked as Placeholder with real references)

### Model--Paper Mapping

| Code Component | Paper Section |
|---|---|
| RevIN | \S 3.2 |
| Variate Token Embedding | \S 3.3 |
| `MultiResolutionSpectrum` | \S 3.4 Multi-Resolution Spectrum Feature Extraction |
| `SpectrumCompressor` | \S 3.5 Spectrum Compressor |
| `FiLMGenerator` (per-layer) + `log_gamma_base` + `decay` | \S 3.6 Layer-wise Progressive FiLM |
| Transformer Encoder + Head | \S 3.7 Prediction Head and Task Adaptation |
| Zero-init + tanh + log1p + clamp | \S 3.8 Graceful Degradation and Stabilization |

### Compilation

```bash
cd paper
latexmk -xelatex -interaction=nonstopmode main.tex
```

Without `latexmk`:

```bash
cd paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

### Remaining Work

- **Dataset details**: fill in `sections/04_experiments.tex`
- **Baseline names and citations**: fill in `sections/04_experiments.tex`
- **Numerical results and figures**: fill in tables and add files to `figures/`
- **Real bibliography**: replace placeholder entries in `references.bib`
