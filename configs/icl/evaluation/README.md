# Evaluating saved runs

`python -m scripts.evaluate_icl` reads saved experiment runs and produces
summaries and Plotly figures without model inference. Use `--runs` and
`--output` to select input and output directories; `--formats` optionally
exports static figures.

The report records coverage, action and model-call costs, calibration and
paired comparisons. Check the configuration and evaluation cohort before
combining runs. A mock result is an interface check, not a model measurement.
