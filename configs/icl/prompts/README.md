# Prompt configuration

Prompt templates for the generator, low-level policy and value component are
configured under `prompts` in `configs/icl/experiments/base.json`. Child
configurations can override individual fields using `extends`.

Use `python -m scripts.run_icl_search preview --config <config>` to inspect
assembled messages without making a model request. Policy prompt, examples,
decoding and model identity are part of the controller configuration and must
match the calibration used for evaluation. Keep credentials outside configuration
files and version control.
