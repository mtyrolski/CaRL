# In-context experiment configurations

`base.json` defines shared component settings. Other configurations can inherit
from it using `extends`; the loader validates fields and rejects inheritance
cycles. Matrix files list condition configurations for paired runs.

Inspect available commands with:

```bash
uv run --extra icl python -m scripts.run_icl_search --help
```

Use `--mock` for an offline interface check, `--replay` for previously cached
responses, and `--execute` for live model calls. Changes to the controller,
prompt or decoding require a matching calibration. Data and outputs are stored
locally rather than in this directory.
