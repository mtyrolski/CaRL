"""Exercise the interactive walkthrough without Jupyter network sockets or API calls."""
import contextlib
import io
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/figure1_reachability.ipynb"


def execute_notebook(tmp_path, monkeypatch, *, provider):
    """Run notebook cells in-process with an isolated output directory."""
    import IPython.display
    import plotly.basedatatypes
    from carl.icl.api import CachedOpenAI

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(IPython.display, "display", lambda *_a, **_k: None)
    figures = []
    monkeypatch.setattr(plotly.basedatatypes.BaseFigure, "show", lambda fig, *_a, **_k: figures.append(fig))

    def no_client(*_args, **_kwargs):
        pytest.fail("The walkthrough unexpectedly attempted to create a live API client")

    monkeypatch.setattr(CachedOpenAI, "_client", no_client)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source.replace(
            'OUTPUT_DIR = ROOT / "analysis/luna_icl" / f"figure1_walkthrough_v2_{PROVIDER}"',
            f"OUTPUT_DIR = Path({str(tmp_path / 'output')!r})",
        )
        source = source.replace('raw["output_dir"] = str(OUTPUT_DIR)',
                                f'raw["output_dir"] = str(OUTPUT_DIR)\nraw["cache_dir"] = {str(tmp_path / "cache")!r}')
        if provider == "openai":
            source = source.replace('PROVIDER = "mock"', 'PROVIDER = "openai"')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f"{NOTEBOOK}:cell{index}", "exec"), namespace)  # noqa: S102
    namespace["captured_figures"] = figures
    return namespace


def test_walkthrough_runs_mock_freeze_replay_and_report(tmp_path, monkeypatch):
    if not (ROOT / "data/icl/real_12x12x4/validation.json").exists():
        pytest.skip("Prepared ICL validation trajectories are not available")
    result = execute_notebook(tmp_path, monkeypatch, provider="mock")
    assert result["manifest"]["provider"] == "mock"
    assert len(result["records"]) == 36
    assert all(row["status"] == "complete" for row in result["records"])
    assert len(result["targets"]) == 12
    assert result["selected_record"]["status"] == "complete"
    assert any("[MOCK]" in (fig.layout.title.text or "") for fig in result["captured_figures"])
    assert (tmp_path / "output/targets.csv").exists()
    assert (tmp_path / "output/reachability.pdf").exists()


def test_live_mode_does_not_execute_without_explicit_opt_in(tmp_path, monkeypatch):
    result = execute_notebook(tmp_path, monkeypatch, provider="openai")
    assert result["RUN_PAID"] is False
    assert result["manifest"] is None
    assert result["records"] == []
    assert not (tmp_path / "output").exists()
