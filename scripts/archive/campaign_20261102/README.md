# Historical campaign tools (November 2026)

These 34 tools preserve historical experiment preparation, calibration diagnostics and reports. The supported new campaign entrypoint is `uv run --extra icl python -m scripts.launch100`; these archived tools are not part of that runtime.

Run an archived tool as a module from the repository root. For example:

```bash
uv run --extra icl python -m scripts.archive.campaign_20261102.campaign_budget_status
```

Existing `analysis/` reports, frozen manifests, cached responses and `campaign.md` retain their original paths and hashes. They were not rewritten to suggest that old runs used the relocated code. A historical frozen manifest may therefore no longer match the current source tree; preserve its frozen source archive for replay. Moving an entrypoint does not authorize paid execution.

No forwarding wrappers remain at the old script paths.

| Previous file | Current module |
| --- | --- |
| `scripts/audit_calibration_v2.py` | `scripts.archive.campaign_20261102.audit_calibration_v2` |
| `scripts/audit_scale_io.py` | `scripts.archive.campaign_20261102.audit_scale_io` |
| `scripts/calibration_progress.py` | `scripts.archive.campaign_20261102.calibration_progress` |
| `scripts/campaign_bfs32_cllp.py` | `scripts.archive.campaign_20261102.campaign_bfs32_cllp` |
| `scripts/campaign_budget_status.py` | `scripts.archive.campaign_20261102.campaign_budget_status` |
| `scripts/campaign_components_validation.py` | `scripts.archive.campaign_20261102.campaign_components_validation` |
| `scripts/campaign_generator_prompt.py` | `scripts.archive.campaign_20261102.campaign_generator_prompt` |
| `scripts/campaign_pilot_runner.py` | `scripts.archive.campaign_20261102.campaign_pilot_runner` |
| `scripts/campaign_policy_successors.py` | `scripts.archive.campaign_20261102.campaign_policy_successors` |
| `scripts/campaign_policy_validation.py` | `scripts.archive.campaign_20261102.campaign_policy_validation` |
| `scripts/campaign_primitive_baseline.py` | `scripts.archive.campaign_20261102.campaign_primitive_baseline` |
| `scripts/campaign_real12_stress.py` | `scripts.archive.campaign_20261102.campaign_real12_stress` |
| `scripts/campaign_scale_stress.py` | `scripts.archive.campaign_20261102.campaign_scale_stress` |
| `scripts/campaign_value_current.py` | `scripts.archive.campaign_20261102.campaign_value_current` |
| `scripts/check_matched_calibration.py` | `scripts.archive.campaign_20261102.check_matched_calibration` |
| `scripts/estimate_campaign_pilot.py` | `scripts.archive.campaign_20261102.estimate_campaign_pilot` |
| `scripts/estimate_real12_pilot.py` | `scripts.archive.campaign_20261102.estimate_real12_pilot` |
| `scripts/export_campaign_plot_data.py` | `scripts.archive.campaign_20261102.export_campaign_plot_data` |
| `scripts/freeze_campaign_components.py` | `scripts.archive.campaign_20261102.freeze_campaign_components` |
| `scripts/freeze_campaign_policy.py` | `scripts.archive.campaign_20261102.freeze_campaign_policy` |
| `scripts/freeze_small_pilot.py` | `scripts.archive.campaign_20261102.freeze_small_pilot` |
| `scripts/plot_calibration_v2.py` | `scripts.archive.campaign_20261102.plot_calibration_v2` |
| `scripts/plot_campaign_components.py` | `scripts.archive.campaign_20261102.plot_campaign_components` |
| `scripts/plot_small_pilot_progress.py` | `scripts.archive.campaign_20261102.plot_small_pilot_progress` |
| `scripts/recover_scale_case.py` | `scripts.archive.campaign_20261102.recover_scale_case` |
| `scripts/render_campaign_plot_data.py` | `scripts.archive.campaign_20261102.render_campaign_plot_data` |
| `scripts/report_calibration_v2.py` | `scripts.archive.campaign_20261102.report_calibration_v2` |
| `scripts/report_campaign_components.py` | `scripts.archive.campaign_20261102.report_campaign_components` |
| `scripts/report_campaign_data.py` | `scripts.archive.campaign_20261102.report_campaign_data` |
| `scripts/report_campaign_policy.py` | `scripts.archive.campaign_20261102.report_campaign_policy` |
| `scripts/resume_calibration_v2.py` | `scripts.archive.campaign_20261102.resume_calibration_v2` |
| `scripts/smoke_small_pilot_resume.py` | `scripts.archive.campaign_20261102.smoke_small_pilot_resume` |
| `scripts/tune_campaign_scales.py` | `scripts.archive.campaign_20261102.tune_campaign_scales` |
| `scripts/tune_real12_campaign_scales.py` | `scripts.archive.campaign_20261102.tune_real12_campaign_scales` |

Reusable helpers were extracted without changing their behavior:

| Helper | Current import |
| --- | --- |
| `CappedAPI` | `scripts.experiment_support.api` |
| `DurableTree`, `DurableEnvironment`, `verify_tree`, `verify_actions` | `scripts.experiment_support.trace` |

Current goal/generator probes use these helpers directly rather than depending on archived experiment entrypoints.
