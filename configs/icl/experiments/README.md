# In-context experiment configurations

`base.json` defines shared component settings. Other configurations can inherit
from it using `extends`; the loader validates fields and rejects inheritance
cycles. Matrix files list condition configurations for paired runs.

Inspect available commands with:

```bash
uv run --extra icl python -m scripts.run_icl_search --help
```

Use `preview` to inspect prepared prompts, `--mock` for offline component checks,
`--replay` for saved provider responses, and `--execute` to call the configured
endpoints. Preview works with prepared data and skips credentials, ledger and SDK calls.
Logical token limits use the local `o200k_base` vocabulary when available.
Without it the runner counts UTF-8 bytes conservatively and records the fallback
in the run metrics. This keeps replay usable offline without fetching a tokenizer.

Set `roles.policy`, `roles.generator` and `roles.value` to choose models independently.
Unspecified roles inherit `model` and the original per-role decoding settings.
Explicit roles declare `model`, `provider`, `api`, optional `revision`, `reasoning_effort`,
`response_format`, optional model `temperature`, and `capabilities`. The revision is
part of the cache and model identity. Use a provider's versioned model ID in
`model` to request a pinned deployment. The separate `policy_temperature` controls
categorical sampling of the action distribution in the simulator.

- [OpenAI Responses example](provider_openai.example.json) uses `OPENAI_API_KEY`.
- [Ministral example](provider_ministral.example.json) uses OpenAI for policy and value,
  and `ministral-8b-2512` via Mistral's compatible Chat Completions endpoint for proposals.
  The generator uses `MISTRAL_API_KEY` and JSON object mode. The compatible endpoint
  is in public preview. [Mistral Chat Completions documentation](https://docs.mistral.ai/inference/conversations/chat-completion)

Use `response_format: "text"` for prompt-enforced JSON when model-side structured
output is unavailable. The component validates the decoded output. `json_max_retries`
permits at most one additional billed attempt with the original wire request and a
separate cache slot. Configuration rejects unsupported features before dispatch.
Changing the controller model, endpoint, revision, prompt or decoding requires
matching calibration. Hybrid search accepts `roles.generator` and uses its existing
local checkpoint settings for policy and value.

For live work, configure a local campaign ledger using
`configs/icl/campaign_budget.example.json`. The example has a zero spending limit.
Price each new route under `routes["provider/endpoint/model"]` with
`input_usd_per_million` and `max_input_tokens`. Chat Completions and Responses also
need `output_usd_per_million` and `max_output_tokens`. Existing `models` prices
remain scoped to OpenAI Responses. OpenAI Embeddings require their own price
route. Reuse the ledger and set `CARL_CAMPAIGN_STAGE` to a configured allocation.

```bash
uv run --extra icl python -m scripts.run_icl_search preview \
  --config configs/icl/experiments/provider_ministral.example.json
uv run --extra icl python -m scripts.run_icl_search run --replay \
  --config configs/icl/experiments/provider_openai.example.json
```
