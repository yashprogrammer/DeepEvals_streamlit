"""core/config.py — model names and tunable constants in one place."""

# Generation ("model under test") AND the default judge for every metric except the 4 trace-
# based agent metrics — llama-3.3-70b-versatile, proven reliable for both generating and closely
# following structured-output/JSON instructions across this app's prompts. 30 RPM / 12K TPM per
# Groq account. See core/providers.py::_build_judge for why this is the *default* judge, not
# GROQ_JUDGE_MODEL: smaller judge models have shown real, reproducible schema/JSON failures on
# this app's more complex prompts (Argument Correctness, DAG's TaskNode).
GROQ_MODEL = "llama-3.3-70b-versatile"

# Judge for the 4 trace-based agent metrics ONLY (Step Efficiency, Plan Adherence, Plan Quality,
# Task Completion) — see core/providers.py::_build_trace_judge. Each of those metrics makes 2-3
# internal LLM calls apiece (extract, then score) with verbose prompt templates, so a single
# fully-evaluated agent run is more like 12-13 real judge calls, not 6 — GROQ_MODEL's 12K TPM cap
# gets exhausted fast across a batch of several runs. This used to point at
# meta-llama/llama-4-scout-17b-16e-instruct (30K TPM, 2.5x headroom) specifically for that reason,
# but Groq decommissioned that model (its API calls now 404 with model_not_found). As of
# console.groq.com/docs/rate-limits, no other free-tier model has meaningfully more TPM than
# GROQ_MODEL's 12K without also losing structured-output reliability (see _build_judge below), so
# both judge roles share the same model until Groq ships a better-suited free-tier option.
GROQ_JUDGE_MODEL = GROQ_MODEL

# Groq's API is OpenAI-compatible; DeepEval's LocalModel talks to any OpenAI-compatible endpoint.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Short pacing gap between sequential judge calls within one batch (seconds). Doesn't reduce
# total token usage, but spreads it across more of the rolling per-minute window instead of
# bursting, so a multi-metric/multi-run batch is less likely to blow through TPM in one go.
JUDGE_CALL_PACING_SECONDS = 0.5

# Worker-thread timeout for a single demo run (seconds).
WORK_TIMEOUT = 120
