"""core/config.py — model names and tunable constants in one place."""

# Generation ("model under test") — llama-3.3-70b-versatile, proven reliable for the short
# RAG/Agent/Multiturn prompts this app sends. 30 RPM / 12K TPM per Groq account.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Judge — deliberately a different model from generation. The 4 trace-based agent metrics
# (Step Efficiency, Plan Adherence, Plan Quality, Task Completion) each make 2-3 internal LLM
# calls apiece (extract, then score) with verbose prompt templates, so a single fully-evaluated
# agent run is more like 12-13 real judge calls, not 6 — llama-3.3-70b-versatile's 12K TPM cap
# gets exhausted fast across a batch of several runs. meta-llama/llama-4-scout-17b-16e-instruct
# has 30K TPM (2.5x) for the same 1K RPD, at the same free tier, giving much more headroom for
# exactly this workload — see console.groq.com/docs/rate-limits.
GROQ_JUDGE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Groq's API is OpenAI-compatible; DeepEval's LocalModel talks to any OpenAI-compatible endpoint.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Short pacing gap between sequential judge calls within one batch (seconds). Doesn't reduce
# total token usage, but spreads it across more of the rolling per-minute window instead of
# bursting, so a multi-metric/multi-run batch is less likely to blow through TPM in one go.
JUDGE_CALL_PACING_SECONDS = 0.5

# Worker-thread timeout for a single demo run (seconds).
WORK_TIMEOUT = 120
