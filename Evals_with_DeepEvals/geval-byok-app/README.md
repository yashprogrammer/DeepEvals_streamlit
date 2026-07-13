# DeepEval Course Companion — a bring-your-own-key (BYOK) Streamlit app

A hands-on companion to the course notebooks (`01_rag_evals_with_deepeval.ipynb`,
`02_agent_evals_with_deepeval.ipynb`, `03_chatbot_conversation_evals_with_deepeval.ipynb`,
`06_g_eval_with_deepeval.ipynb`). Learners paste their own Groq key(s) into the sidebar; once
it validates, a topic dropdown picks which section renders on the right:

- **Types of Metrics** — G-Eval, DAG, QAG. Define a metric via plain-English `criteria` or an
  `evaluation_steps` checklist (G-Eval), a rule-based decision tree — the official docs example:
  extract a summary's headings, check they're all present, then check they're in order (DAG), or
  claim-by-claim yes/no/idk questioning with the score computed mathematically instead of guessed
  by the LLM (QAG, via `FaithfulnessMetric`).
- **RAG Evals** — a 3-tab workflow mirroring a real RAG eval pipeline: **Ground Truths** (the
  knowledge base + a curated question/expected-answer dataset, extendable in the UI) →
  **RAG Pipeline** (the naive retriever + a Groq-generated answer, grounded in what was
  retrieved) → **Evaluation** (5 RAG metrics — Faithfulness, Answer Relevancy, Contextual
  Relevancy/Precision/Recall — scoring the retriever and the generator separately).
- **Agent Evals** — a 3-tab workflow: **Ground Truths** (the agent's tools + goal, plus a
  curated set of scenarios — mostly single-turn, one multi-turn — extendable in the UI) →
  **Interact** (a real chat interface: replay a scenario, auto-run all of them, or chat with
  the agent directly, with every tool call shown live) → **Evaluation** (batch-scores every
  saved run against up to 6 agent metrics — Tool Correctness, Argument Correctness, Step
  Efficiency, Plan Adherence, Plan Quality, Task Completion).
- **Multiturn Evals** — an editable hand-written conversation (with a bonus button to generate
  the next reply live via Groq), scored with the full 7-metric multi-turn suite: Role Adherence,
  Conversation Completeness, Turn Relevancy, Knowledge Retention, Goal Accuracy, Topic Adherence,
  and a Conversational G-Eval.

Every section includes a Mermaid diagram of its own pipeline plus a full breakdown table
(metric, verdict, score, reason) for explainability. Deployable as one public Streamlit
Community Cloud link — no author keys shipped.

## Keys — and why there are two

Groq plays both roles in this app: it's the **judge** for every metric (via DeepEval's
`LocalModel` pointed at Groq's OpenAI-compatible endpoint) and the **model under test** that
generates the RAG answer / agent trace / bonus conversation turn — but *not the same model* for
both, and not even the same judge model for every metric. Generation uses
`llama-3.3-70b-versatile` (30 RPM / 12K TPM). Most judge calls use
`meta-llama/llama-4-scout-17b-16e-instruct` (30 RPM / **30K TPM**) instead: several agent metrics
(Step Efficiency, Plan Adherence, Plan Quality, Task Completion) each make 2-3 internal LLM calls
apiece with verbose prompts, so a single fully-scored agent run is realistically ~12-13 judge
calls, not 6 — Scout's extra TPM headroom matters a lot there. **Argument Correctness is the one
exception** — it's judged with `llama-3.3-70b-versatile` instead. Its prompt is the most complex
one any metric here sends (a multi-tool-call few-shot example in raw Python-repr syntax, dense
with nested braces, right before a strict JSON-only instruction), and DeepEval extracts the
judge's JSON with a naive first-`{`-to-last-`}` scan rather than a real parser — so a smaller
model adding any stray commentary around its answer breaks extraction (`DeepEvalError:
Evaluation LLM outputted an invalid JSON`). Argument Correctness only makes 1-2 calls per run,
so TPM isn't the constraint there that it is for the trace-based metrics; reliability wins.
Groq's free-tier rate limits are applied **per account, not per key** (see
[console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)), so judge calls
and generation calls sharing one key share one limit regardless of which model each is calling.

| Key | Used for | Required? | Get it |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | judge (always) + generation (if no 2nd key) | Yes | https://console.groq.com/keys |
| `GROQ_API_KEY_2` | generation only — from a **different** Groq account | Optional | https://console.groq.com/keys |

Add a second key from a different Groq account to split judge and generation across two
separate rate-limit pools — roughly double the headroom before you'd see a 429. With one key,
everything shares that account's limits.

## Run locally

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # optional: fill real keys
./.venv/bin/streamlit run app.py
```

You can skip `secrets.toml` and just paste keys into the sidebar.

## Deploy to Streamlit Community Cloud

1. Push to GitHub.
2. Create an app on https://share.streamlit.io pointing at this repo/branch.
3. Set **Main file path** to `geval-byok-app/app.py`.
4. (Optional) Paste keys into **Settings → Secrets**; learners can instead use their own.

## Rate limits

Every run is scoped to one test case — no batching or fan-out. Most sections stay comfortably
within budget on a single key; Agent Evals' batch evaluation is the one that can genuinely add
up, since each of its 4 trace-based metrics does 2-3 internal LLM calls, not 1:

| Section | Calls per click |
| --- | --- |
| G-Eval | 1 judge call |
| DAG | ≤3 judge calls (extract, gate, then the order check — skipped if the gate fails) |
| QAG | a handful of judge calls (truths, claims, verdicts) |
| RAG Evals | 1 generation call (Pipeline tab) + ≤5 judge calls (Evaluation tab) — two separate clicks |
| Agent Evals | ~2 generation calls per turn (Interact tab); **~12-13 judge calls per saved run** (Tool/Argument Correctness are 0-2 calls each; Step Efficiency 2, Plan Adherence 3, Plan Quality 3, Task Completion 2), all at once per click in the Evaluation tab — 4 runs batched is ~50 calls |
| Multiturn Evals | 7 judge calls (+ 1 optional generation call for the bonus reply button) |

Two things soften this: the judge runs on `meta-llama/llama-4-scout-17b-16e-instruct` specifically
for its 30K TPM headroom (see "Keys" above), and `_measure_metrics`/`run_agent_batch_evaluation`
add small pacing gaps between calls so a batch spreads across more of Groq's rolling per-minute
window instead of bursting into one. DeepEval's `LocalModel` and Groq's own SDK also both retry
transient 429s with backoff automatically. None of that raises the total token budget, though —
for a big batch (auto-run all scenarios, then evaluate all of them), a second key still helps
most, since it moves generation calls to a separate account's pool entirely.

## Layout

```
app.py               sidebar (BYOK + topic dropdown) + 4 section renderers
core/keys.py           provider registry (GROQ_API_KEY required, GROQ_API_KEY_2 optional), resolution, validation, gating
core/config.py          Groq model names (generation vs judge), base URL, pacing, timeout
core/demo_data.py       bundled RAG corpus/retriever, agent tools, default multiturn conversation
core/providers.py       run_geval/run_dag/run_qag/run_rag_pipeline/run_rag_evaluation/run_agent_turn/run_agent_scenario/run_agent_evaluation/run_agent_batch_evaluation/build_agent_trace/run_multiturn_eval/generate_next_reply
ui/async_utils.py       worker-thread runner (event-loop-safe)
ui/tabs.py              render_mermaid() — pipeline diagrams via mermaid.js (CDN, no extra dep)
```

Torch-free — everything is API-based (Groq's SDK + DeepEval's `LocalModel`), well within Cloud's
1 GB limit.
