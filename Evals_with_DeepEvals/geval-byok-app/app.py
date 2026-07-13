"""DeepEval Course Companion — a bring-your-own-key (BYOK) Streamlit app.

A course companion covering four DeepEval topics, picked from the sidebar dropdown:
- Types of Metrics — G-Eval, DAG, QAG (companion to 06_g_eval_with_deepeval.ipynb).
- RAG Evals — companion to 01_rag_evals_with_deepeval.ipynb.
- Agent Evals — companion to 02_agent_evals_with_deepeval.ipynb.
- Multiturn Evals — companion to 03_chatbot_conversation_evals_with_deepeval.ipynb.

Groq plays both roles: the judge for every metric, and the "model under test" for
RAG/Agent/Multiturn generation. A second, optional Groq key (a different account) splits
those two roles across two separate rate-limit pools instead of one — see core/keys.py.
Every run is one test case — no fan-out — so a single click stays well under free-tier
rate limits (console.groq.com/docs/rate-limits).

Run:  streamlit run app.py   (from this folder)
"""
from __future__ import annotations

import streamlit as st

from core import demo_data
from core import keys as keymod
from core.config import WORK_TIMEOUT
from core.providers import (
    build_agent_trace,
    generate_next_reply,
    run_agent_batch_evaluation,
    run_agent_scenario,
    run_agent_turn,
    run_dag,
    run_geval,
    run_multiturn_eval,
    run_qag,
    run_rag_evaluation,
    run_rag_pipeline,
)
from ui.async_utils import run_in_thread
from ui.tabs import render_mermaid

APP_TITLE = "🧑‍⚖️ DeepEval Course Companion"
APP_TAGLINE = "Bring your own Groq key(s) and pick a topic from the sidebar."

SECTIONS = ["Types of Metrics", "RAG Evals", "Agent Evals", "Multiturn Evals"]

st.set_page_config(page_title=APP_TITLE, page_icon="🧑‍⚖️", layout="wide")


# ──────────────────────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("valid", {k: False for k in keymod.PROVIDERS})
    ss.setdefault("geval_result", None)
    ss.setdefault("dag_result", None)
    ss.setdefault("qag_result", None)
    ss.setdefault("rag_ground_truths", [dict(q) for q in demo_data.RAG_EXAMPLE_QUESTIONS])
    ss.setdefault("rag_pipeline_result", None)
    ss.setdefault("rag_eval_result", None)
    ss.setdefault("agent_scenarios", [dict(s) for s in demo_data.AGENT_SCENARIOS])
    ss.setdefault("agent_chat_display", [])
    ss.setdefault("agent_chat_groq_messages",
                   [{"role": "system", "content": demo_data.AGENT_SYSTEM_PROMPT}])
    ss.setdefault("agent_chat_trace", [])
    ss.setdefault("agent_runs", [])
    ss.setdefault("agent_eval_results", None)
    ss.setdefault("multiturn_result", None)
    ss.setdefault("mt_transcript", "\n".join(f"{r}: {c}" for r, c in demo_data.DEFAULT_CONVERSATION))


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — registry-driven keys + validation + status + topic picker
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    ss = st.session_state
    st.sidebar.title("🔑 API keys")
    st.sidebar.caption("Bring your own keys — they stay in this browser session only.")
    overrides = {}
    for name, p in keymod.PROVIDERS.items():
        overrides[name] = st.sidebar.text_input(
            p.label, type="password", value=keymod.get_secret(name) or "",
            help=p.help_url + ("" if p.required else
                                "  (optional — a 2nd Groq account for more rate-limit headroom)"))
    keymod.apply_keys(overrides)

    if st.sidebar.button("Validate keys", type="primary", use_container_width=True):
        valid = {}
        with st.sidebar, st.spinner("Validating…"):
            for name, p in keymod.PROVIDERS.items():
                ok, msg = p.validate(keymod.get_secret(name) or None)
                valid[name] = ok
                st.write(f"{'✅' if ok else '❌'} {p.label} — {msg}")
        ss["valid"] = valid

    st.sidebar.divider()
    st.sidebar.markdown("**Status**")
    for name, p in keymod.PROVIDERS.items():
        mark = "✅" if ss["valid"].get(name) else ("⬜" if p.required else "▫️")
        st.sidebar.write(f"{mark} {p.label}")

    if not keymod.all_required_valid(ss["valid"]):
        st.sidebar.info("Add your Groq key and click **Validate** to unlock the app.")

    st.sidebar.divider()
    return st.sidebar.selectbox("Choose a topic", SECTIONS, key="section")


def unlocked() -> bool:
    return keymod.all_required_valid(st.session_state["valid"])


# ──────────────────────────────────────────────────────────────────────────────
# Shared result display
# ──────────────────────────────────────────────────────────────────────────────
def render_metric_breakdown(breakdown: list[dict]):
    rows = []
    for row in breakdown:
        verdict = "✅ PASS" if row["success"] else ("⚠️ ERROR" if row["score"] is None else "❌ FAIL")
        rows.append({
            "Metric": row["metric"],
            "Verdict": verdict,
            "Score": f"{row['score']:.2f}" if row["score"] is not None else "—",
            "Reason": row["reason"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — G-Eval
# ──────────────────────────────────────────────────────────────────────────────
def render_geval_tab():
    st.caption(
        "G-Eval grades subjective, open-ended criteria — correctness, tone, coherence, safety — "
        "with one LLM judge call guided by chain-of-thought reasoning.")

    st.subheader("1️⃣ The test case")
    col1, col2 = st.columns(2)
    with col1:
        input_text = st.text_area("Input (the question/prompt)",
                                   value="What's your refund policy?", height=100, key="ge_input")
        actual_output = st.text_area(
            "Actual output (the answer to grade)",
            value="Happy to help! We offer full refunds within 30 days of purchase, no "
                  "questions asked. Just reply here with your order number.", height=100,
            key="ge_actual")
    with col2:
        expected_output = st.text_area(
            "Expected output (optional — needed if your criteria compares against a reference)",
            value="Refunds are available within 30 days of purchase.", height=100, key="ge_expected")

    st.subheader("2️⃣ The metric")
    st.caption("Describe what to check in plain English, or write the checklist yourself.")
    metric_name = st.text_input("Metric name", value="Correctness", key="ge_name")
    mode = st.radio("Define it with",
                     ["criteria (auto CoT steps)", "evaluation_steps (your own checklist)"],
                     horizontal=True, key="ge_mode")

    criteria, evaluation_steps = None, None
    if mode.startswith("criteria"):
        criteria = st.text_area(
            "Criteria",
            value="Determine whether the actual output is factually correct given the "
                  "expected output.", height=80, key="ge_criteria")
    else:
        steps_text = st.text_area(
            "Evaluation steps (one per line)",
            value="Check whether the response is polite and professional.\n"
                  "Penalize sarcasm, rudeness, or dismissive language.\n"
                  "Reward clear, respectful phrasing even if brief.", height=100, key="ge_steps")
        evaluation_steps = [s.strip() for s in steps_text.splitlines() if s.strip()]

    st.subheader("3️⃣ Run")
    if st.button("Run G-Eval", type="primary", key="ge_run"):
        with st.spinner("Judging with Groq…"):
            try:
                result = run_in_thread(
                    run_geval, input_text, actual_output, expected_output,
                    metric_name, criteria, evaluation_steps, timeout=WORK_TIMEOUT)
                st.session_state["geval_result"] = result
            except Exception as e:  # surface errors — never hide them in a BYOK app
                st.session_state["geval_result"] = {"error": f"{type(e).__name__}: {e}"}

    result = st.session_state.get("geval_result")
    if result:
        st.divider()
        if "error" in result:
            st.error(result["error"])
        elif result["score"] is None:
            st.error(f"Judge call failed: {result['reason']}")
        else:
            verdict = "✅ PASS" if result["success"] else "❌ FAIL"
            st.metric(f"{metric_name} — {verdict}", f"{result['score']:.2f}")
            st.write("**Reason:**", result["reason"])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — DAG
# ──────────────────────────────────────────────────────────────────────────────
def render_dag_tab():
    st.caption(
        "DAG (Deep Acyclic Graph) builds a small decision tree instead of one big prompt — good "
        "for objective, rule-based checks. This is the official docs example "
        "(deepeval.com/docs/metrics-dag): check whether a meeting-summary has the right headings, "
        "and — only if it does — whether they're in the right order. DAG can also embed a "
        "subjective G-Eval node as a child of a Verdict Node, mixing rigid checks with judgment "
        "calls, though this particular example doesn't need one.")
    st.markdown(
        "| Node type | Role |\n"
        "| --- | --- |\n"
        "| Task | Preprocess/extract data from the test case before evaluation (used below to "
        "extract headings) |\n"
        "| Binary Judgement | Strict yes/no check (used below as the gate) |\n"
        "| Non-Binary Judgement | Nuanced scoring — more than yes/no (used below for ordering) |\n"
        "| Verdict | Leaf node — assigns the score, or hands off to a child node/metric |\n")

    render_mermaid("""
        graph TD
            TC[Test case] --> Task[Task Node - extract headings]
            Task --> Gate[Binary Judgement Node - has all three headings]
            Task --> Order[Non-Binary Judgement Node - correct order]
            Gate -->|False| V0[Verdict Node - score 0]
            Gate -->|True| Order
            Order -->|Yes| V10[Verdict Node - score 10]
            Order -->|Two out of order| V4[Verdict Node - score 4]
            Order -->|All out of order| V2[Verdict Node - score 2]
    """, height=560)

    st.subheader("1️⃣ The test case")
    input_text = st.text_area(
        "Input (the meeting transcript)",
        value='Alice: "Today\'s agenda: product update, blockers, and marketing timeline. Bob, '
              'updates?"\nBob: "Core features are done, but we\'re optimizing performance for '
              'large datasets. Fixes by Friday, testing next week."\nAlice: "Charlie, does this '
              'timeline work for marketing?"\nCharlie: "We need finalized messaging by Monday."\n'
              'Alice: "Bob, can we provide a stable version by then?"\nBob: "Yes, we\'ll share an '
              'early build."\nCharlie: "Great, we\'ll start preparing assets."\nAlice: "Plan: '
              'fixes by Friday, marketing prep Monday, sync next Wednesday. Thanks, everyone!"',
        height=180, key="dag_input")
    actual_output = st.text_area(
        "Actual output (the summary to grade)",
        value="Intro:\nAlice outlined the agenda: product updates, blockers, and marketing "
              "alignment.\n\nBody:\nBob reported performance issues being optimized, with fixes "
              "expected by Friday. Charlie requested finalized messaging by Monday for marketing "
              "preparation. Bob confirmed an early stable build would be ready.\n\nConclusion:\n"
              "The team aligned on next steps: engineering finalizing fixes, marketing preparing "
              "content, and a follow-up sync scheduled for Wednesday.", height=180, key="dag_actual")

    st.subheader("2️⃣ The Task Node")
    st.caption("Preprocesses the output before any judgement — here, pulling out the headings.")
    task_instructions = st.text_area(
        "Task instructions",
        value="Extract all headings in `actual_output`", height=60, key="dag_task")

    st.subheader("3️⃣ The gate (Binary Judgement Node)")
    st.caption("A strict yes/no check. If it fails, the metric scores 0 and the order check "
               "never runs.")
    heading_criteria = st.text_area(
        "Gate criteria",
        value="Does the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
        height=60, key="dag_gate")

    st.subheader("4️⃣ The ordering check (Non-Binary Judgement Node)")
    st.caption("Only runs if the gate passes. Scored 10 / 4 / 2 depending on how out of order.")
    order_criteria = st.text_area(
        "Order criteria",
        value="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
        height=60, key="dag_order")

    st.subheader("5️⃣ Run")
    if st.button("Run DAG", type="primary", key="dag_run"):
        with st.spinner("Walking the graph with Groq…"):
            try:
                result = run_in_thread(
                    run_dag, input_text, actual_output, task_instructions, heading_criteria,
                    order_criteria, timeout=WORK_TIMEOUT)
                st.session_state["dag_result"] = result
            except Exception as e:  # surface errors — never hide them in a BYOK app
                st.session_state["dag_result"] = {"error": f"{type(e).__name__}: {e}"}

    result = st.session_state.get("dag_result")
    if result:
        st.divider()
        if "error" in result:
            st.error(result["error"])
        elif result["score"] is None:
            st.error(f"Judge call failed: {result['reason']}")
        else:
            verdict = "✅ PASS" if result["success"] else "❌ FAIL"
            st.metric(f"Format Correctness — {verdict}", f"{result['score']:.0f} / 10")
            st.caption("Verdict Node scores here follow the docs example's own scale (0/2/4/10, "
                       "not 0-1) — the default threshold (0.5) still treats any nonzero score as "
                       "a pass.")
            st.write("**Reason:**", result["reason"])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — QAG
# ──────────────────────────────────────────────────────────────────────────────
def render_qag_tab():
    st.caption(
        "QAG (Question-Answer Generation) never lets the LLM pick a score. It extracts claims "
        "from the output, asks a closed **yes / no / idk** question per claim against the "
        "context, then computes the score mathematically. This is what powers DeepEval's RAG "
        "metrics — shown below via **Faithfulness**.")

    render_mermaid("""
        graph TD
            AO[Actual output] --> C[Claim extraction]
            RC[Retrieval context] --> Q[Closed yes or no or idk question per claim]
            C --> Q
            Q --> M[Mathematical computation]
            M --> S[Score is truthful claims divided by total claims]
    """, height=560)

    st.subheader("1️⃣ The test case")
    input_text = st.text_area("Input (the question)",
                               value="When was the Eiffel Tower built and how tall is it?",
                               height=80, key="qag_input")
    actual_output = st.text_area(
        "Actual output (the answer to check for hallucinations)",
        value="The Eiffel Tower was completed in 1889 and stands 330 meters tall. It was "
              "designed by Gustave Eiffel and is made entirely of marble.", height=100,
        key="qag_actual")
    context_text = st.text_area(
        "Retrieval context (one chunk per line — the ground truth to check claims against)",
        value="The Eiffel Tower was completed in 1889 for the World's Fair.\n"
              "It stands 330 meters tall and was designed by Gustave Eiffel's engineering company.\n"
              "The tower is built primarily of wrought iron.", height=100, key="qag_context")
    retrieval_context = [c.strip() for c in context_text.splitlines() if c.strip()]

    st.subheader("2️⃣ Run")
    if st.button("Run QAG (Faithfulness)", type="primary", key="qag_run"):
        with st.spinner("Extracting claims and questioning Groq…"):
            try:
                result = run_in_thread(
                    run_qag, input_text, actual_output, retrieval_context, timeout=WORK_TIMEOUT)
                st.session_state["qag_result"] = result
            except Exception as e:  # surface errors — never hide them in a BYOK app
                st.session_state["qag_result"] = {"error": f"{type(e).__name__}: {e}"}

    result = st.session_state.get("qag_result")
    if result:
        st.divider()
        if "error" in result:
            st.error(result["error"])
        elif result["score"] is None:
            st.error(f"Judge call failed: {result['reason']}")
        else:
            verdict = "✅ PASS" if result["success"] else "❌ FAIL"
            st.metric(f"Faithfulness — {verdict}", f"{result['score']:.2f}")
            st.write("**Reason:**", result["reason"])
            if result["breakdown"]:
                st.caption("Claim-by-claim breakdown:")
                st.dataframe(result["breakdown"], use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Section: Types of Metrics (existing G-Eval / DAG / QAG tabs, unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def render_types_of_metrics_section():
    geval_tab, dag_tab, qag_tab = st.tabs(["🧑‍⚖️ G-Eval", "🌳 DAG", "❓ QAG"])
    with geval_tab:
        render_geval_tab()
    with dag_tab:
        render_dag_tab()
    with qag_tab:
        render_qag_tab()


# ──────────────────────────────────────────────────────────────────────────────
# Section: RAG Evals — Ground Truths → RAG Pipeline → Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def render_rag_ground_truths_tab():
    st.caption(
        "A RAG eval needs a **golden dataset**: questions paired with the knowledge that should "
        "answer them, and — where possible — the expected answer. This is the knowledge base "
        "and a small hand-curated set of ground-truth questions the pipeline tab runs against.")

    st.subheader("📚 The knowledge base")
    for doc in demo_data.RAG_CORPUS:
        st.markdown(f"**{doc['title']}** — {doc['content']}")

    st.subheader("✅ Ground-truth questions")
    gts = st.session_state["rag_ground_truths"]
    st.dataframe(
        [{"Question": g["input"], "Expected output": g["expected_output"] or "(none — outside the KB)"}
         for g in gts],
        use_container_width=True, hide_index=True)

    with st.expander("➕ Add a custom ground truth"):
        new_q = st.text_input("Question", key="rag_new_q")
        new_a = st.text_input("Expected output (optional — leave blank to also test the "
                               "out-of-KB case)", key="rag_new_a")
        if st.button("Add", key="rag_add_gt"):
            if new_q.strip():
                st.session_state["rag_ground_truths"].append(
                    {"input": new_q.strip(), "expected_output": new_a.strip()})
                st.success("Added — pick it in the RAG Pipeline tab.")
            else:
                st.warning("Question can't be empty.")


def render_rag_pipeline_tab():
    st.caption(
        "Pick a ground-truth question (or write your own), then run it through the pipeline: "
        "the naive retriever pulls context from the knowledge base, and Groq generates an "
        "answer grounded in that context.")

    render_mermaid("""
        graph TD
            Q[Question] --> R[Naive keyword retriever]
            R --> CTX[Retrieved context]
            CTX --> G[Groq generates the answer]
            G --> AO[Actual output]
    """, height=560)

    gts = st.session_state["rag_ground_truths"]
    labels = [g["input"] for g in gts] + ["Write your own..."]
    choice = st.selectbox("Question", labels, key="rag_pipeline_choice")

    if choice == "Write your own...":
        question = st.text_input("Your question", key="rag_pipeline_custom_q")
        expected_output = st.text_input(
            "Expected output (optional — unlocks Contextual Precision/Recall in Evaluation)",
            key="rag_pipeline_custom_a")
    else:
        gt = next(g for g in gts if g["input"] == choice)
        question = choice
        expected_output = gt["expected_output"]
        st.caption(f"Expected output: {expected_output or '_(none — outside the KB)_'}")

    if st.button("Run RAG Pipeline", type="primary", key="rag_pipeline_run"):
        with st.spinner("Retrieving and generating with Groq…"):
            try:
                result = run_in_thread(run_rag_pipeline, question, timeout=WORK_TIMEOUT)
                result["question"] = question
                result["expected_output"] = expected_output
                st.session_state["rag_pipeline_result"] = result
                st.session_state["rag_eval_result"] = None  # stale eval — clear it
            except Exception as e:  # surface errors — never hide them in a BYOK app
                st.session_state["rag_pipeline_result"] = {"error": f"{type(e).__name__}: {e}"}

    result = st.session_state.get("rag_pipeline_result")
    if result:
        st.divider()
        if "error" in result:
            st.error(result["error"])
        else:
            st.caption("Retrieved context:")
            for chunk in result["retrieved_context"]:
                st.markdown(f"> {chunk}")
            st.write("**Actual output (Groq):**", result["actual_output"])
            st.info("👉 Head to the **Evaluation** tab to score this with RAG metrics.")


def render_rag_evaluation_tab():
    st.caption(
        "Scores the last RAG Pipeline run with 5 DeepEval RAG metrics. Faithfulness and Answer "
        "Relevancy check the **generator**; Contextual Relevancy/Precision/Recall check the "
        "**retriever** — so a low score tells you which stage to fix.")

    render_mermaid("""
        graph TD
            CTX[Retrieved context] --> M[5 RAG metrics]
            AO[Actual output] --> M
            EO[Expected output - optional] --> M
            M --> S[Faithfulness, Answer Relevancy, Contextual Relevancy, Precision, Recall]
    """, height=460)

    pipeline_result = st.session_state.get("rag_pipeline_result")
    if not pipeline_result or "error" in pipeline_result:
        st.info("Run the **RAG Pipeline** tab first to produce something to evaluate.")
        return

    st.write("**Question:**", pipeline_result["question"])
    st.write("**Actual output:**", pipeline_result["actual_output"])

    if st.button("Run Evaluation", type="primary", key="rag_eval_run"):
        with st.spinner("Judging with Groq…"):
            try:
                result = run_in_thread(
                    run_rag_evaluation, pipeline_result["question"], pipeline_result["actual_output"],
                    pipeline_result["expected_output"], pipeline_result["retrieved_context"],
                    timeout=WORK_TIMEOUT)
                st.session_state["rag_eval_result"] = result
            except Exception as e:  # surface errors — never hide them in a BYOK app
                st.session_state["rag_eval_result"] = {"error": f"{type(e).__name__}: {e}"}

    result = st.session_state.get("rag_eval_result")
    if result:
        st.divider()
        if "error" in result:
            st.error(result["error"])
        else:
            st.caption("Metric breakdown:")
            render_metric_breakdown(result["breakdown"])


def render_rag_section():
    st.caption(
        "A full RAG eval workflow: curate ground truths from a knowledge base, run the "
        "retrieval + generation pipeline, then score the results with RAG metrics.")

    gt_tab, pipeline_tab, eval_tab = st.tabs(
        ["1️⃣ Ground Truths", "2️⃣ RAG Pipeline", "3️⃣ Evaluation"])
    with gt_tab:
        render_rag_ground_truths_tab()
    with pipeline_tab:
        render_rag_pipeline_tab()
    with eval_tab:
        render_rag_evaluation_tab()


# ──────────────────────────────────────────────────────────────────────────────
# Section: Agent Evals — Ground Truths → Interact → Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def render_agent_ground_truths_tab():
    st.caption(
        "The agent's tools and goal, and a curated set of ground-truth scenarios to run it "
        "against — most are single-turn, one is multi-turn.")

    st.subheader("🛠️ Available tools")
    for tool in demo_data.AGENT_TOOL_SCHEMAS:
        fn = tool["function"]
        st.markdown(f"**{fn['name']}** — {fn['description']}")

    st.subheader("🎯 Goal")
    st.write(demo_data.AGENT_GOAL)

    st.subheader("✅ Ground-truth scenarios")
    scenarios = st.session_state["agent_scenarios"]
    st.dataframe(
        [{"Scenario": s["name"], "Turns": " → ".join(s["turns"]),
          "Expected tools": ", ".join(s["expected_tools"]), "Goal": s["goal"]}
         for s in scenarios],
        use_container_width=True, hide_index=True)

    with st.expander("➕ Add a custom scenario"):
        name = st.text_input("Scenario name", key="agent_new_name")
        turns_text = st.text_area("Turns (one user message per line — multiple lines = "
                                   "multi-turn)", key="agent_new_turns")
        tools_text = st.text_input(
            "Expected tools (comma-separated, e.g. search_docs, run_code, escalate_to_human)",
            key="agent_new_tools")
        goal = st.text_input("Goal", key="agent_new_goal")
        if st.button("Add", key="agent_add_scenario"):
            turns = [t.strip() for t in turns_text.splitlines() if t.strip()]
            if name.strip() and turns:
                st.session_state["agent_scenarios"].append({
                    "name": name.strip(),
                    "turns": turns,
                    "expected_tools": [t.strip() for t in tools_text.split(",") if t.strip()],
                    "goal": goal.strip(),
                })
                st.success("Added — find it in the Interact tab.")
            else:
                st.warning("Need at least a name and one turn.")


def _save_agent_run(name: str, query: str, result: dict, expected_tools: list[str] | None,
                     goal: str | None):
    st.session_state["agent_runs"].append({
        "name": name,
        "query": query,
        "final_output": result["final_output"] if "final_output" in result else result["reply"],
        "tool_events": result["tool_events"],
        "trace": result.get("trace") or build_agent_trace(
            query, result.get("final_output") or result.get("reply", ""), result["tool_events"]),
        "expected_tools": expected_tools,
        "goal": goal,
    })


def render_agent_interact_tab():
    st.caption(
        "Replay a ground-truth scenario, auto-run all of them, or chat with the agent directly — "
        "every tool call is shown as it happens.")

    render_mermaid("""
        graph TD
            U[Your message] --> G1[Groq proposes tool calls]
            G1 --> T[Tools execute locally]
            T --> G2[Groq writes a reply]
            G2 --> R[Reply shown in chat]
            R -->|next turn| U
    """, height=560)

    with st.expander("🛠️ Available tools", expanded=False):
        for tool in demo_data.AGENT_TOOL_SCHEMAS:
            fn = tool["function"]
            st.markdown(f"**{fn['name']}** — {fn['description']}")

    st.subheader("🎬 Presets")
    scenarios = st.session_state["agent_scenarios"]
    labels = [s["name"] for s in scenarios]
    choice = st.selectbox("Pick a scenario to replay", labels, key="agent_preset_choice")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run this scenario", key="agent_run_preset"):
            scenario = next(s for s in scenarios if s["name"] == choice)
            with st.spinner("Running the agent through the scenario…"):
                try:
                    result = run_in_thread(run_agent_scenario, scenario["turns"],
                                            timeout=WORK_TIMEOUT)
                    _save_agent_run(scenario["name"], scenario["turns"][0], result,
                                     scenario["expected_tools"], scenario["goal"])
                    st.session_state["agent_chat_display"] = list(result["transcript"])
                    st.session_state["agent_chat_trace"] = list(result["tool_events"])
                    st.success(f"Ran \"{scenario['name']}\" — saved for evaluation.")
                except Exception as e:  # surface errors — never hide them in a BYOK app
                    st.error(f"{type(e).__name__}: {e}")
    with col2:
        if st.button(f"⏩ Auto-run ALL {len(scenarios)} scenarios", key="agent_run_all"):
            with st.spinner(f"Running the agent through all {len(scenarios)} scenarios…"):
                try:
                    for scenario in scenarios:
                        result = run_in_thread(run_agent_scenario, scenario["turns"],
                                                timeout=WORK_TIMEOUT)
                        _save_agent_run(scenario["name"], scenario["turns"][0], result,
                                         scenario["expected_tools"], scenario["goal"])
                    st.success(f"Ran all {len(scenarios)} scenarios — see the Evaluation tab.")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

    st.divider()
    st.subheader("💬 Or chat with the agent directly")

    for role, content in st.session_state["agent_chat_display"]:
        with st.chat_message(role):
            st.write(content)

    user_msg = st.chat_input("Ask the agent something…")
    if user_msg:
        with st.spinner("Thinking…"):
            try:
                result = run_in_thread(
                    run_agent_turn, st.session_state["agent_chat_groq_messages"], user_msg,
                    timeout=WORK_TIMEOUT)
                st.session_state["agent_chat_groq_messages"] = result["messages"]
                st.session_state["agent_chat_display"].append(("user", user_msg))
                st.session_state["agent_chat_display"].append(("assistant", result["reply"]))
                st.session_state["agent_chat_trace"].extend(result["tool_events"])
                st.rerun()
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

    if st.session_state["agent_chat_trace"]:
        st.caption("Tool-call trace for this conversation:")
        st.dataframe(st.session_state["agent_chat_trace"], use_container_width=True,
                      hide_index=True)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("💾 Save this conversation as a run", key="agent_save_chat"):
            display = st.session_state["agent_chat_display"]
            if display:
                query = display[0][1]
                final_output = display[-1][1]
                run_name = f"Manual chat #{len(st.session_state['agent_runs']) + 1}"
                _save_agent_run(run_name, query,
                                 {"final_output": final_output,
                                  "tool_events": st.session_state["agent_chat_trace"]},
                                 expected_tools=None, goal=None)
                st.success(f"Saved as \"{run_name}\" — find it in the Evaluation tab.")
            else:
                st.warning("Nothing to save yet — send a message first.")
    with col4:
        if st.button("🔄 Reset conversation", key="agent_reset_chat"):
            st.session_state["agent_chat_display"] = []
            st.session_state["agent_chat_groq_messages"] = [
                {"role": "system", "content": demo_data.AGENT_SYSTEM_PROMPT}]
            st.session_state["agent_chat_trace"] = []
            st.rerun()


def render_agent_evaluation_tab():
    st.caption(
        "Scores every saved run — scenario replays and any manually saved chats — against up "
        "to 6 agent metrics: Tool Correctness and Argument Correctness (only for runs with "
        "expected tools), plus Step Efficiency, Plan Adherence, Plan Quality, and Task "
        "Completion (all built from the tool-call trace).")

    render_mermaid("""
        graph TD
            TR[Tool-call trace] --> M[Up to 6 agent metrics]
            EX[Expected tools - if any] --> M
            GO[Goal - if any] --> M
            M --> S[Tool Correctness, Argument Correctness, Step Efficiency, Plan Adherence, Plan Quality, Task Completion]
    """, height=420)

    runs = st.session_state["agent_runs"]
    if not runs:
        st.info("No runs yet — go to the **Interact** tab, run a preset scenario (or auto-run "
                 "all of them), or save a manual chat.")
        return

    st.write(f"**{len(runs)} run(s) ready to evaluate:**")
    st.dataframe(
        [{"Run": r["name"], "Query": r["query"], "Final output": r["final_output"]}
         for r in runs],
        use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        run_clicked = st.button("Evaluate all runs", type="primary", key="agent_eval_run")
    with col2:
        if st.button("🗑️ Clear all runs", key="agent_clear_runs"):
            st.session_state["agent_runs"] = []
            st.session_state["agent_eval_results"] = None
            st.rerun()

    if run_clicked:
        with st.spinner(f"Scoring {len(runs)} run(s) across up to 6 metrics each — this can "
                         f"take a while…"):
            try:
                results = run_in_thread(run_agent_batch_evaluation, runs,
                                         timeout=WORK_TIMEOUT * max(1, len(runs)))
                st.session_state["agent_eval_results"] = results
            except Exception as e:  # surface errors — never hide them in a BYOK app
                st.session_state["agent_eval_results"] = [{"name": "error", "error": str(e)}]

    results = st.session_state.get("agent_eval_results")
    if results:
        st.divider()
        for res in results:
            st.markdown(f"**{res['name']}**")
            if "error" in res:
                st.error(res["error"])
            else:
                render_metric_breakdown(res["breakdown"])


def render_agent_section():
    gt_tab, interact_tab, eval_tab = st.tabs(
        ["1️⃣ Ground Truths", "2️⃣ Interact", "3️⃣ Evaluation"])
    with gt_tab:
        render_agent_ground_truths_tab()
    with interact_tab:
        render_agent_interact_tab()
    with eval_tab:
        render_agent_evaluation_tab()


# ──────────────────────────────────────────────────────────────────────────────
# Section: Multiturn Evals
# ──────────────────────────────────────────────────────────────────────────────
def render_multiturn_section():
    st.caption(
        "A hand-written conversation, evaluated turn-by-turn and as a whole with the full "
        "7-metric multi-turn suite: adherence to role/topic/goal, completeness, relevancy, "
        "knowledge retention, and a subjective Conversational G-Eval.")

    render_mermaid("""
        graph TD
            T[Turns] --> CTC[ConversationalTestCase]
            CTC --> M[7 multi-turn metrics]
            M --> S[Role Adherence, Completeness, Turn Relevancy, Knowledge Retention, Goal Accuracy, Topic Adherence, Helpfulness]
    """, height=520)

    # This block must run BEFORE the "mt_transcript" text_area below is instantiated: Streamlit
    # forbids mutating st.session_state[key] after a widget with that key has already rendered
    # in the same script run.
    with st.expander("➕ Bonus: generate the next reply live via Groq"):
        next_user_msg = st.text_input("Your next message as the user", key="mt_next_user")
        if st.button("Get live Groq reply", key="mt_generate"):
            turns_data = _parse_transcript(st.session_state["mt_transcript"])
            with st.spinner("Asking Groq…"):
                try:
                    reply = run_in_thread(
                        generate_next_reply, turns_data, next_user_msg, timeout=WORK_TIMEOUT)
                    st.session_state["mt_transcript"] += f"\nuser: {next_user_msg}\nassistant: {reply}"
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

    st.subheader("1️⃣ The conversation")
    st.caption("One turn per line, as `role: content` (roles: user / assistant).")
    st.text_area("Transcript", key="mt_transcript", height=200)

    col1, col2 = st.columns(2)
    with col1:
        chatbot_role = st.text_input("Chatbot role", value=demo_data.DEFAULT_CHATBOT_ROLE,
                                      key="mt_role")
        scenario = st.text_input("Scenario", value=demo_data.DEFAULT_SCENARIO, key="mt_scenario")
    with col2:
        expected_outcome = st.text_input("Expected outcome",
                                          value=demo_data.DEFAULT_EXPECTED_OUTCOME, key="mt_outcome")

    st.subheader("2️⃣ Run")
    if st.button("Run Multiturn Eval", type="primary", key="mt_run"):
        turns_data = _parse_transcript(st.session_state["mt_transcript"])
        with st.spinner("Judging the conversation with Groq (7 metrics)…"):
            try:
                result = run_in_thread(
                    run_multiturn_eval, turns_data, chatbot_role, scenario, expected_outcome,
                    timeout=WORK_TIMEOUT)
                st.session_state["multiturn_result"] = result
            except Exception as e:  # surface errors — never hide them in a BYOK app
                st.session_state["multiturn_result"] = {"error": f"{type(e).__name__}: {e}"}

    result = st.session_state.get("multiturn_result")
    if result:
        st.divider()
        if "error" in result:
            st.error(result["error"])
        else:
            st.caption("Metric breakdown:")
            render_metric_breakdown(result["breakdown"])


def _parse_transcript(text: str) -> list[tuple[str, str]]:
    turns = []
    for line in text.splitlines():
        if ":" not in line or not line.strip():
            continue
        role, content = line.split(":", 1)
        role = role.strip().lower()
        if role in ("user", "assistant") and content.strip():
            turns.append((role, content.strip()))
    return turns


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    section = render_sidebar()

    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)

    if not unlocked():
        st.info("👈 Paste your Groq API key in the sidebar and click **Validate** to begin.")
        return

    if section == "Types of Metrics":
        render_types_of_metrics_section()
    elif section == "RAG Evals":
        render_rag_section()
    elif section == "Agent Evals":
        render_agent_section()
    elif section == "Multiturn Evals":
        render_multiturn_section()


if __name__ == "__main__":
    main()
