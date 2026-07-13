"""core/providers.py — the DeepEval calls, isolated from Streamlit so they're safe in a worker
thread. Groq plays both roles: the judge for every metric, and the "model under test" for
RAG/Agent/Multiturn generation. Every run_*() call is scoped to one test case — no fan-out —
keeping calls well under Groq's free-tier rate limits (30 RPM / 12K TPM per account for
llama-3.3-70b-versatile — see console.groq.com/docs/rate-limits).

With one GROQ_API_KEY, judge and generation calls share that one account's limits. With a
second GROQ_API_KEY_2 (a different Groq account), judge calls stay on the first key and
generation calls move to the second — two separate rate-limit pools instead of one.

Every metric below is built with async_mode=False. This app runs many metrics sequentially
within one worker thread (see ui/async_utils.py), all sharing that thread's single asyncio event
loop across calls. With real network latency, deepeval's async path (metric.a_measure(), which
lazily compiles/caches Jinja prompt templates keyed by class+method) has shown a real, repeatable
failure here: a later metric in the same batch occasionally hits
`MetricTemplateInterpolationError: 'stringified_tools_called' is undefined` even though the exact
same template + kwargs render correctly in isolation — consistent with a race in that shared
per-thread event loop / template cache, not a bug in the kwargs we pass. Forcing async_mode=False
takes every metric through its synchronous code path instead, which never touches that shared
loop, and has not reproduced the failure in testing.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from types import SimpleNamespace

from core.config import GROQ_BASE_URL, GROQ_JUDGE_MODEL, GROQ_MODEL, JUDGE_CALL_PACING_SECONDS


def _build_judge():
    from deepeval.models import LocalModel
    return LocalModel(model=GROQ_JUDGE_MODEL, api_key=os.environ.get("GROQ_API_KEY"),
                       base_url=GROQ_BASE_URL, temperature=0)


def _build_argument_judge():
    """ArgumentCorrectnessMetric's prompt is the most complex one any metric in this app sends:
    a 3-tool-call few-shot example written in raw Python-repr syntax (`ToolCall(name=...,
    input_parameters={...})`), heavy with nested braces, right before a strict "ONLY return
    valid JSON" instruction. DeepEval extracts JSON from the judge's raw text with a naive
    first-'{'-to-last-'}' scan (deepeval/models/llms/utils.py:trim_and_load_json) rather than a
    real parser, so any stray brace in commentary the model adds around its answer breaks
    extraction and surfaces as "outputted an invalid JSON". GROQ_JUDGE_MODEL (a smaller model,
    picked for its higher TPM headroom on the trace-heavy metrics) is more prone to that than
    GROQ_MODEL, which this app already relies on elsewhere for reliable structured tool-call
    output. Argument Correctness only makes 1-2 calls, so TPM isn't the constraint here that it
    is for the trace-based metrics -- reliability is worth more than headroom for this one.
    """
    from deepeval.models import LocalModel
    return LocalModel(model=GROQ_MODEL, api_key=os.environ.get("GROQ_API_KEY"),
                       base_url=GROQ_BASE_URL, temperature=0)


def _build_groq_client():
    """The generation-role client. Uses the second Groq key if one was given (a different
    account, so it draws from a separate rate-limit pool than the judge); falls back to the
    primary key otherwise.
    """
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY_2") or os.environ.get("GROQ_API_KEY")
    return Groq(api_key=api_key)


def _measure_metrics(metrics: list, test_case) -> list[dict]:
    """Measure each metric against one test case, catching failures per metric so one bad judge
    call doesn't lose the rest of the breakdown. A short pacing gap between metrics spreads
    token usage across more of Groq's rolling per-minute window instead of bursting.

    Returns [{"metric", "score", "success", "reason"}, ...].
    """
    breakdown = []
    for i, metric in enumerate(metrics):
        if i > 0:
            time.sleep(JUDGE_CALL_PACING_SECONDS)
        try:
            metric.measure(test_case)
            score = metric.score
            breakdown.append({
                "metric": metric.__name__,
                "score": score,
                "success": metric.is_successful() if score is not None else False,
                "reason": metric.reason or getattr(metric, "error", None) or "",
            })
        except Exception as e:
            breakdown.append({
                "metric": getattr(metric, "__name__", type(metric).__name__),
                "score": None,
                "success": False,
                "reason": f"{type(e).__name__}: {e}",
            })
    return breakdown


def run_geval(
    input_text: str,
    actual_output: str,
    expected_output: str | None,
    metric_name: str,
    criteria: str | None,
    evaluation_steps: list[str] | None,
) -> dict:
    """Build the judge + metric, evaluate one test case, return a plain-dict result.

    Returns {"score": float | None, "reason": str, "success": bool}.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]
    if expected_output:
        params.append(SingleTurnParams.EXPECTED_OUTPUT)

    metric = GEval(
        name=metric_name,
        criteria=criteria or None,
        evaluation_steps=evaluation_steps or None,
        evaluation_params=params,
        model=_build_judge(),
        async_mode=False,
    )

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output or None,
    )

    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason, "success": metric.is_successful()}


def run_dag(
    input_text: str,
    actual_output: str,
    task_instructions: str,
    heading_criteria: str,
    order_criteria: str,
) -> dict:
    """DeepEval docs' own DAG example (https://deepeval.com/docs/metrics-dag): a TaskNode
    extracts the summary's headings, a BinaryJudgementNode checks all three are present (score 0
    and stop if not), and — only then — a NonBinaryJudgementNode checks their order (score 10/4/2
    depending on how out of order). The order node is a genuine second child of the TaskNode too,
    which is what makes this a graph rather than a tree. Worst case 3 judge calls (extract, gate,
    order), best case 2 (order skipped if the gate fails).

    Returns {"score": float | None, "reason": str, "success": bool}.
    """
    from deepeval.metrics import DAGMetric
    from deepeval.metrics.dag import (
        BinaryJudgementNode,
        DeepAcyclicGraph,
        NonBinaryJudgementNode,
        TaskNode,
        VerdictNode,
    )
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    correct_order_node = NonBinaryJudgementNode(
        criteria=order_criteria,
        children=[
            VerdictNode(verdict="Yes", score=10),
            VerdictNode(verdict="Two are out of order", score=4),
            VerdictNode(verdict="All out of order", score=2),
        ],
    )

    correct_headings_node = BinaryJudgementNode(
        criteria=heading_criteria,
        children=[
            VerdictNode(verdict=False, score=0),
            VerdictNode(verdict=True, child=correct_order_node),
        ],
    )

    extract_headings_node = TaskNode(
        instructions=task_instructions,
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        output_label="Summary headings",
        children=[correct_headings_node, correct_order_node],
    )

    dag = DeepAcyclicGraph(root_nodes=[extract_headings_node])
    metric = DAGMetric(name="Format Correctness", dag=dag, model=_build_judge(), async_mode=False)

    test_case = LLMTestCase(input=input_text, actual_output=actual_output)
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason, "success": metric.is_successful()}


def run_qag(
    input_text: str,
    actual_output: str,
    retrieval_context: list[str],
) -> dict:
    """FaithfulnessMetric — DeepEval's QAG (Question-Answer Generation) engine: extract claims
    from actual_output, ask a closed yes/no/idk question per claim against retrieval_context,
    then compute the score mathematically (no LLM score-guessing). One test case per call.

    Returns {"score": float | None, "reason": str, "success": bool, "breakdown": list[dict]}.
    """
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    metric = FaithfulnessMetric(model=_build_judge(), include_reason=True, async_mode=False)

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )

    metric.measure(test_case)

    claims = getattr(metric, "claims", []) or []
    verdicts = getattr(metric, "verdicts", []) or []
    breakdown = [
        {"claim": claim, "verdict": v.verdict, "reason": v.reason or ""}
        for claim, v in zip(claims, verdicts)
    ]

    return {
        "score": metric.score,
        "reason": metric.reason,
        "success": metric.is_successful(),
        "breakdown": breakdown,
    }


def run_rag_pipeline(question: str) -> dict:
    """Retrieve -> Groq answers grounded in the retrieved context. The pipeline half of
    01_rag_evals_with_deepeval.ipynb, with no evaluation attached — just produces something to
    evaluate. 1 Groq generation call.

    Returns {"retrieved_context": list[str], "actual_output": str}.
    """
    from core.demo_data import RAG_SYSTEM_PROMPT, retrieve

    passages = retrieve(question)
    prompt = f"Context:\n{chr(10).join(passages)}\n\nQuestion: {question}\n\nAnswer:"
    resp = _build_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip()
    return {"retrieved_context": passages, "actual_output": answer}


def run_rag_evaluation(
    question: str,
    actual_output: str,
    expected_output: str | None,
    retrieval_context: list[str],
) -> dict:
    """The 5-RAG-metric half of 01_rag_evals_with_deepeval.ipynb, scoring an already-produced
    pipeline output. Up to 5 judge calls (all Groq).

    ContextualPrecision/ContextualRecall need expected_output, so they're only included when one
    is given (mirrors the notebook's skip_on_missing_params=True, but decided at call time here
    since this app calls .measure() directly rather than going through evaluate()).

    Returns {"breakdown": list[dict]}.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge = _build_judge()
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output or None,
        retrieval_context=retrieval_context,
    )

    metrics = [
        FaithfulnessMetric(model=judge, async_mode=False),
        AnswerRelevancyMetric(model=judge, async_mode=False),
        ContextualRelevancyMetric(model=judge, async_mode=False),
    ]
    if expected_output:
        metrics += [ContextualPrecisionMetric(model=judge, async_mode=False),
                    ContextualRecallMetric(model=judge, async_mode=False)]

    breakdown = _measure_metrics(metrics, test_case)
    return {"breakdown": breakdown}


def _parse_failed_generation(text: str) -> tuple[list, str]:
    """Groq's Llama tool-calling models sometimes emit a malformed pseudo tool-call as raw text
    (e.g. `<function=search_docs {"query": "..."}</function>`, often preceded by our own
    requested "Plan: ..." sentence) instead of a structured tool_call, and Groq's own server
    then fails to parse it -- a 400 'tool_use_failed' BadRequestError. The intended call(s) are
    still recoverable from the error's `failed_generation` field; whatever text comes before the
    first `<function=...>` tag (typically the plan sentence) is preserved too, so it isn't lost.

    Returns (calls, plan_text) -- calls is [] if nothing recoverable was found.
    """
    text = (text or "").strip()
    matches = list(re.finditer(r"<function=(\w+)\s*(\{.*?\})\s*</function>", text, re.DOTALL))
    if not matches:
        return [], text
    calls = [
        SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                         function=SimpleNamespace(name=m.group(1), arguments=m.group(2)))
        for m in matches
    ]
    plan_text = text[:matches[0].start()].strip()
    return calls, plan_text


def _first_agent_completion(client, model: str, messages: list):
    from groq import BadRequestError

    from core.demo_data import AGENT_TOOL_SCHEMAS
    try:
        return client.chat.completions.create(
            model=model, messages=messages, tools=AGENT_TOOL_SCHEMAS, tool_choice="auto", temperature=0,
        ).choices[0].message
    except BadRequestError as e:
        body = getattr(e, "body", None) or {}
        error = body.get("error", {})
        if error.get("code") != "tool_use_failed":
            raise
        calls, plan_text = _parse_failed_generation(error.get("failed_generation", ""))
        if not calls:
            raise
        return SimpleNamespace(content=plan_text, tool_calls=calls)


def run_agent_turn(messages: list[dict], user_message: str) -> dict:
    """Runs one turn of the agent: appends user_message to the running Groq-format `messages`,
    lets the agent propose + execute tool calls, then produces a reply. `messages` should start
    as `[{"role": "system", "content": AGENT_SYSTEM_PROMPT}]` for a fresh conversation and is
    threaded through on each subsequent call so the agent keeps context turn to turn.

    Returns {"reply": str, "tool_events": list[dict], "messages": list[dict]} where each
    tool_event is {"tool", "input", "output", "reasoning"}.
    """
    from core.demo_data import AGENT_TOOLS

    client = _build_groq_client()
    messages = [*messages, {"role": "user", "content": user_message}]

    first = _first_agent_completion(client, GROQ_MODEL, messages)
    calls = first.tool_calls or []
    plan_text = (first.content or "").strip()

    messages.append({"role": "assistant", "content": first.content or "", "tool_calls": [
        {"id": c.id, "type": "function",
         "function": {"name": c.function.name, "arguments": c.function.arguments}}
        for c in calls
    ]})

    tool_events = []
    for c in calls:
        name = c.function.name
        args = json.loads(c.function.arguments or "{}")
        output = AGENT_TOOLS[name](**args)
        tool_events.append({"tool": name, "input": args, "output": output, "reasoning": plan_text})
        messages.append({"role": "tool", "tool_call_id": c.id, "content": output})

    if calls:
        reply = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0).choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})
    else:
        reply = plan_text or "(no tool called)"

    return {"reply": reply, "tool_events": tool_events, "messages": messages}


def build_agent_trace(query: str, final_output: str, tool_events: list[dict]) -> dict:
    """Hand-built nested trace matching the shape DeepEval's trace-based agent metrics
    (TaskCompletion/StepEfficiency/PlanAdherence/PlanQuality) expect: a root 'agent' span with
    'tool' children, each optionally carrying a 'reasoning' note. We build this manually instead
    of using the real @observe/tracing machinery, since this app calls .measure() directly
    rather than going through evaluate()'s observed-callback flow.
    """
    children = [
        {
            "name": ev["tool"],
            "type": "tool",
            "input": {"inputParameters": ev["input"]},
            "output": ev["output"],
            "reasoning": ev.get("reasoning") or "",
            "children": [],
        }
        for ev in tool_events
    ]
    return {
        "name": "agent",
        "type": "agent",
        "input": {"input": query},
        "output": {"summary": final_output},
        "children": children,
    }


def run_agent_scenario(turns: list[str]) -> dict:
    """Plays every turn of a (possibly multi-turn) scenario through the agent in sequence,
    carrying conversation history from one turn to the next. 2 generation calls per turn (tool
    decision + reply), 0 if no tool was called that turn.

    Returns {"transcript": list[(role, content)], "trace": dict, "final_output": str,
             "tool_events": list[dict]}.
    """
    from core.demo_data import AGENT_SYSTEM_PROMPT

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    transcript: list[tuple[str, str]] = []
    all_events: list[dict] = []
    final_output = ""

    for turn_text in turns:
        result = run_agent_turn(messages, turn_text)
        messages = result["messages"]
        transcript.append(("user", turn_text))
        transcript.append(("assistant", result["reply"]))
        all_events.extend(result["tool_events"])
        final_output = result["reply"]

    trace = build_agent_trace(turns[0] if turns else "", final_output, all_events)
    return {"transcript": transcript, "trace": trace, "final_output": final_output,
            "tool_events": all_events}


def run_agent_evaluation(run: dict) -> dict:
    """Scores one saved agent run against up to 6 metrics. Tool Correctness/Argument Correctness
    only run if the run has expected_tools (ground-truth scenario runs do; ad-hoc saved chats
    don't); the 4 trace-based metrics always run. Up to 6 judge calls (all Groq).

    run: {"query", "final_output", "tool_events", "trace", "expected_tools": list[str] | None,
          "goal": str | None}.

    Returns {"breakdown": list[dict]}.
    """
    from deepeval.metrics import (
        ArgumentCorrectnessMetric,
        PlanAdherenceMetric,
        PlanQualityMetric,
        StepEfficiencyMetric,
        TaskCompletionMetric,
        ToolCorrectnessMetric,
    )
    from deepeval.test_case import LLMTestCase, ToolCall

    judge = _build_judge()
    tools_called = [
        ToolCall(name=ev["tool"], input_parameters=ev["input"], output=ev["output"])
        for ev in run["tool_events"]
    ]
    expected_tools = run.get("expected_tools")

    test_case = LLMTestCase(
        input=run["query"],
        actual_output=run["final_output"],
        tools_called=tools_called,
        expected_tools=[ToolCall(name=n) for n in expected_tools] if expected_tools else None,
    )
    test_case._trace_dict = run["trace"]

    metrics = [
        StepEfficiencyMetric(model=judge, async_mode=False),
        PlanAdherenceMetric(model=judge, async_mode=False),
        PlanQualityMetric(model=judge, async_mode=False),
        TaskCompletionMetric(model=judge, task=run.get("goal") or None, async_mode=False),
    ]
    if expected_tools:
        metrics = [ToolCorrectnessMetric(model=judge, async_mode=False),
                   ArgumentCorrectnessMetric(model=_build_argument_judge(), async_mode=False)] + metrics

    breakdown = _measure_metrics(metrics, test_case)
    return {"breakdown": breakdown}


def run_agent_batch_evaluation(runs: list[dict]) -> list[dict]:
    """Evaluates every saved run in one worker-thread call (one click = one batch), so the UI
    only needs a single spinner for the whole set. A run with expected_tools makes ~12-13 real
    judge calls (Step Efficiency/Plan Adherence/Plan Quality/Task Completion each do 2-3 internal
    extract-then-score calls apiece), so a short extra pause between runs gives Groq's rolling
    TPM window a bit more room on top of the per-metric pacing in _measure_metrics.

    Returns [{"name": str, "breakdown": list[dict]}, ...].
    """
    results = []
    for i, run in enumerate(runs):
        if i > 0:
            time.sleep(JUDGE_CALL_PACING_SECONDS * 2)
        results.append({"name": run["name"], **run_agent_evaluation(run)})
    return results


def run_multiturn_eval(
    turns_data: list[tuple[str, str]],
    chatbot_role: str,
    scenario: str,
    expected_outcome: str,
) -> dict:
    """Builds a ConversationalTestCase and runs the full 7-metric suite from
    03_chatbot_conversation_evals_with_deepeval.ipynb. 7 judge calls (Groq).

    Returns {"breakdown": list[dict]}.
    """
    from deepeval.metrics import (
        ConversationalGEval,
        ConversationCompletenessMetric,
        GoalAccuracyMetric,
        KnowledgeRetentionMetric,
        RoleAdherenceMetric,
        TopicAdherenceMetric,
        TurnRelevancyMetric,
    )
    from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn

    from core.demo_data import MULTITURN_RELEVANT_TOPICS

    turns = [Turn(role=role, content=content) for role, content in turns_data]
    test_case = ConversationalTestCase(
        turns=turns,
        chatbot_role=chatbot_role or None,
        scenario=scenario or None,
        expected_outcome=expected_outcome or None,
    )

    judge = _build_judge()
    metrics = [
        RoleAdherenceMetric(model=judge, async_mode=False),
        ConversationCompletenessMetric(model=judge, async_mode=False),
        TurnRelevancyMetric(model=judge, async_mode=False),
        KnowledgeRetentionMetric(model=judge, async_mode=False),
        GoalAccuracyMetric(model=judge, async_mode=False),
        TopicAdherenceMetric(relevant_topics=MULTITURN_RELEVANT_TOPICS, model=judge, async_mode=False),
        ConversationalGEval(
            name="Helpfulness",
            criteria="Across the conversation, did the assistant give technically accurate, easy "
                     "to understand explanations, and handle the off-topic request gracefully?",
            evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
            model=judge,
            async_mode=False,
        ),
    ]
    breakdown = _measure_metrics(metrics, test_case)
    return {"breakdown": breakdown}


def generate_next_reply(turns_data: list[tuple[str, str]], user_message: str) -> str:
    """One live Groq call to generate the assistant's next reply, for the Multiturn bonus button.

    Not the full ConversationSimulator -- just enough to demo Groq live in this tab too.
    """
    from core.demo_data import MULTITURN_ASSISTANT_SYSTEM

    history = [{"role": role, "content": content} for role, content in turns_data]
    messages = [{"role": "system", "content": MULTITURN_ASSISTANT_SYSTEM}, *history,
                {"role": "user", "content": user_message}]
    resp = _build_groq_client().chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0)
    return resp.choices[0].message.content.strip()
