"""ui/async_utils.py — run blocking work off Streamlit's script thread.

Streamlit's rerun model breaks libraries that spin up their own asyncio loop (langchain async
paths, langgraph, ragas, some SDKs) with "no current event loop" / "event loop is closed".
Running that work in a worker thread — with its own loop — sidesteps the whole class of
problem WITHOUT nest_asyncio (which is flaky on 3.12+). See references/async-worker.md.

The pool below is a single, process-wide, persistent ThreadPoolExecutor, deliberately NOT
recreated per call. Spinning up a brand-new OS thread for every click (a fresh
ThreadPoolExecutor per call, torn down right after) is a real, reproducible macOS crash here:
pandas/pyarrow's bundled mimalloc allocator (used by st.dataframe, and apparently touched
internally by deepeval too) segfaults (EXC_BAD_ACCESS in mi_heap_main) the first time it's
touched from a *specific* OS thread. Reusing pooled threads fixes repeat calls on a thread
that's already been touched once — but with multiple workers, each individual worker thread is
still "fresh" (and still at risk) the first time IT personally runs a job. So every worker is
pre-warmed once, up front, by having it import pandas/pyarrow and build a throwaway DataFrame
before the pool is used for anything real.
"""
from __future__ import annotations

import asyncio
import concurrent.futures

_MAX_WORKERS = 4


def _warm_pyarrow() -> None:
    try:
        import pandas as pd
        pd.DataFrame({"a": [1]})
    except Exception:
        pass  # best-effort warmup; a failure here shouldn't block the app


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)
# Block until every worker has warmed up at least once, so no real call is ever the "first touch"
# on its thread.
concurrent.futures.wait(
    [_executor.submit(_warm_pyarrow) for _ in range(_MAX_WORKERS)],
    timeout=60,
)


def _ensure_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def run_in_thread(fn, *args, timeout: int = 1800, **kwargs):
    """Execute a synchronous callable on the shared, pre-warmed worker pool; return its result.

    Do NOT call st.* inside fn — collect results/errors into plain objects and render after.
    """
    def _wrapped():
        _ensure_loop()
        return fn(*args, **kwargs)

    return _executor.submit(_wrapped).result(timeout=timeout)
