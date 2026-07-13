"""core/keys.py — BYOK key handling: a provider registry + resolution + validation + gating.

Groq is the only provider now — it's both the judge for every metric and the "model under
test" for RAG/Agent/Multiturn. A single Groq account's free-tier limits (30 RPM / 12K TPM for
llama-3.3-70b-versatile — see console.groq.com/docs/rate-limits) are shared across every call a
click makes, since Groq's limits are org-level, not per-key. `GROQ_API_KEY` (required) covers
everything on its own. `GROQ_API_KEY_2` (optional) is a second key from a *different* Groq
account: when present, judge calls use the first key and generation calls use the second, so
the two draw from two separate rate-limit pools instead of one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# Validators — one cheap call each; return (ok, short_message). Lazy-import the SDKs.
# ──────────────────────────────────────────────────────────────────────────────
def validate_groq(api_key: str | None) -> tuple[bool, str]:
    if not api_key:
        return False, "no key provided"
    try:
        from groq import Groq
        Groq(api_key=api_key).models.list()
        return True, "key OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


# ──────────────────────────────────────────────────────────────────────────────
# Provider registry — edit this to match the use case.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Provider:
    env: str
    label: str
    help_url: str
    validate: Callable[[str | None], tuple[bool, str]]
    required: bool = True


PROVIDERS: dict[str, Provider] = {
    "GROQ_API_KEY": Provider("GROQ_API_KEY", "Groq (judge, + generation if no 2nd key)",
                              "https://console.groq.com/keys", validate_groq, True),
    "GROQ_API_KEY_2": Provider("GROQ_API_KEY_2", "Groq (optional 2nd key, different account, "
                                "for generation)", "https://console.groq.com/keys",
                                validate_groq, False),
}


# ──────────────────────────────────────────────────────────────────────────────
# Resolution + gating
# ──────────────────────────────────────────────────────────────────────────────
def get_secret(name: str) -> str | None:
    """Read a key from Streamlit secrets first, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def apply_keys(overrides: dict[str, str]) -> dict[str, bool]:
    """Resolve each provider key (sidebar override → secrets → env) and push to os.environ.

    Returns {env_name: present}. Call on every run before any provider call.
    """
    present = {}
    for name in PROVIDERS:
        val = (overrides.get(name) or "").strip() or get_secret(name)
        if val:
            os.environ[name] = val
        else:
            os.environ.pop(name, None)
        present[name] = bool(val)
    return present


def required_keys() -> list[str]:
    return [n for n, p in PROVIDERS.items() if p.required]


def all_required_valid(valid: dict[str, bool]) -> bool:
    return all(valid.get(n) for n in required_keys())
