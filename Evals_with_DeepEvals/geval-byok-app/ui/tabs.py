"""ui/tabs.py — small presentation helpers."""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def render_mermaid(code: str, height: int = 320):
    """Render a Mermaid diagram via an iframe (mermaid.js from CDN, no extra dependency).

    Keep node labels plain (letters/numbers/spaces/hyphens); quotes, slashes, dots, and
    parentheses inside labels can trip the parser.

    st.tabs() renders every tab panel into the DOM upfront and just hides inactive ones with
    display:none, so a diagram in a non-default tab loads inside a zero-size iframe. Mermaid's
    startOnLoad measures text against that zero-size canvas and throws, surfacing as a generic
    "Syntax error in text" — not an actual syntax problem.

    requestAnimationFrame does NOT fix this: rAF callbacks are tied to the paint cycle, and
    browsers skip painting entirely for display:none content, so a hidden iframe's first rAF
    callback never fires — the diagram silently never renders, even after switching to that
    tab, since nothing else re-triggers the check. setTimeout is not tied to painting and keeps
    firing regardless of display:none, so polling with it reliably catches the moment the tab
    becomes visible (offsetWidth flips from 0 to nonzero) and renders then.
    """
    html = ('<div class="mermaid">\n' + code.strip() + '\n</div>\n'
            '<script type="module">\n'
            "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n"
            "mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' });\n"
            "function tryRender() {\n"
            "  if (document.body.offsetWidth > 0) { mermaid.run(); }\n"
            "  else { setTimeout(tryRender, 150); }\n"
            "}\n"
            "tryRender();\n"
            '</script>')
    components.html(html, height=height, scrolling=True)


TOOL_ICONS = {"search_docs": "📚", "web_search": "🌐", "escalate_to_human": "🆘"}


def render_tool_badges(tool_events: list[dict]) -> None:
    """Compact chip row under a chat bubble -- which tool(s) fired, and whether the agent stated
    a plan first -- so the bubble itself stays just the final answer. Full inputs/outputs live in
    the dedicated Trace section instead of cluttering the conversation.
    """
    if not tool_events:
        return
    reasoning = next((ev.get("reasoning") for ev in tool_events if ev.get("reasoning")), "")
    chips = ["🧠 planned"] if reasoning else []
    seen = set()
    for ev in tool_events:
        if ev["tool"] in seen:
            continue
        seen.add(ev["tool"])
        chips.append(f"{TOOL_ICONS.get(ev['tool'], '🔧')} {ev['tool']}")
    st.caption(" · ".join(chips))
    if reasoning:
        with st.expander("🧠 Plan", expanded=False):
            st.write(reasoning)
