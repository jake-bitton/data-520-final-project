"""Streamlit widgets, kept here so app.py stays short."""
from __future__ import annotations

import json
from typing import Optional

import streamlit as st

import api_client

EXAMPLE_PROMPTS = [
    "Propose a governance structure for coordinating flood response in the Lake Champlain–Richelieu River basin.",
    "Recommend policies to improve U.S.–Canada data sharing on Great Lakes water quality.",
    "Suggest how the IJC could better include Indigenous nations in watershed board decisions.",
]

DISCLAIMER = "AI-generated output for research use; requires human review."


@st.cache_data(ttl=30, show_spinner=False)
def cached_config() -> dict:
    return api_client.config()


def sidebar() -> Optional[dict]:
    """Render the sidebar and return the generation settings (or None if the backend is down)."""
    with st.sidebar:
        st.header("Settings")
        try:
            cfg = cached_config()
            health = api_client.health()
        except api_client.BackendError as e:
            st.error(str(e))
            return None

        mode = " (fake mode)" if cfg["llm_mode"] == "fake" else ""
        if health.get("llm_reachable"):
            st.success(f"Backend OK · LLM reachable{mode}", icon="✅")
        else:
            st.warning(f"Backend OK · LLM not reachable at its endpoint{mode}", icon="⚠️")

        st.markdown(
            f"**Model:** `{cfg['model']}`  \n**Server:** `{cfg['llm_server']}`  \n"
            f"**Prompt version:** `{cfg['prompt_version']}`"
        )

        rag_ok = cfg["rag_available"]
        use_rag = st.toggle("Use RAG (IJC corpus)", value=False, disabled=not rag_ok)
        if not rag_ok:
            st.caption("Corpus not connected yet")

        d, lim = cfg["defaults"], cfg["limits"]
        with st.expander("Advanced"):
            temperature = st.slider(
                "Temperature", float(lim["temperature"]["min"]), float(lim["temperature"]["max"]),
                float(d["temperature"]), 0.05,
            )
            max_tokens = st.number_input(
                "Max tokens (includes reasoning)", lim["max_tokens"]["min"], lim["max_tokens"]["max"],
                int(d["max_tokens"]), step=500,
            )
            efforts = lim["reasoning_effort"]
            reasoning_effort = st.selectbox("Reasoning effort", efforts, index=efforts.index(d["reasoning_effort"]))
            k = st.slider("Top-k sources", lim["k"]["min"], lim["k"]["max"], int(d["k"]), disabled=not use_rag)
            use_seed = st.checkbox("Fix seed", value=d["seed"] is not None)
            seed = st.number_input(
                "Seed", lim["seed"]["min"], lim["seed"]["max"], int(d["seed"] or 42), disabled=not use_seed
            )

    payload = {
        "use_rag": use_rag,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "reasoning_effort": reasoning_effort,
    }
    if use_rag:
        payload["k"] = int(k)
    if use_seed:
        payload["seed"] = int(seed)
    return payload


def sources_list(chunks: list[dict]) -> None:
    if not chunks:
        st.caption("No sources: this run used the baseline (no retrieval).")
        return
    for i, c in enumerate(chunks, 1):
        title = c.get("title") or "Untitled"
        link = f"[{title}]({c['source_url']})" if c.get("source_url") else title
        score = f" · score {c['score']:.3f}" if c.get("score") is not None else ""
        st.markdown(f"**[{i}]** {link} — {c.get('section') or ''}{score}")
        st.caption((c.get("text") or "")[:400])


def run_footer(run_id: Optional[str], model: str, done: dict) -> None:
    u = done.get("usage") or {}
    st.caption(
        f"Run `{run_id}` · {model} · {done.get('latency_ms', '—')} ms · "
        f"prompt {u.get('prompt_tokens', '—')} / completion {u.get('completion_tokens', '—')} / "
        f"reasoning {u.get('reasoning_tokens', '—')} tokens · finish: {done.get('finish_reason')}"
    )


def truncation_warning() -> None:
    st.warning(
        "This answer was truncated (finish_reason = length). The model ran out of tokens, "
        "possibly while reasoning. Increase max tokens and retry; do not use this run for evaluation.",
        icon="✂️",
    )


def run_detail(run: dict) -> None:
    status = run["status"]
    st.markdown(f"**Status:** `{status}` · **Created:** {run['created_at']} · **RAG:** {'yes' if run['use_rag'] else 'no'}")
    if status == "truncated":
        truncation_warning()
    if run.get("error"):
        st.error(run["error"])
    st.markdown("**Query**")
    st.write(run["query"])
    st.markdown("**Output**")
    st.markdown(run.get("output") or "_(empty)_")
    with st.expander("Model reasoning"):
        st.text(run.get("reasoning") or "(none)")
    with st.expander("Sources"):
        sources_list(run.get("sources") or [])
    with st.expander("Parameters"):
        keys = ["model", "llm_server", "prompt_version", "temperature", "max_tokens", "reasoning_effort",
                "seed", "k", "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                "retrieval_ms", "latency_ms", "app_git_sha", "batch_id"]
        st.json({k: run.get(k) for k in keys})
    with st.expander("Rendered prompt"):
        st.json(run.get("rendered_prompt") or [])
    st.download_button(
        "Download JSON",
        data=json.dumps(run, indent=2, ensure_ascii=False),
        file_name=f"run_{run['run_id']}.json",
        mime="application/json",
    )
