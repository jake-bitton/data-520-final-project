"""Streamlit front-end v1.  Run:  streamlit run ui/app.py"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import api_client
import components as ui

st.set_page_config(page_title="IJC Policy Assistant", page_icon="💧", layout="wide")
st.title("IJC Policy Assistant")
st.info(ui.DISCLAIMER, icon="⚠️")

for key, default in {"last_run": None, "query": "", "generating": False}.items():
    st.session_state.setdefault(key, default)

settings = ui.sidebar()
ask_tab, history_tab = st.tabs(["Ask", "History"])

# ---------------------------------------------------------------- Ask
with ask_tab:
    st.caption("Each request is independent (no conversation memory), so runs stay reproducible.")
    cols = st.columns(len(ui.EXAMPLE_PROMPTS))
    for col, example in zip(cols, ui.EXAMPLE_PROMPTS):
        if col.button(example[:60] + "…", help=example, use_container_width=True):
            st.session_state.query = example

    query = st.text_area("Policy request", key="query", height=120, max_chars=4000)
    submitted = st.button(
        "Generate", type="primary", disabled=settings is None or st.session_state.generating
    )

    if submitted and query.strip():
        st.session_state.generating = True
        run = {"run_id": None, "sources": [], "reasoning": "", "output": "", "done": None, "error": None}

        status_box = st.empty()
        with st.expander("Model reasoning", expanded=False):
            reasoning_box = st.empty()
        answer_box = st.empty()
        status_box.info("Thinking… (local reasoning models can take a while)", icon="🤔")

        try:
            for ev in api_client.generate_stream({"query": query, **settings}):
                t = ev["type"]
                if t == "meta":
                    run["run_id"] = ev["run_id"]
                elif t == "sources":
                    run["sources"] = ev["chunks"]
                elif t == "reasoning":
                    run["reasoning"] += ev["text"]
                    reasoning_box.text(run["reasoning"])
                elif t == "token":
                    if not run["output"]:
                        status_box.empty()
                    run["output"] += ev["text"]
                    answer_box.markdown(run["output"] + "▌")
                elif t == "done":
                    run["done"] = ev
                elif t == "error":
                    run["error"] = ev["message"]
        except api_client.BackendError as e:
            run["error"] = str(e)
        finally:
            st.session_state.generating = False
            status_box.empty()
            answer_box.empty()
            reasoning_box.empty()
        st.session_state.last_run = run
        st.rerun()  # re-render the finished run below with the button re-enabled
    elif submitted:
        st.warning("Enter a policy request first.")

    run = st.session_state.last_run
    if run:
        if run["error"]:
            st.error(run["error"])
        done = run.get("done") or {}
        if done.get("status") == "truncated":
            ui.truncation_warning()
        for w in done.get("warnings") or []:
            st.warning(w)
        if run["output"]:
            st.markdown(run["output"])
        elif done:
            st.warning("The model returned an empty answer.")
        with st.expander("Model reasoning"):
            st.text(run["reasoning"] or "(none)")
        with st.expander("Sources"):
            ui.sources_list(run["sources"])
        if done:
            ui.run_footer(run["run_id"], (ui.cached_config() or {}).get("model", ""), done)

# ---------------------------------------------------------------- History
with history_tab:
    if st.button("Refresh"):
        pass
    try:
        runs = api_client.list_runs(limit=200)
    except api_client.BackendError as e:
        st.error(str(e))
        runs = []

    if not runs:
        st.caption("No runs yet.")
    else:
        df = pd.DataFrame(runs)
        local_tz = datetime.now().astimezone().tzinfo
        df["time"] = pd.to_datetime(df["created_at"]).dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d %H:%M:%S")
        df["query"] = df["query"].str.slice(0, 90)
        df["RAG"] = df["use_rag"].map({True: "yes", False: "no"})
        view = df[["time", "query", "RAG", "status", "latency_ms"]]
        event = st.dataframe(
            view, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="history_table",
        )
        rows = event.selection.rows if event and event.selection else []
        if rows:
            run_id = df.iloc[rows[0]]["run_id"]
            st.session_state.history_selection = run_id
            st.divider()
            try:
                ui.run_detail(api_client.get_run(run_id))
            except api_client.BackendError as e:
                st.error(str(e))
        else:
            st.caption("Select a row to see the full run.")
