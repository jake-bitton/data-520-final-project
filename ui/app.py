"""Session-based chat with a separate per-run archive."""
from __future__ import annotations

from datetime import datetime
import hashlib
import pandas as pd
import streamlit as st
import api_client
import components as ui

st.set_page_config(page_title="IJC Policy Assistant", page_icon="💧", layout="wide")
st.title("IJC Policy Assistant")
st.info(ui.DISCLAIMER, icon="⚠️")
st.session_state.setdefault("turns", [])
st.session_state.setdefault("chat_mode", None)
settings = ui.sidebar()
chat, history_tab = st.tabs(["Chat", "History"])

with chat:
    if st.button("New chat"):
        st.session_state.turns = []
        st.session_state.chat_mode = None
        st.rerun()
    st.caption("Conversation memory lasts for this browser session. Runs are saved separately.")
    mismatch = bool(settings and st.session_state.turns and st.session_state.chat_mode != settings["use_rag"])
    if mismatch:
        st.warning("Start a new chat to switch between RAG and baseline mode.")
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            if turn["output"]:
                st.markdown(turn["output"])
            if turn.get("error"):
                st.error(turn["error"])
            done = turn.get("done") or {}
            if done.get("status") == "truncated":
                ui.truncation_warning()
            for warning in done.get("warnings", []):
                st.warning(warning)
            if done and not turn["output"]:
                st.warning("The model returned an empty answer.")
            if turn.get("reasoning"):
                with st.expander("Model reasoning"):
                    st.text(turn["reasoning"])
            with st.expander("Sources"):
                if turn.get("sources"):
                    ui.sources_list(turn["sources"])
                else:
                    st.caption("No sources for this response.")
            if done:
                ui.run_footer(turn.get("run_id"), (ui.cached_config() or {}).get("model", ""), done)

    query = st.chat_input("Ask about IJC policy", disabled=settings is None or mismatch, max_chars=4000, submit_mode="disable")
    if query and settings:
        prior = []
        for turn in st.session_state.turns:
            if (turn.get("done") or {}).get("status") == "completed" and turn.get("output") and not turn.get("error"):
                prior.extend([{"role": "user", "content": turn["query"]}, {"role": "assistant", "content": turn["output"]}])
        st.session_state.chat_mode = settings["use_rag"]
        run = {"query": query, "run_id": None, "sources": [], "reasoning": "", "output": "", "done": None, "error": None}
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            status = st.empty()
            answer = st.empty()
            status.info("Generating response…")
            try:
                for ev in api_client.generate_stream({"query": query, "history": prior, **settings}):
                    kind = ev["type"]
                    if kind == "meta": run["run_id"] = ev["run_id"]
                    elif kind == "sources": run["sources"] = ev["chunks"]
                    elif kind == "reasoning": run["reasoning"] += ev["text"]
                    elif kind == "token":
                        run["output"] += ev["text"]
                        answer.markdown(run["output"] + "▌")
                    elif kind == "done": run["done"] = ev
                    elif kind == "error": run["error"] = ev["message"]
                if not run["done"] and not run["error"]:
                    run["error"] = "The response stream ended unexpectedly."
            except api_client.BackendError as exc:
                run["error"] = str(exc)
            status.empty()
        st.session_state.turns.append(run)
        st.rerun()

with history_tab:
    st.button("Refresh")
    if st.session_state.pop("history_deleted", False):
        st.success("Run deleted.")
    try:
        runs = api_client.list_runs(limit=200)
    except api_client.BackendError as exc:
        st.error(str(exc))
        runs = []
    if not runs:
        st.caption("No runs yet.")
    else:
        df = pd.DataFrame(runs)
        local_tz = datetime.now().astimezone().tzinfo
        df["time"] = pd.to_datetime(df["created_at"]).dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d %H:%M:%S")
        df["query"] = df["query"].str.slice(0, 90)
        df["RAG"] = df["use_rag"].map({True: "yes", False: "no"})
        # Reset row selection when other sessions add or delete runs, so a row index
        # cannot silently refer to a different run after a refresh.
        row_ids = "|".join(df["run_id"].astype(str))
        table_version = hashlib.sha256(row_ids.encode()).hexdigest()[:16]
        event = st.dataframe(df[["time", "query", "RAG", "status", "latency_ms"]], width="stretch",
                             hide_index=True, on_select="rerun", selection_mode="single-row",
                             key=f"history_table_{table_version}", alt="Saved run history")
        rows = event.selection.rows if event and event.selection else []
        if rows:
            selected_id = df.iloc[rows[0]]["run_id"]
            try:
                run = api_client.get_run(selected_id)
            except api_client.BackendError as exc:
                st.error(str(exc))
                st.session_state.pop("delete_run_pending", None)
            else:
                ui.run_detail(run)
                if st.session_state.get("delete_run_pending") != selected_id:
                    st.session_state.pop("delete_run_pending", None)
                if st.session_state.get("delete_run_pending") == selected_id:
                    st.warning("Permanently delete this shared history entry?")
                    st.text(run["query"])
                    confirm, cancel = st.columns(2)
                    if confirm.button("Confirm delete", type="primary"):
                        try:
                            api_client.delete_run(selected_id)
                        except api_client.BackendError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state.pop("delete_run_pending", None)
                            st.session_state.history_deleted = True
                            st.rerun()
                    if cancel.button("Cancel"):
                        st.session_state.pop("delete_run_pending", None)
                        st.rerun()
                elif st.button("Delete run", disabled=run["status"] == "running"):
                    st.session_state.delete_run_pending = selected_id
                    st.rerun()
        else:
            st.session_state.pop("delete_run_pending", None)
