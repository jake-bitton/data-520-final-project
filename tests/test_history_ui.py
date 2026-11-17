"""Exercise the confirmation workflow without a live backend or Arrow dependency."""
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


def test_history_delete_confirmation_and_cancel(monkeypatch):
    ui_path = str(Path(__file__).resolve().parents[1] / "ui")
    monkeypatch.syspath_prepend(ui_path)
    import api_client
    import components
    import streamlit as st

    runs = [{"run_id": "a", "created_at": "2026-01-01T00:00:00Z", "query": "A policy question",
             "status": "completed", "use_rag": False, "latency_ms": 1}]
    selected = {"value": True}
    deleted = []
    monkeypatch.setattr(components, "sidebar", lambda: {"use_rag": False})
    monkeypatch.setattr(components, "run_detail", lambda run: None)
    monkeypatch.setattr(api_client, "list_runs", lambda **kwargs: list(runs))
    monkeypatch.setattr(api_client, "get_run", lambda run_id: dict(runs[0]))
    monkeypatch.setattr(st, "dataframe", lambda *args, **kwargs: SimpleNamespace(
        selection=SimpleNamespace(rows=[0] if selected["value"] else [])))

    def delete(run_id):
        deleted.append(run_id)
        runs.clear()
        selected["value"] = False
    monkeypatch.setattr(api_client, "delete_run", delete)

    app = AppTest.from_file(str(Path(ui_path) / "app.py")).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Delete run").click().run()
    assert any("Permanently delete" in w.value for w in app.warning)
    assert any(t.value == "A policy question" for t in app.text)
    next(b for b in app.button if b.label == "Cancel").click().run()
    assert not deleted and not any(b.label == "Confirm delete" for b in app.button)
    next(b for b in app.button if b.label == "Delete run").click().run()
    def fail_delete(run_id):
        raise api_client.BackendError("Deletion failed")
    monkeypatch.setattr(api_client, "delete_run", fail_delete)
    next(b for b in app.button if b.label == "Confirm delete").click().run()
    assert runs and any("Deletion failed" in e.value for e in app.error)
    monkeypatch.setattr(api_client, "delete_run", delete)
    next(b for b in app.button if b.label == "Confirm delete").click().run()
    assert deleted == ["a"] and not app.exception
    assert any("Run deleted" in s.value for s in app.success)
    assert any("No runs yet" in c.value for c in app.caption)
