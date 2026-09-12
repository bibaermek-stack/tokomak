"""Advisor / IMAS catalog tests. No live network calls."""

import json

from fastapi.testclient import TestClient

from ai.files import apply_ops, read_file, safe_name, write_file
from ai.operator import _extra_headers, _validate_proposals, chat
from ai.state_map import map_snapshot, pfus_path, telemetry_table
from api.main import app
from imas.catalog import get, search


client = TestClient(app)


def test_extra_headers_ascii_when_title_is_cyrillic(monkeypatch):
    monkeypatch.setenv("AI_APP_TITLE", "СИНТЕЗ")
    monkeypatch.delenv("AI_HTTP_REFERER", raising=False)
    headers = _extra_headers()
    assert headers["X-Title"] == "SINTEZ"
    for value in headers.values():
        value.encode("ascii")


def test_safe_name_blocks_secrets_and_traversal():
    assert safe_name("../.env") is None
    assert safe_name(".env") is None
    assert safe_name("foo.exe") is None
    assert safe_name("shot.csv") == "shot.csv"
    assert safe_name("plot.png") == "plot.png"


def test_write_read_roundtrip(tmp_path, monkeypatch):
    import ai.files as fb
    monkeypatch.setattr(fb, "FILES_DIR", tmp_path)
    w = write_file("note.md", "# hello")
    assert w["ok"] is True
    r = read_file("note.md")
    assert r["ok"] is True
    assert r["content"] == "# hello"


def test_apply_ops_list_after_write(tmp_path, monkeypatch):
    import ai.files as fb
    monkeypatch.setattr(fb, "FILES_DIR", tmp_path)
    apply_ops([{"op": "write", "name": "a.txt", "content": "x"}])
    listed = apply_ops([{"op": "list"}])
    assert listed[0]["ok"] is True
    names = [f["name"] for f in listed[0]["files"]]
    assert "a.txt" in names


def test_chat_saves_csv_when_asked(monkeypatch, tmp_path):
    import ai.files as fb
    monkeypatch.setattr(fb, "FILES_DIR", tmp_path)
    monkeypatch.setattr("ai.operator._key", lambda: "x")
    monkeypatch.setattr(
        "ai.operator._complete",
        lambda payload, images=None: json.dumps({
            "reply": "сақтадым",
            "proposals": [],
            "imas_queries": [],
            "simulate": None,
            "files": [],
        }),
    )
    out = chat("кестені файлға жаз", "live", {"Pfus": 12.5, "Q": 0.4})
    assert out["files"]
    assert out["files"][0]["ok"] is True
    assert out["files"][0]["name"] == "diagnostics.csv"


def test_catalog_finds_core_ids():
    for q in ("summary", "nbi", "core_profiles", "equilibrium"):
        hits = search(q, limit=8)
        assert hits, q
        assert any(q in h["ids"] or q in h["path"] for h in hits)


def test_nbi_power_path():
    hits = search("nbi power", limit=8)
    paths = [h["path"] for h in hits]
    assert any("nbi" in p and "power" in p for p in paths)


def test_get_exact_path():
    e = get("nbi.power_launched")
    assert e is not None
    assert e["units"] == "W"


def test_pfus_maps_to_summary_fusion_power():
    mapped = map_snapshot({"Pfus": 500.0}, "live")
    paths = [f["path"] for f in mapped["fields"]]
    assert pfus_path() == "summary.fusion.power"
    assert "summary.fusion.power" in paths
    fus = next(f for f in mapped["fields"] if f["key"] == "Pfus")
    assert fus["si"] == 500.0 * 1e6


def test_telemetry_table_has_imas_paths():
    rows = telemetry_table(map_snapshot({"Pfus": 12.5, "Q": 0.4}, "live"))
    paths = [r["path"] for r in rows]
    assert "summary.fusion.power" in paths
    assert "summary.global_quantities.q_plus" in paths


def test_live_table_request_overrides_compute_refusal(monkeypatch):
    snap = {"Pfus": 12.5, "Q": 0.4, "PnbiSet": 0}
    monkeypatch.setattr("ai.operator._key", lambda: "x")
    monkeypatch.setattr(
        "ai.operator._complete",
        lambda payload, images=None: json.dumps({
            "reply": "Қазір тікелей режимдесіз. Кесте беру compute режимінде ғана мүмкін.",
            "proposals": [],
            "imas_queries": [],
            "simulate": None,
        }),
    )
    out = chat("маган кесте керек", "live", snap)
    assert out["table"]
    assert "compute режимінде ғана" not in out["reply"]
    assert any(r["path"] == "summary.fusion.power" for r in out["table"])


def test_live_and_compute_sources_stay_labelled():
    assert map_snapshot({}, "live")["source"] == "live"
    assert map_snapshot({}, "compute")["source"] == "compute"


def test_nbi_limit_clip():
    out = _validate_proposals(
        [{"actuator": "p_nbi", "to": 99, "reason": "test"}],
        {"PnbiSet": 10},
    )
    assert len(out) == 1
    assert out[0]["to"] == 50.0
    assert out[0]["clipped"] is True
    assert out[0]["from"] == 10.0


def test_unknown_actuator_dropped():
    out = _validate_proposals([{"actuator": "warp", "to": 1}], {})
    assert out == []


def test_health_schema_has_no_model(monkeypatch):
    monkeypatch.setattr("ai.operator._key", lambda: "")
    r = client.get("/ai/health")
    body = r.json()
    blob = json.dumps(body).lower()
    assert "ai_available" in body
    assert "model" not in body
    assert "deepseek" not in blob
    assert "openrouter" not in blob


def test_imas_endpoint():
    r = client.get("/ai/imas", params={"q": "nbi"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert body["hits"][0]["path"]


def test_chat_without_key_is_safe(monkeypatch):
    monkeypatch.setattr("ai.operator._key", lambda: "")
    r = client.post("/ai/chat", json={
        "message": "NBI 33",
        "mode": "live",
        "snapshot": {"PnbiSet": 0, "Pfus": 1},
    })
    assert r.status_code == 503
    body = r.json()
    assert body["ai_available"] is False
    assert "model" not in body
    assert body["proposals"] == []


def test_empty_chat_rejected():
    r = client.post("/ai/chat", json={"message": ""})
    assert r.status_code == 400


def test_simulate_allowed(monkeypatch):
    monkeypatch.setattr("ai.operator._key", lambda: "x")
    monkeypatch.setattr(
        "ai.operator._complete",
        lambda payload, images=None: json.dumps({
            "reply": "есеп дайын",
            "proposals": [],
            "imas_queries": ["nbi"],
            "simulate": {"machine": "iter", "t_end": 10},
        }),
    )
    monkeypatch.setattr(
        "ai.operator._run_simulate",
        lambda req: {"machine": "ITER", "t_end": 10, "final": {"Q": 1.2}},
    )
    out = chat("120 с есепте", "live", {"Pfus": 1, "PnbiSet": 0})
    assert out["simulate"]["t_end"] == 10
    assert out["ai_available"] is True
    assert "model" not in out
    assert any("nbi" in c["path"] for c in out["citations"])


def test_save_upload_png(tmp_path, monkeypatch):
    import ai.files as fb
    monkeypatch.setattr(fb, "FILES_DIR", tmp_path)
    out = fb.save_upload("plot.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    assert out["ok"] is True
    assert out["kind"] == "image"
    assert out["name"] == "plot.png"


def test_propose_does_not_mutate_snapshot(monkeypatch):
    snap = {"PnbiSet": 5.0, "Pfus": 10}
    monkeypatch.setattr("ai.operator._key", lambda: "x")
    monkeypatch.setattr(
        "ai.operator._complete",
        lambda payload, images=None: json.dumps({
            "reply": "NBI көтерейін бе?",
            "proposals": [{"actuator": "p_nbi", "to": 33, "reason": "Q"}],
            "imas_queries": [],
            "simulate": None,
        }),
    )
    out = chat("NBI 33", "live", snap)
    assert snap["PnbiSet"] == 5.0
    assert out["proposals"][0]["to"] == 33.0
    assert out["proposals"][0]["from"] == 5.0
