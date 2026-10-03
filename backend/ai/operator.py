"""Advisor orchestration. Never applies actuators; never exposes model ids."""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from imas.catalog import get as imas_get
from imas.catalog import search as imas_search
from . import files as filebox
from .prompts import SYSTEM
from .state_map import map_snapshot, telemetry_table

ROOT = Path(__file__).resolve().parents[2]

_LIMITS = {
    "p_nbi": (0.0, 50.0, "МВт"),
    "p_icrf": (0.0, 30.0, "МВт"),
    "p_ecrf": (0.0, 30.0, "МВт"),
    "gas": (0.0, 1.0, ""),
    "ip": (0.0, 17.0, "МА"),
    "bt": (1.0, 6.0, "Тл"),
    "pellet": (0.0, 1.0, ""),
    "elm": (0.0, 1.0, ""),
    "disrupt": (0.0, 1.0, ""),
    "mitigate": (0.0, 1.0, ""),
}
_DANGER = {"disrupt", "mitigate"}
_FROM_KEYS = {
    "p_nbi": "PnbiSet",
    "p_icrf": "PicrSet",
    "p_ecrf": "PecrSet",
    "gas": "gasSet",
    "ip": "IpSet",
    "bt": "Bt",
}
_TABLE_ASK = re.compile(
    r"кесте|таблица|table|диагностик|ids\b", re.IGNORECASE)
_FILE_WRITE = re.compile(
    r"файлға|файл жаса|файл жаз|сақта|write file|\.csv|\.json|\.md|\.txt",
    re.IGNORECASE)
_FILE_READ = re.compile(r"оқы|read|ашып бер", re.IGNORECASE)
_FILE_LIST = re.compile(r"файлдар|list files|қандай файл", re.IGNORECASE)
_REFUSE = re.compile(
    r"compute режимінде ғана|тек есеп|тек compute|only in compute",
    re.IGNORECASE)

_STRIP = re.compile(
    r"(?i)\b(deepseek|openrouter|openai|anthropic|grok|gpt-?\d|claude|"
    r"flash:free|v4-flash|sk-or-v1)\b"
)


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


def _key() -> str:
    return (os.environ.get("OPENROUTER_API_KEY") or "").strip()


def _scrub(text: str) -> str:
    if not text:
        return text
    return _STRIP.sub("кеңесші", text)


def health() -> dict:
    from imas.catalog import load
    try:
        n = len(load())
    except Exception:
        n = 0
    return {
        "ai_available": bool(_key()),
        "catalog": n > 0,
        "catalog_entries": n,
    }


def _clamp(actuator: str, value: float) -> tuple:
    lo, hi, unit = _LIMITS[actuator]
    clipped = min(hi, max(lo, float(value)))
    return clipped, clipped != float(value), unit


def _validate_proposals(raw: Any, snapshot: Optional[dict]) -> List[dict]:
    snap = snapshot or {}
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        act = str(item.get("actuator") or "").strip().lower()
        if act not in _LIMITS:
            continue
        try:
            to = float(item.get("to"))
        except (TypeError, ValueError):
            continue
        to, clipped, unit = _clamp(act, to)
        from_key = _FROM_KEYS.get(act)
        from_val = item.get("from")
        if from_val is None and from_key and from_key in snap:
            from_val = snap[from_key]
        try:
            from_val = float(from_val) if from_val is not None else None
        except (TypeError, ValueError):
            from_val = None
        danger = act in _DANGER or bool(item.get("danger"))
        imas_path = str(item.get("imas_path") or "")
        if not imas_path:
            hit = imas_search(act, limit=1)
            imas_path = hit[0]["path"] if hit else ""
        rec = {
            "id": str(uuid.uuid4()),
            "actuator": act,
            "from": from_val,
            "to": to,
            "unit": item.get("unit") or unit,
            "reason": _scrub(str(item.get("reason") or "")),
            "imas_path": imas_path,
            "danger": danger,
            "clipped": clipped,
        }
        out.append(rec)
    return out


def _citations(queries: Any, extra_paths: List[str]) -> List[dict]:
    seen = set()
    hits: List[dict] = []

    def add(entry: Optional[dict]) -> None:
        if not entry:
            return
        p = entry["path"]
        if p in seen:
            return
        seen.add(p)
        hits.append({
            "path": entry["path"],
            "units": entry.get("units", ""),
            "documentation": entry.get("documentation", ""),
        })

    if isinstance(queries, list):
        for q in queries[:6]:
            if not isinstance(q, str):
                continue
            exact = imas_get(q)
            if exact:
                add(exact)
            else:
                for e in imas_search(q, limit=4):
                    add(e)
    for path in extra_paths:
        add(imas_get(path))
    return hits[:15]


def _parse_model_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _ascii_or(value: str, fallback: str) -> str:
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        return fallback


def _extra_headers() -> dict:
    headers = {}
    title = (os.environ.get("AI_APP_TITLE") or "").strip()
    referer = (os.environ.get("AI_HTTP_REFERER") or "").strip()
    if title:
        headers["X-Title"] = _ascii_or(title, "SINTEZ")
    if referer:
        headers["HTTP-Referer"] = _ascii_or(referer, "")
        if not headers["HTTP-Referer"]:
            del headers["HTTP-Referer"]
    return headers


def _user_content(user_payload: str, images: Optional[list]) -> Any:
    if not images:
        return user_payload
    parts = [{"type": "text", "text": user_payload}]
    for img in images[:3]:
        parts.append({
            "type": "image_url",
            "image_url": {
                "url": "data:%s;base64,%s" % (img["mime"], img["b64"]),
            },
        })
    return parts


def _complete(user_payload: str, images: Optional[list] = None) -> str:
    from openai import NotFoundError, OpenAI
    client = OpenAI(
        api_key=_key(),
        base_url=os.environ.get("AI_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=60.0,
    )
    model = os.environ.get("AI_MODEL", "")
    headers = _extra_headers()

    def call(name: str, with_images: bool):
        kwargs = {
            "model": name,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": _user_content(user_payload,
                                          images if with_images else None)},
            ],
        }
        if headers:
            kwargs["extra_headers"] = headers
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    try:
        return call(model, bool(images))
    except NotFoundError:
        if model.endswith(":free"):
            try:
                return call(model[: -len(":free")], bool(images))
            except Exception:
                if images:
                    return call(model[: -len(":free")], False)
                raise
        if images:
            return call(model, False)
        raise
    except Exception:
        if images:
            return call(model, False)
        raise


def _run_simulate(req: dict) -> dict:
    from tokamak.solver import Simulator, SolverConfig
    from tokamak.machines import get_machine
    from tokamak.geometry import attach_geometry

    machine = str(req.get("machine") or "iter")
    t_end = float(req.get("t_end") or 120.0)
    t_end = min(max(t_end, 5.0), 200.0)
    cfg = SolverConfig(
        machine=machine,
        confinement_model=str(req.get("confinement_model") or "ipb98y2"),
        lh_model=str(req.get("lh_model") or "martin08"),
        transport_model=str(req.get("transport_model") or "scaling_anchored"),
        disruption_enabled=True,
    )
    m = attach_geometry(get_machine(machine))
    sim = Simulator(cfg, machine=m)
    trace = sim.run_scenario(t_end=t_end, record_every=max(t_end / 40.0, 1.0))
    final = sim.scalars()
    mapped = map_snapshot(final, "compute")
    return {
        "machine": m.label,
        "t_end": t_end,
        "final": {k: final[k] for k in (
            "t", "Ip", "Pfus", "Q", "Te", "Ti", "ne", "fG", "H_factor",
            "q95", "betaN", "tauE", "hMode", "disrupted") if k in final},
        "imas": mapped,
        "alarms": list(sim.alarms)[-8:],
        "trace_len": len(trace),
    }


def chat(message: str, mode: str = "live", snapshot: Optional[dict] = None,
         history: Optional[list] = None,
         attachments: Optional[list] = None) -> dict:
    if not _key():
        return {
            "reply": "Кеңесші әлі қосылмаған.",
            "proposals": [],
            "citations": [],
            "mode": "live",
            "ai_available": False,
            "simulate": None,
        }

    mapped = map_snapshot(snapshot, "live")
    hist = []
    for turn in (history or [])[-8:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            hist.append(f"{role}: {content}")

    attached = []
    images = []
    for name in attachments or []:
        if not isinstance(name, str):
            continue
        info = filebox.read_file(name)
        attached.append(info)
        img = filebox.load_image_b64(name)
        if img:
            images.append(img)

    payload = json.dumps({
        "message": message,
        "telemetry": mapped,
        "snapshot_hud": snapshot or {},
        "history": hist,
        "existing_files": filebox.list_files(),
        "attachments": [
            {k: v for k, v in a.items() if k != "content" or len(str(v)) < 8000}
            for a in attached
        ],
    }, ensure_ascii=False, default=str)

    try:
        raw_text = _complete(payload, images)
    except Exception as exc:
        sys.stderr.write("advisor complete failed: %s\n" % type(exc).__name__)
        return {
            "reply": "Кеңесші уақытша қолжетімсіз.",
            "proposals": [],
            "citations": [],
            "mode": "live",
            "ai_available": True,
            "simulate": None,
            "table": [],
            "files": [],
        }

    parsed = _parse_model_json(raw_text)
    reply = _scrub(str(parsed.get("reply") or raw_text or "Жауап жоқ."))
    proposals = _validate_proposals(parsed.get("proposals"), snapshot)

    simulate_out = None
    sim_req = parsed.get("simulate")
    if isinstance(sim_req, dict):
        try:
            simulate_out = _run_simulate(sim_req)
        except Exception as exc:
            reply += f"\nЕсеп қатесі: {exc}"

    table = telemetry_table(mapped)
    if _TABLE_ASK.search(message or "") and _REFUSE.search(reply):
        reply = "Тірі пульттің ағымдағы диагностикасы (IMAS):"

    extra_paths = [p["imas_path"] for p in proposals if p.get("imas_path")]
    citations = _citations(parsed.get("imas_queries"), extra_paths)

    file_ops = parsed.get("files") if isinstance(parsed.get("files"), list) else []
    msg = message or ""
    if _FILE_LIST.search(msg) and not any(
            str(x.get("op")) == "list" for x in file_ops if isinstance(x, dict)):
        file_ops = list(file_ops) + [{"op": "list"}]
    if _FILE_WRITE.search(msg) and table and not any(
            str(x.get("op")) == "write" for x in file_ops if isinstance(x, dict)):
        file_ops = list(file_ops) + [{
            "op": "write",
            "name": "diagnostics.csv",
            "content": filebox.table_csv(table),
        }]
    file_results = filebox.apply_ops(file_ops)

    show_table = bool(_TABLE_ASK.search(msg) or _FILE_WRITE.search(msg))
    out = {
        "reply": reply,
        "proposals": proposals,
        "citations": citations,
        "mode": "live",
        "ai_available": True,
        "simulate": simulate_out,
        "table": table if show_table else [],
        "files": file_results,
    }
    return out
