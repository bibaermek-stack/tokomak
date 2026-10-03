"""Sandboxed file read/write for the advisor. Only ai_files/."""

from __future__ import annotations

import csv
import io
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
FILES_DIR = ROOT / "ai_files"
MAX_TEXT = 200_000
MAX_IMAGE = 4_000_000
_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_TEXT_EXT = {".txt", ".md", ".csv", ".json", ".html"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_ALLOWED_EXT = _TEXT_EXT | _IMAGE_EXT
_BLOCKED = {".env", ".env.example", ".git"}
_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif",
    ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv",
    ".json": "application/json", ".html": "text/html",
}


def _ensure() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep = FILES_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")


def safe_name(name: str) -> Optional[str]:
    raw = (name or "").strip().replace("\\", "/").split("/")[-1]
    if not raw or raw in _BLOCKED or raw.startswith("."):
        return None
    if not _SAFE.match(raw):
        return None
    ext = Path(raw).suffix.lower()
    if ext not in _ALLOWED_EXT:
        return None
    return raw


def is_image(name: str) -> bool:
    return Path(name or "").suffix.lower() in _IMAGE_EXT


def mime_of(name: str) -> str:
    return _MIME.get(Path(name or "").suffix.lower(), "application/octet-stream")


def unique_name(name: str) -> Optional[str]:
    n = safe_name(name)
    if not n:
        return None
    _ensure()
    path = FILES_DIR / n
    if not path.exists():
        return n
    stem, ext = Path(n).stem, Path(n).suffix
    for i in range(2, 40):
        cand = "%s_%d%s" % (stem, i, ext)
        if not (FILES_DIR / cand).exists():
            return cand
    return "%s_%s%s" % (stem, uuid.uuid4().hex[:6], ext)


def resolve(name: str) -> Optional[Path]:
    n = safe_name(name)
    if not n:
        return None
    path = (FILES_DIR / n).resolve()
    try:
        path.relative_to(FILES_DIR.resolve())
    except ValueError:
        return None
    return path


def list_files() -> List[dict]:
    _ensure()
    out = []
    for p in sorted(FILES_DIR.iterdir()):
        if p.name.startswith(".") or not p.is_file():
            continue
        if not safe_name(p.name):
            continue
        out.append({"name": p.name, "bytes": p.stat().st_size})
    return out


def read_file(name: str) -> dict:
    path = resolve(name)
    if path is None or not path.is_file():
        return {"ok": False, "op": "read", "name": name, "error": "табылмады"}
    data = path.read_bytes()
    limit = MAX_IMAGE if is_image(path.name) else MAX_TEXT
    if len(data) > limit:
        return {"ok": False, "op": "read", "name": path.name, "error": "тым үлкен"}
    if is_image(path.name):
        return {"ok": True, "op": "read", "name": path.name, "kind": "image",
                "bytes": len(data), "mime": mime_of(path.name)}
    text = data.decode("utf-8", errors="replace")
    return {"ok": True, "op": "read", "name": path.name, "kind": "text",
            "content": text, "bytes": len(data)}


def write_file(name: str, content: str) -> dict:
    path = resolve(name)
    if path is None:
        return {"ok": False, "op": "write", "name": name, "error": "жарамсыз ат"}
    text = content if isinstance(content, str) else str(content)
    raw = text.encode("utf-8")
    if len(raw) > MAX_TEXT:
        return {"ok": False, "op": "write", "name": path.name, "error": "тым үлкен"}
    _ensure()
    path.write_bytes(raw)
    return {"ok": True, "op": "write", "name": path.name, "kind": "text",
            "bytes": len(raw)}


def save_upload(name: str, data: bytes) -> dict:
    n = unique_name(name)
    if not n:
        return {"ok": False, "op": "upload", "name": name, "error": "жарамсыз ат"}
    raw = data if isinstance(data, (bytes, bytearray)) else b""
    limit = MAX_IMAGE if is_image(n) else MAX_TEXT
    if len(raw) > limit:
        return {"ok": False, "op": "upload", "name": n, "error": "тым үлкен"}
    if not raw:
        return {"ok": False, "op": "upload", "name": n, "error": "бос"}
    _ensure()
    path = FILES_DIR / n
    path.write_bytes(bytes(raw))
    return {
        "ok": True, "op": "upload", "name": n,
        "kind": "image" if is_image(n) else "text",
        "bytes": len(raw), "mime": mime_of(n),
    }


def load_image_b64(name: str) -> Optional[dict]:
    import base64
    path = resolve(name)
    if path is None or not path.is_file() or not is_image(path.name):
        return None
    data = path.read_bytes()
    if len(data) > MAX_IMAGE:
        return None
    return {
        "name": path.name,
        "mime": mime_of(path.name),
        "b64": base64.b64encode(data).decode("ascii"),
    }


def table_csv(rows: List[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["path", "value", "unit"])
    for r in rows:
        w.writerow([r.get("path", ""), r.get("value", ""), r.get("unit", "")])
    return buf.getvalue()


def apply_ops(ops: Any) -> List[dict]:
    results: List[dict] = []
    if not isinstance(ops, list):
        return results
    for item in ops[:8]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "").strip().lower()
        if op == "list":
            results.append({"ok": True, "op": "list", "files": list_files()})
        elif op == "read":
            results.append(read_file(str(item.get("name") or "")))
        elif op == "write":
            results.append(write_file(str(item.get("name") or ""),
                                      str(item.get("content") or "")))
    return results
