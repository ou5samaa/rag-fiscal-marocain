"""Entrées/sorties locales, traçabilité et protection des sources."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inside_project(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Chemin hors projet : {resolved}")
    return resolved


def safe_write(path: Path, content: str | bytes) -> None:
    """Écrire atomiquement dans le projet, sans modifier les PDF sources."""
    path = inside_project(path)
    relative = path.relative_to(ROOT)
    if relative.parts[0] in {"sources", "historique", ".git", ".agents", ".codex"}:
        raise ValueError(f"Écriture interdite : {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    try:
        if isinstance(content, bytes):
            temporary.write_bytes(content)
        else:
            temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value) -> None:
    safe_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    safe_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def new_run(category: str) -> Path:
    if category not in {"extractions", "indexes", "searches"}:
        raise ValueError(category)
    path = ROOT / "data" / category / run_id()
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_provenance(edition: int, config: dict) -> dict:
    path = inside_project(ROOT / config["file"])
    matches = [item for item in read_json(ROOT / "sources/sources_manifest.json")
               if int(item["year"]) == edition and item["file"] == path.name]
    if len(matches) != 1:
        raise ValueError(f"Manifeste ambigu ou absent pour {edition}")
    manifest = matches[0]
    digest = sha256(path)
    if digest.lower() != config["sha256"].lower() or digest.lower() != manifest["sha256"].lower():
        raise ValueError(f"Empreinte inattendue : {path}")
    if path.stat().st_size != manifest["size_bytes"]:
        raise ValueError(f"Taille inattendue : {path}")
    return {"edition": edition, "language": "fr", "file": config["file"],
            "sha256": digest, "source_url": manifest["source_url"],
            "downloaded_at": manifest.get("downloaded_at"),
            "provenance_status": "local_manifest_hash_verified"}


def code_hashes() -> dict:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted((ROOT / "rag_fiscal").glob("*.py"))}


def reviewed_articles(run: Path, allow_agent_review: bool = False) -> tuple[dict, list[dict]]:
    """Revue humaine par défaut ; revue technique IA seulement sur option explicite."""
    run = inside_project(run)
    meta = read_json(run / "run.json")
    if meta["status"] != "extracted_pending_manual_review" or not meta["structural_checks_passed"]:
        raise ValueError("Extraction incomplète ou contrôles structurels en échec.")
    for name, digest in meta["artifact_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Artefact modifié après extraction : {name}. Réextraire après correction du code.")
    for source in meta["sources"]:
        if sha256(ROOT / source["file"]) != source["sha256"]:
            raise ValueError("PDF source modifié depuis l'extraction.")
    review = read_json(run / "review_checklist.json")
    agent_review = review.get("review_kind") == "agent_assisted_technical"
    if agent_review:
        if not allow_agent_review:
            raise ValueError("Corpus revu techniquement par IA, pas par un humain. Option --allow-agent-review requise pour la baseline expérimentale.")
        audit = read_json(run / "technical_audit.json")
        inspection = read_json(run / "agent_review.json")
        if (audit["status"] != "passed" or audit["corpus_artifact_sha256"] != meta["artifact_sha256"]
                or inspection.get("status") != "completed_technical_review"
                or inspection.get("corpus_artifact_sha256") != meta["artifact_sha256"]):
            raise ValueError("Preuves de revue technique absentes ou périmées.")
    articles = read_jsonl(run / "articles.jsonl")
    items = review["articles"]
    if len(items) != len(articles) or {item["article_id"] for item in items} != {a["id"] for a in articles}:
        raise ValueError("La revue doit couvrir chaque article exactement une fois.")
    required_checks = {"boundaries", "main_text", "title_hierarchy_subdivisions", "notes_and_links", "pagination"}
    for item in items:
        checks = item.get("checks", {})
        expected_status = "accepted_technical" if agent_review else "accepted"
        if (item.get("status") != expected_status or not item.get("reviewer") or not item.get("reviewed_at")
                or not required_checks.issubset(checks) or not all(checks[key] is True for key in required_checks)):
            raise ValueError(f"Revue manuelle incomplète : {item['article_id']}")
    if review.get("all_issues_reviewed") is not True or not review.get("issue_resolution_notes"):
        raise ValueError("Les alertes doivent être examinées et leur résolution documentée.")
    return meta, articles


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
