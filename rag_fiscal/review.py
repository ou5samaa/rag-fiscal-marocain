"""Rendu de pages témoins et enregistrement explicite de la revue humaine."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (ROOT, configure_console, inside_project, read_json,
                     run_id, safe_write, sha256, write_json)


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="Rendre uniquement les pages des échantillons de revue")
    render.add_argument("--run", type=Path, required=True)
    render.add_argument("--dpi", type=int, default=130)
    accept = sub.add_parser("accept", help="À utiliser seulement APRÈS comparaison manuelle au PDF")
    accept.add_argument("--run", type=Path, required=True)
    accept.add_argument("--article-id", required=True)
    accept.add_argument("--reviewer", required=True)
    accept.add_argument("--comment", required=True)
    accept.add_argument("--all-checks-confirmed", action="store_true", required=True)
    issues = sub.add_parser("resolve-issues", help="Consigner la résolution manuelle de toutes les alertes")
    issues.add_argument("--run", type=Path, required=True)
    issues.add_argument("--comment", required=True)
    snapshot = sub.add_parser("snapshot", help="Archiver un fichier avant son édition manuelle")
    snapshot.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        path = inside_project(args.file)
        if not path.is_file():
            raise ValueError("Fichier à archiver introuvable.")
        # safe_write conserve la version précédente ; le contenu courant reste identique.
        safe_write(path, path.read_bytes())
        return
    run = inside_project(args.run)
    metadata = read_json(run / "run.json")
    if metadata["status"] != "extracted_pending_manual_review":
        raise ValueError("Il faut une extraction terminée avant la revue.")
    for name, digest in metadata["artifact_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Artefact modifié : {name}. Corriger le parser et réextraire.")
    checklist = read_json(run / "review_checklist.json")
    if args.command == "render":
        if not 72 <= args.dpi <= 250:
            raise ValueError("DPI attendu entre 72 et 250.")
        try:
            import pymupdf
        except ImportError as exc:
            raise SystemExit("PyMuPDF manque ; aucune installation automatique.") from exc
        target = run / "review_images" / run_id()
        selected = sorted({(s["edition"], page) for s in checklist["samples"] for page in s["pdf_pages"]})
        for edition in sorted({edition for edition, _ in selected}):
            source = next(s for s in metadata["sources"] if s["edition"] == edition)
            path = ROOT / source["file"]
            if sha256(path) != source["sha256"]:
                raise ValueError("PDF source modifié depuis l'extraction.")
            with pymupdf.open(path) as doc:
                for year, page in selected:
                    if year == edition:
                        png = doc[page - 1].get_pixmap(dpi=args.dpi, alpha=False).tobytes("png")
                        safe_write(target / f"cgi_{edition}_pdf_{page:04d}.png", png)
        print(target)
        return
    if not args.comment.strip():
        raise ValueError("Un commentaire de revue est requis.")
    if args.command == "accept":
        from datetime import datetime, timezone
        matches = [item for item in checklist["articles"] if item["article_id"] == args.article_id]
        if len(matches) != 1 or not args.reviewer.strip():
            raise ValueError("Article inconnu ou nom du réviseur vide.")
        item = matches[0]
        item.update(status="accepted", reviewer=args.reviewer, reviewed_at=datetime.now(timezone.utc).isoformat(),
                    comments=args.comment, review_kind="human", human_review_status="declared_completed")
        item["checks"] = {key: True for key in item["checks"]}
        if checklist.get("review_kind") == "agent_assisted_technical":
            checklist["all_issues_reviewed"] = False
            checklist["issue_resolution_notes"] = None
            if all(entry["status"] == "accepted" for entry in checklist["articles"]):
                checklist.update(review_kind="human", human_review_status="declared_completed")
    else:
        checklist.update(all_issues_reviewed=True, issue_resolution_notes=args.comment)
    write_json(run / "review_checklist.json", checklist)


if __name__ == "__main__":
    main()
