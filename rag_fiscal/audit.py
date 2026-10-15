"""Audit technique du corpus par comparaison à un second moteur PDF, sans validation juridique."""
from __future__ import annotations

import argparse
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from .common import (ROOT, configure_console, inside_project, read_json,
                     read_jsonl, sha256, write_json)


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pdftotext", type=Path, required=True)
    args = parser.parse_args()
    run = inside_project(args.run)
    meta = read_json(run / "run.json")
    pages = read_jsonl(run / "pages.jsonl")
    articles = read_jsonl(run / "articles.jsonl")
    notes = read_jsonl(run / "notes.jsonl")
    issues = read_jsonl(run / "issues.jsonl")
    for name, digest in meta["artifact_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Artefact modifié : {name}")
    checks = []
    note_sequences = {}
    for source in meta["sources"]:
        if sha256(ROOT / source["file"]) != source["sha256"]:
            raise ValueError("Source modifiée")
        edition = source["edition"]
        group = [p for p in pages if p["edition"] == edition]
        result = subprocess.run([str(args.pdftotext), "-f", str(group[0]["pdf_page"]),
                                 "-l", str(group[-1]["pdf_page"]), "-layout", "-enc", "UTF-8",
                                 str(ROOT / source["file"]), "-"], capture_output=True, check=True)
        if result.stderr.strip():
            raise ValueError(result.stderr.decode("utf-8", "replace"))
        texts = result.stdout.decode("utf-8").split("\f")
        if texts[-1].strip() == "":
            texts.pop()
        if len(texts) != len(group):
            raise ValueError("Nombre de pages du second moteur différent")
        for page, independent_text in zip(group, texts):
            raw = "".join(row["raw_text"] for row in page["lines"])
            checks.append({"edition": edition, "pdf_page": page["pdf_page"],
                           "printed_page": page["printed_page"],
                           "same_nonwhitespace_characters": Counter(compact(raw)) == Counter(compact(independent_text)),
                           "same_nonwhitespace_reading_order": compact(raw) == compact(independent_text),
                           "unassigned_main_lines": [r["id"] for r in page["lines"]
                                                     if r["role"] == "main" and not r["article_id"]],
                           "unclassified_lines": [r["id"] for r in page["lines"]
                                                  if r["role"] == "unclassified_small_text"]})
        labels = [n["label"] for n in notes if n["edition"] == edition]
        note_sequences[str(edition)] = labels == list(map(str, range(1, len(labels) + 1)))
    article_checks = []
    for article in articles:
        related = [c for c in checks if c["edition"] == article["edition"]
                   and article["pdf_page_start"] <= c["pdf_page"] <= article["pdf_page_end"]]
        article_checks.append({"article_id": article["id"], "pages_checked": [c["pdf_page"] for c in related],
                               "all_pages_match": all(c["same_nonwhitespace_reading_order"] for c in related),
                               "text_equals_assigned_lines": article["text"] == "\n".join(r["text"] for r in article["lines"]),
                               "hierarchy_present": all(any(h["level"] == level for h in article["hierarchy"])
                                                        for level in ["livre", "partie", "titre", "chapitre"]),
                               "main_body_present": any(r["role"] == "main" for r in article["lines"])})
    passed = (not issues and all(note_sequences.values())
              and all(c["same_nonwhitespace_reading_order"] and not c["unassigned_main_lines"]
                      and not c["unclassified_lines"] and c["printed_page"] is not None for c in checks)
              and all(c["all_pages_match"] and c["text_equals_assigned_lines"]
                      and c["hierarchy_present"] and c["main_body_present"] for c in article_checks)
              and all(n["article_id"] or n["attachment_status"] == "outside_scope_or_unresolved" for n in notes))
    version = subprocess.run([str(args.pdftotext), "-v"], capture_output=True)
    report = {"status": "passed" if passed else "failed", "review_kind": "automated_technical_audit",
              "human_review_status": "pending", "corpus_artifact_sha256": meta["artifact_sha256"],
              "reference_engine": (version.stdout + version.stderr).decode("utf-8", "replace").strip(),
              "reference_executable_sha256": sha256(args.pdftotext), "pages": checks,
              "articles": article_checks, "consecutive_note_labels": note_sequences,
              "note_status_counts": dict(Counter(n["attachment_status"] for n in notes)),
              "limits": ["Comparaison après NFKC et retrait des blancs, pas une preuve de mise en page identique.",
                         "La portée juridique des paragraphes non numérotés n'est pas inférée par le dernier préfixe.",
                         "Le contrôle de texte brut ne suffit pas à garantir toutes les classifications sémantiques."]}
    write_json(run / "technical_audit.json", report)
    print(run / "technical_audit.json")
    if not passed:
        raise SystemExit("Audit en échec : examiner le rapport avant de poursuivre.")


if __name__ == "__main__":
    main()
