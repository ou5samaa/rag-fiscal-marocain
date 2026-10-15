"""Corpus candidat intégral du CGI. N'altère jamais le corpus actif.

Les limites de pages proviennent d'une inspection des PDF et sont vérifiées
sur leurs octets à chaque exécution. Les anomalies restent dans issues.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import html
import re
from collections import Counter
from pathlib import Path

from .common import ROOT, configure_console, read_json, safe_write, sha256, source_provenance, write_json, write_jsonl
from .extract import LEVELS, attach_notes, folded, hierarchy_level, issue, page_lines, subdivision

CONFIG = ROOT / "config/full_scope.json"
DEFAULT_OUTPUT = ROOT / "data/candidates/full_cgi"
ARTICLE_FULL = re.compile(
    r"^Article\s+(premier|\d+)(?:\s+(bis|ter|quater|quinquies))?\s*(?:[.\-–—]+\s*(.*))?$",
    re.I,
)
FULL_LEVELS = LEVELS + ["paragraphe", "sous_paragraphe"]


def candidate_path(path: Path) -> Path:
    path = path.resolve()
    allowed = (ROOT / "data/candidates").resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("Les sorties du corpus complet doivent rester dans data/candidates/.")
    return path


def full_hierarchy_level(text: str) -> str | None:
    value = folded(text)
    if re.match(r"^SOUS[- ]PARAGRAPHE\s+\S+", value):
        return "sous_paragraphe"
    if re.match(r"^PARAGRAPHE\s+\S+", value):
        return "paragraphe"
    return hierarchy_level(text)


def article_heading_at(rows: list[dict], position: int, width: float, left_ratio: float):
    row = rows[position]
    if row["role"] != "main" or not row["bold"] or row["bbox"][0] / width > left_ratio:
        return None
    match = ARTICLE_FULL.match(row["text"])
    if match:
        return match, [position]
    if folded(row["text"]) != "ARTICLE":
        return None
    indices = [position]
    last_y = row["bbox"][1]
    for j in range(position + 1, min(position + 22, len(rows))):
        next_row = rows[j]
        next_y = next_row["bbox"][1]
        if (next_row["role"] != "main" or not next_row["bold"]
                or next_y - last_y > 16 or subdivision(next_row["text"])):
            break
        indices.append(j)
        last_y = next_y
    joined = " ".join(rows[j]["text"] for j in indices)
    match = ARTICLE_FULL.match(joined)
    return (match, indices) if match else None


def verify_sources(output: Path) -> dict:
    import pymupdf

    output = candidate_path(output)
    config = read_json(CONFIG)
    layout = read_json(ROOT / config["layout_source"])["layout"]
    manifest = {int(x["year"]): x for x in read_json(ROOT / "sources/sources_manifest.json")}
    if set(manifest) != {2023, 2024, 2025, 2026} or set(map(int, config["editions"])) != set(manifest):
        raise ValueError("Les quatre éditions doivent être présentes une seule fois.")
    results = []
    for year in sorted(manifest):
        entry = config["editions"][str(year)]
        source = source_provenance(year, {"file": entry["file"], "sha256": manifest[year]["sha256"]})
        path = ROOT / entry["file"]
        with pymupdf.open(path) as doc:
            if doc.needs_pass or doc.is_repaired or len(doc) != entry["pdf_pages"]:
                raise ValueError(f"PDF corrompu/chiffré ou nombre de pages inattendu : {year}")
            cover = folded(doc[entry["cover_pdf_page"] - 1].get_text("text"))
            if not ("CODE GENERAL DES IMPOTS" in cover and str(year) in cover
                    and "LOI DE FINANCES" in cover and "ANNEE BUDGETAIRE" in cover):
                raise ValueError(f"Couverture, édition ou français non confirmés : {year}")
            start = entry["body_start_pdf_page"]
            end = entry["body_end_pdf_page"]
            annex = entry["annex_marker_pdf_page"]
            if annex != end + 1 or start >= end:
                raise ValueError(f"Bornes incohérentes : {year}")
            start_text = folded(doc[start - 1].get_text("text"))
            end_text = folded(doc[end - 1].get_text("text"))
            annex_text = folded(doc[annex - 1].get_text("text"))
            if ("ARTICLE PREMIER" not in start_text or "ARTICLE 5" not in start_text
                    or "LOI DE FINANCES" not in start_text):
                raise ValueError(f"Début du code ou loi introductive non confirmé : {year}")
            if not re.search(rf"ARTICLE\s+{entry['last_article']}\s*[.\-–—]", end_text):
                raise ValueError(f"Dernier article non confirmé : {year}")
            if "ANNEXES AU CODE GENERAL DES IMPOTS" not in annex_text:
                raise ValueError(f"Fin du corps et début des annexes non confirmés : {year}")
            first = page_lines(doc[start - 1], year, start, layout, [])
            last = page_lines(doc[end - 1], year, end, layout, [])
            if (first["printed_page"] != entry["first_printed_page"]
                    or last["printed_page"] != entry["last_printed_page"]):
                raise ValueError(f"Pagination imprimée inattendue : {year}")
            low_text = [p + 1 for p in range(start - 1, end)
                        if len(doc[p].get_text("text").strip()) < 40]
            results.append({"edition": year, "source": source, "size_bytes": path.stat().st_size,
                            "pdf_pages": len(doc), "cover_pdf_page": entry["cover_pdf_page"],
                            "body_start_pdf_page": start, "body_end_pdf_page": end,
                            "annex_marker_pdf_page": annex,
                            "first_printed_page": first["printed_page"],
                            "last_printed_page": last["printed_page"],
                            "last_article": entry["last_article"],
                            "body_pages_without_exploitable_text": low_text,
                            "language_evidence": ["Code général des impôts", "loi de finances", "année budgétaire"],
                            "boundary_checks_passed": True})
    report = {"status": "verified_local_pdf_bytes", "schema_version": "1.0",
              "scope_sha256": sha256(CONFIG), "manifest_sha256": sha256(ROOT / "sources/sources_manifest.json"),
              "editions": results}
    write_json(output / "source_audit.json", report)
    return report


def extract_edition(doc, year: int, settings: dict, layout: dict, source: dict):
    pages, articles, issues, boundaries = [], [], [], []
    hierarchy: dict = {}
    subdivisions: dict = {}
    hierarchy_pending = None
    title_pending = False
    current = None
    started = False
    occurrences: Counter = Counter()
    for pdf_page in range(settings["body_start_pdf_page"], settings["body_end_pdf_page"] + 1):
        try:
            page = page_lines(doc[pdf_page - 1], year, pdf_page, layout, issues)
        except ValueError as exc:
            issue(issues, "page_without_exploitable_text", edition=year, pdf_page=pdf_page, detail=str(exc))
            pages.append({"edition": year, "pdf_page": pdf_page, "printed_page": None, "lines": []})
            continue
        pages.append(page)
        rows = page["lines"]
        i = 0
        while i < len(rows):
            row = rows[i]
            if row["role"] != "main":
                i += 1
                continue
            heading = article_heading_at(rows, i, page["width"], layout["body_left_x_max_ratio"])
            if heading:
                match, heading_indices = heading
                base = 1 if match[1].lower() == "premier" else int(match[1])
                suffix = match[2].lower() if match[2] else None
                label = str(base) + (" " + suffix if suffix else "")
                if not started:
                    if base != 1 or pdf_page != settings["body_start_pdf_page"]:
                        issue(issues, "first_article_unexpected", edition=year, pdf_page=pdf_page, label=label)
                    started = True
                occurrences[label] += 1
                if occurrences[label] > 1:
                    issue(issues, "duplicate_article_label", edition=year, pdf_page=pdf_page, label=label)
                article_id = f"cgi-{year}:{source['sha256'][:12].lower()}:art-{label.replace(' ', '-')}"
                if occurrences[label] > 1:
                    article_id += f":occ{occurrences[label]}"
                heading_text = " ".join(rows[j]["text"] for j in heading_indices)
                boundaries.append({"edition": year, "article": label, "line_ids": [rows[j]["id"] for j in heading_indices],
                                   "pdf_page": pdf_page, "printed_page": page["printed_page"],
                                   "heading_text": heading_text})
                current = {"id": article_id, "edition": year, "article": label,
                           "article_number": base, "article_suffix": suffix, "title": match[3] or "",
                           "heading_line_ids": [rows[j]["id"] for j in heading_indices],
                           "hierarchy": [copy.deepcopy(hierarchy[k]) for k in FULL_LEVELS if k in hierarchy],
                           "source": source, "lines": [], "note_ids": [],
                           "verification_status": "candidate_with_diagnostics"}
                articles.append(current)
                for n, j in enumerate(heading_indices):
                    rows[j]["role"] = "article_heading" if n == 0 else "article_heading_fragment"
                    rows[j]["article_id"] = article_id
                    current["lines"].append(rows[j])
                subdivisions = {}
                hierarchy_pending = None
                title_pending = True
                i = heading_indices[-1] + 1
                continue
            if re.match(r"^Article\b", row["text"], re.I) and row["bold"] and row["bbox"][0] / page["width"] <= layout["body_left_x_max_ratio"]:
                issue(issues, "unparsed_article_like_heading", line_id=row["id"], edition=year,
                      pdf_page=pdf_page, text=row["text"])
            level = full_hierarchy_level(row["text"])
            if level and row["bold"]:
                for key in FULL_LEVELS[FULL_LEVELS.index(level):]:
                    hierarchy.pop(key, None)
                hierarchy[level] = {"level": level, "label": row["text"], "title": "", "line_ids": [row["id"]]}
                row["role"] = "hierarchy_heading"
                hierarchy_pending = level
                title_pending = False
                i += 1
                continue
            if hierarchy_pending:
                if row["bold"] or row["text"].isupper():
                    h = hierarchy[hierarchy_pending]
                    h["title"] += (" " if h["title"] else "") + row["text"]
                    h["line_ids"].append(row["id"])
                    row["role"] = "hierarchy_heading"
                    i += 1
                    continue
                hierarchy_pending = None
            if not started:
                row["role"] = "preface"
                i += 1
                continue
            part = subdivision(row["text"])
            if title_pending and row["bold"] and not part:
                current["title"] += (" " if current["title"] else "") + row["text"]
                current["heading_line_ids"].append(row["id"])
                row["role"] = "article_heading_continuation"
            else:
                title_pending = False
                if part:
                    rank, part_label = part
                    subdivisions = {k: v for k, v in subdivisions.items() if k < rank}
                    subdivisions[rank] = part_label
            row["article_id"] = current["id"]
            row["subdivision_path"] = [subdivisions[k] for k in sorted(subdivisions)]
            current["lines"].append(row)
            i += 1
    numbers = [a["article_number"] for a in articles if a["article_suffix"] is None]
    expected = list(range(1, settings["last_article"] + 1))
    missing = sorted(set(expected) - set(numbers))
    extras = sorted(set(numbers) - set(expected))
    if missing:
        issue(issues, "missing_main_article_numbers", edition=year, numbers=missing)
    if extras:
        issue(issues, "out_of_range_article_numbers", edition=year, numbers=extras)
    if numbers != sorted(numbers):
        issue(issues, "main_article_order_break", edition=year)
    for article in articles:
        article["text"] = "\n".join(row["text"] for row in article["lines"])
        article["pdf_page_start"] = article["lines"][0]["pdf_page"]
        article["pdf_page_end"] = article["lines"][-1]["pdf_page"]
        article["printed_page_start"] = article["lines"][0]["printed_page"]
        article["printed_page_end"] = article["lines"][-1]["printed_page"]
        article["page_map"] = list({row["pdf_page"]: {"pdf_page": row["pdf_page"],
                                                           "printed_page": row["printed_page"]}
                                    for row in article["lines"]}.values())
        if not any(row["role"] == "main" for row in article["lines"]):
            code = "abrogated_heading_only" if "ABROGE" in folded(article["title"]) else "article_without_main_text"
            issue(issues, code, article_id=article["id"])
    notes = attach_notes(pages, articles, issues)
    for previous, page in zip(pages, pages[1:]):
        if previous["printed_page"] and page["printed_page"]:
            if int(page["printed_page"]) != int(previous["printed_page"]) + 1:
                issue(issues, "printed_pagination_discontinuity", edition=year,
                      pdf_page=page["pdf_page"], previous=previous["printed_page"],
                      current=page["printed_page"])
    return pages, articles, notes, issues, boundaries


def make_diagnostics(year: int, settings: dict, pages: list, articles: list, notes: list, issues: list) -> dict:
    main_numbers = [a["article_number"] for a in articles if a["article_suffix"] is None]
    unassigned = [r["id"] for p in pages for r in p["lines"]
                  if r["role"] == "main" and r["article_id"] is None]
    if unassigned:
        issue(issues, "unassigned_main_lines", edition=year, count=len(unassigned), samples=unassigned[:30])
    return {"edition": year, "status": "candidate_not_active", "body_pages": len(pages),
            "article_count": len(articles), "main_article_count": len(main_numbers),
            "first_article": articles[0]["article"] if articles else None,
            "last_article": articles[-1]["article"] if articles else None,
            "expected_last_main_article": settings["last_article"],
            "missing_main_numbers": sorted(set(range(1, settings["last_article"] + 1)) - set(main_numbers)),
            "duplicate_labels": [label for label, count in Counter(a["article"] for a in articles).items() if count > 1],
            "multi_page_articles": sum(a["pdf_page_end"] > a["pdf_page_start"] for a in articles),
            "unassigned_main_lines": unassigned[:100],
            "note_count": len(notes), "note_status_counts": dict(Counter(n["attachment_status"] for n in notes)),
            "issue_count": len(issues), "issue_counts": dict(Counter(x["code"] for x in issues)),
            "pages_without_text": [x["pdf_page"] for x in issues if x["code"] == "page_without_exploitable_text"],
            "no_indexing_blockers": bool(articles and not any(x["code"] in {
                "page_without_exploitable_text", "missing_main_article_numbers", "duplicate_article_label",
                "out_of_range_article_numbers", "main_article_order_break", "unparsed_article_like_heading",
                "article_without_main_text", "unassigned_main_lines"} for x in issues))}


def extract_full(output: Path) -> dict:
    import pymupdf

    output = candidate_path(output)
    audit = read_json(output / "source_audit.json")
    if audit["status"] != "verified_local_pdf_bytes" or audit["scope_sha256"] != sha256(CONFIG):
        raise ValueError("Audit des sources absent ou périmé.")
    config = read_json(CONFIG)
    layout = read_json(ROOT / config["layout_source"])["layout"]
    all_pages, all_articles, all_notes, all_issues, all_boundaries, diagnostics = [], [], [], [], [], []
    for audited in audit["editions"]:
        year = audited["edition"]
        settings = config["editions"][str(year)]
        source = audited["source"]
        if sha256(ROOT / source["file"]) != source["sha256"]:
            raise ValueError(f"Source modifiée depuis l'audit : {year}")
        with pymupdf.open(ROOT / source["file"]) as doc:
            pages, articles, notes, issues, boundaries = extract_edition(doc, year, settings, layout, source)
        diagnostic = make_diagnostics(year, settings, pages, articles, notes, issues)
        diagnostics.append(diagnostic)
        print(f"{year}: {len(articles)} articles, {len(notes)} notes, {len(issues)} alertes", flush=True)
        all_pages.extend(pages)
        all_articles.extend(articles)
        all_notes.extend(notes)
        all_issues.extend(issues)
        all_boundaries.extend(boundaries)
    output.mkdir(parents=True, exist_ok=True)
    files = {"pages.jsonl": all_pages, "articles.jsonl": all_articles,
             "notes.jsonl": all_notes, "issues.jsonl": all_issues, "boundaries.jsonl": all_boundaries}
    hashes = {}
    for name, rows in files.items():
        write_jsonl(output / name, rows)
        hashes[name] = sha256(output / name)
    report = {"status": "candidate_extracted", "schema_version": "1.0",
              "source_audit_sha256": sha256(output / "source_audit.json"),
              "config_sha256": sha256(CONFIG), "artifact_sha256": hashes,
              "editions": diagnostics, "article_count": len(all_articles), "note_count": len(all_notes),
              "issue_count": len(all_issues), "ready_for_candidate_index": all(d["no_indexing_blockers"] for d in diagnostics)}
    write_json(output / "extraction_diagnostics.json", report)
    return report


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "extract"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify_sources(args.output) if args.command == "audit" else extract_full(args.output)
    print(result["status"], args.output)


if __name__ == "__main__":
    main()
