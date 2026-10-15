"""Extraction bornée des articles 1 à 20 (intercalaires inclus), sur ordre explicite."""

from __future__ import annotations

import argparse
import copy
import html
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from .common import (ROOT, code_hashes, configure_console, new_run,
                     read_json, safe_write, sha256, source_provenance, write_json, write_jsonl)

ARTICLE = re.compile(
    r"^Article\s+(premier|\d+)(?:\s+(bis|ter|quater|quinquies))?\s*[.\-–—]+\s*(.*)$", re.I
)
NOTE = re.compile(r"^(\d{1,4})\s*(?=[A-Za-zÀ-ÿ(])")
LEVELS = ["livre", "partie", "titre", "chapitre", "section", "sous_section"]


def folded(text: str) -> str:
    value = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).upper()
    return " ".join(value.split())


def issue(issues: list, code: str, **context) -> None:
    issues.append({"code": code, **context})


def note_marker(row: dict) -> str | None:
    """Un millésime en début de continuation n'est pas un numéro de note."""
    spans = [span for span in row["spans"] if span["text"].strip()]
    if len(spans) < 2:
        return None
    first = spans[0]
    label = first["text"].strip()
    if not label.isdigit() or len(label) > 4:
        return None
    following = spans[1:]
    body_size = max(span["size"] for span in following)
    baseline = statistics.median(span["origin"][1] for span in following)
    if first["size"] < body_size * 0.85 and first["origin"][1] < baseline - 1:
        return label
    return None


def reading_order(row: dict) -> tuple:
    baselines = [s["origin"][1] for s in row["spans"]
                 if s["text"].strip() and abs(s["size"] - row["font_size"]) < 0.1]
    return round(statistics.median(baselines), 1), row["bbox"][0]


def page_lines(page, edition: int, pdf_page: int, layout: dict, issues: list) -> dict:
    """Conserve les spans d'origine et leurs coordonnées, sans suppression irréversible."""
    rows = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            spans = [{key: span[key] for key in ("text", "size", "font", "flags", "bbox", "origin")}
                     for span in line["spans"] if span["text"]]
            if not spans or not "".join(span["text"] for span in spans).strip():
                continue
            weights = Counter()
            for span in spans:
                weights[round(span["size"], 2)] += len(span["text"].strip())
            rows.append({"bbox": list(line["bbox"]), "spans": spans,
                         "raw_text": "".join(s["text"] for s in spans).strip(),
                         "font_size": weights.most_common(1)[0][0],
                         "bold": sum(len(s["text"].strip()) for s in spans if s["flags"] & 16)
                         > sum(len(s["text"].strip()) for s in spans) * 0.4})
    # Les puces Wingdings ont une bbox plus basse, mais la même ligne de base.
    rows.sort(key=reading_order)
    sizes = Counter()
    for row in rows:
        if 0.1 * page.rect.height < row["bbox"][1] < 0.8 * page.rect.height:
            sizes[row["font_size"]] += len(row["raw_text"])
    body_size = sizes.most_common(1)[0][0] if sizes else 0
    if not body_size or sum(len(row["raw_text"]) for row in rows) < 40:
        raise ValueError(f"Page non exploitable sans inspection/OCR : {edition}, PDF {pdf_page}")
    printed_candidates = []
    in_notes = False
    for index, row in enumerate(rows, 1):
        row.update(id=f"cgi-{edition}:p{pdf_page:04d}:l{index:03d}", pdf_page=pdf_page,
                   article_id=None, subdivision_path=[], note_refs=[])
        top = row["bbox"][1] / page.rect.height
        text = row["raw_text"]
        if top < layout["header_max_y_ratio"] and folded(text) == "CODE GENERAL DES IMPOTS":
            row["role"] = "header"
        elif top >= layout["printed_page_min_y_ratio"] and re.fullmatch(r"\d{1,4}", text):
            row["role"] = "printed_page"
            printed_candidates.append(text)
        elif (top >= layout["notes_min_y_ratio"]
              and row["font_size"] <= body_size * layout["notes_max_font_ratio"]):
            # La taille ET la zone sont nécessaires. Une note sans numéro reste visible.
            row["role"] = "note" if note_marker(row) or in_notes else "unclassified_small_text"
            in_notes = row["role"] == "note"
            if not in_notes:
                issue(issues, "small_text_not_classified", line_id=row["id"])
        else:
            row["role"] = "main"
            if in_notes:
                issue(issues, "body_after_notes", line_id=row["id"])
            in_notes = False
        # Retirer seulement les appels numériques reconnus comme exposants.
        baseline_spans = [s for s in row["spans"] if s["size"] >= body_size * 0.9]
        baseline = statistics.median(s["origin"][1] for s in baseline_spans) if baseline_spans else None
        clean = []
        for span in row["spans"]:
            marker = span["text"].strip()
            superscript = (span["size"] < body_size * 0.8 and baseline is not None
                           and (span["flags"] & 1 or span["origin"][1] < baseline - 1.5))
            if row["role"] == "main" and marker.isdigit() and superscript:
                row["note_refs"].append({"label": marker, "bbox": list(span["bbox"]),
                                         "detection": "numeric_superscript"})
            else:
                clean.append(span["text"])
        row["text"] = "".join(clean).strip()
        if "\ufffd" in text:
            issue(issues, "replacement_character", line_id=row["id"])
    printed = printed_candidates[0] if len(printed_candidates) == 1 else None
    if printed is None:
        issue(issues, "printed_page_ambiguous_or_missing", edition=edition, pdf_page=pdf_page,
              candidates=printed_candidates)
    for row in rows:
        row["printed_page"] = printed
    return {"edition": edition, "pdf_page": pdf_page, "printed_page": printed,
            "pdf_page_label": page.get_label(), "width": page.rect.width, "height": page.rect.height,
            "dominant_body_font_size": body_size, "lines": rows}


def hierarchy_level(text: str) -> str | None:
    value = folded(text)
    if re.match(r"^(?:PREMIERE|DEUXIEME|TROISIEME|[IVX]+)\s+PARTIE\b", value):
        return "partie"
    match = re.match(r"^(LIVRE|PARTIE|TITRE|CHAPITRE|SOUS[- ]SECTION|SECTION)\s+\S+", value)
    return match[1].lower().replace("-", "_").replace(" ", "_") if match else None


def subdivision(text: str) -> tuple[int, str] | None:
    # Préfixes seulement : les renvois dans les phrases ne créent pas de subdivision.
    intercalated = re.match(r"^(\d{1,2}\s*[°º])\s*[-–—]?\s*(bis|ter)\s*[.\-–—]+", text)
    if intercalated:
        return 2, intercalated[1].replace(" ", "").replace("º", "°") + " " + intercalated[2]
    patterns = [
        (0, r"^([IVX]+)\s*(?:\.\s*)?[-–—](?![A-Z0-9]+[-–—°])"),
        (1, r"^([A-Z])\s*(?:\.\s*)?[-–—](?![A-Z0-9]+[-–—°])"),
        (2, r"^(\d{1,2}\s*[°º])\s*[-–—]"),
        (3, r"^([a-z])\s*(?:\)|[.\-–—]+)\s*"),
        (4, r"^([1-9]\d?)\s*(?:\)|[.\-–—])\s+"),
    ]
    for rank, pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return rank, match[1].replace(" ", "").replace("º", "°")
    return None


def attach_notes(pages: list[dict], articles: list[dict], issues: list) -> list[dict]:
    notes = []
    by_article = {a["id"]: a for a in articles}
    for page in pages:
        page_articles = sorted({r["article_id"] for r in page["lines"] if r["article_id"]})
        current = None
        for row in page["lines"]:
            if row["role"] != "note":
                continue
            marker = note_marker(row)
            if marker or current is None:
                current = {"id": f"cgi-{page['edition']}:p{page['pdf_page']:04d}:note{len(notes) + 1}",
                           "edition": page["edition"], "label": marker,
                           "pdf_page": page["pdf_page"], "printed_page": page["printed_page"],
                           "lines": [], "text": "", "links": [], "candidate_article_ids": page_articles,
                           "article_id": None, "attachment_status": "unresolved"}
                notes.append(current)
            current["lines"].append(row)
            current["text"] += ("\n" if current["text"] else "") + row["raw_text"]
        page_notes = [n for n in notes if n["pdf_page"] == page["pdf_page"]]
        for note in page_notes:
            links = [{"article_id": row["article_id"], "line_id": row["id"],
                      "subdivision_path": row["subdivision_path"], "reference_bbox": ref["bbox"]}
                     for row in page["lines"] if row["article_id"]
                     for ref in row["note_refs"] if ref["label"] == note["label"]]
            unique_label = sum(n["label"] == note["label"] for n in page_notes) == 1
            note["links"] = links
            owners = {link["article_id"] for link in links}
            if len(owners) == 1 and unique_label:
                note["article_id"] = next(iter(owners))
                note["attachment_status"] = "identified_by_page_and_superscript"
                by_article[note["article_id"]]["note_ids"].append(note["id"])
            elif not page_articles:
                note["attachment_status"] = "outside_scope_or_unresolved"
            else:
                issue(issues, "note_attachment_unresolved", note_id=note["id"], candidates=page_articles)
        for row in page["lines"]:
            if row["article_id"]:
                for ref in row["note_refs"]:
                    if not any(n["label"] == ref["label"] for n in page_notes):
                        issue(issues, "note_reference_without_same_page_note", line_id=row["id"], label=ref["label"])
    return notes


def extract_edition(doc, edition: int, config: dict, layout: dict, source: dict) -> tuple:
    pages, articles, issues, boundaries = [], [], [], []
    hierarchy, subdivisions = {}, {}
    hierarchy_pending = None
    title_pending = False
    current = None
    started = stopped = False
    for page_number in range(config["start_pdf_page"], config["stop_scan_pdf_page"] + 1):
        page = page_lines(doc[page_number - 1], edition, page_number, layout, issues)
        pages.append(page)
        for row in page["lines"]:
            if row["role"] != "main":
                continue
            text = row["text"]
            match = ARTICLE.match(text)
            # Un titre d'article doit être en gras et commencer à gauche.
            is_article = (match is not None and row["bold"]
                          and row["bbox"][0] / page["width"] <= layout["body_left_x_max_ratio"])
            if match and not is_article:
                issue(issues, "article_like_line_not_a_heading", line_id=row["id"], text=text)
            if is_article:
                base = 1 if match[1].lower() == "premier" else int(match[1])
                label = str(base) + (" " + match[2].lower() if match[2] else "")
                if not started:
                    if label != "1":
                        row["role"] = "preface"
                        continue
                    if page_number != config["start_pdf_page"]:
                        raise ValueError("L'article premier n'est pas sur la page attendue.")
                    context = " ".join(h["label"] + " " + h["title"] for h in hierarchy.values())
                    if not all(key in hierarchy for key in ("livre", "partie", "titre", "chapitre")):
                        raise ValueError("Hiérarchie initiale incomplète : vérifier la lecture du PDF.")
                    if "SOCIETES" not in folded(context):
                        raise ValueError("L'article premier n'appartient pas au titre attendu sur l'IS.")
                    started = True
                boundaries.append({"edition": edition, "article": label, "line_id": row["id"],
                                   "pdf_page": page_number, "printed_page": page["printed_page"],
                                   "raw_text": row["raw_text"]})
                if base == 21 and not match[2]:
                    row["role"] = "scope_end_boundary"
                    stopped = True
                    break
                if not 1 <= base <= 20:
                    raise ValueError(f"Article inattendu dans le périmètre : {label}")
                current = {"id": f"cgi-{edition}:art-{label.replace(' ', '-')}", "edition": edition,
                           "article": label, "article_number": base, "article_suffix": match[2],
                           "title": match[3], "heading_line_ids": [row["id"]],
                           "hierarchy": [copy.deepcopy(hierarchy[k]) for k in LEVELS if k in hierarchy],
                           "source": source, "lines": [], "note_ids": [], "verification_status": "pending"}
                articles.append(current)
                row["role"] = "article_heading"
                row["article_id"] = current["id"]
                current["lines"].append(row)
                subdivisions = {}
                hierarchy_pending = None
                title_pending = True
                continue
            level = hierarchy_level(text)
            if level and row["bold"]:
                for key in LEVELS[LEVELS.index(level):]:
                    hierarchy.pop(key, None)
                hierarchy[level] = {"level": level, "label": text, "title": "", "line_ids": [row["id"]]}
                row["role"] = "hierarchy_heading"
                hierarchy_pending = level
                title_pending = False
                continue
            if hierarchy_pending:
                if row["bold"] or text.isupper():
                    heading = hierarchy[hierarchy_pending]
                    heading["title"] += (" " if heading["title"] else "") + text
                    heading["line_ids"].append(row["id"])
                    row["role"] = "hierarchy_heading"
                    continue
                hierarchy_pending = None
            if not started:
                row["role"] = "preface"
                continue
            part = subdivision(text)
            if title_pending and row["bold"] and not part:
                current["title"] += " " + text
                current["heading_line_ids"].append(row["id"])
                row["role"] = "article_heading_continuation"
            else:
                title_pending = False
                if part:
                    rank, label = part
                    subdivisions = {k: v for k, v in subdivisions.items() if k < rank}
                    subdivisions[rank] = label
            row["article_id"] = current["id"]
            row["subdivision_path"] = [subdivisions[k] for k in sorted(subdivisions)]
            current["lines"].append(row)
        if stopped:
            # Le reste de cette page sert uniquement à contrôler la borne d'arrêt.
            for row in page["lines"]:
                if row["role"] == "main" and row["article_id"] is None:
                    row["role"] = "outside_scope"
            break
    if not started or not stopped:
        raise ValueError("Bornes non confirmées : l'extraction ne doit jamais être tronquée silencieusement.")
    labels = [article["article"] for article in articles]
    if len(labels) != len(set(labels)):
        raise ValueError("Titres d'articles dupliqués : possible sommaire ou erreur de segmentation.")
    if [a["article_number"] for a in articles if a["article_suffix"] is None] != list(range(1, 21)):
        raise ValueError("Les articles principaux 1 à 20 ne sont pas présents exactement une fois et dans l'ordre.")
    if [a["article_number"] for a in articles] != sorted(a["article_number"] for a in articles):
        raise ValueError("Ordre d'articles incohérent.")
    if pages[0]["printed_page"] != config["observed_start_printed_page"]:
        raise ValueError("Pagination imprimée initiale inattendue.")
    for previous, page in zip(pages, pages[1:]):
        if previous["printed_page"] and page["printed_page"]:
            if int(page["printed_page"]) != int(previous["printed_page"]) + 1:
                issue(issues, "printed_pagination_discontinuity", edition=edition, pdf_page=page["pdf_page"])
    for article in articles:
        article["text"] = "\n".join(row["text"] for row in article["lines"])
        article["pdf_page_start"] = article["lines"][0]["pdf_page"]
        article["pdf_page_end"] = article["lines"][-1]["pdf_page"]
        article["printed_page_start"] = article["lines"][0]["printed_page"]
        article["printed_page_end"] = article["lines"][-1]["printed_page"]
        article["page_map"] = list({row["pdf_page"]: {"pdf_page": row["pdf_page"],
                                                       "printed_page": row["printed_page"]}
                                    for row in article["lines"]}.values())
    notes = attach_notes(pages, articles, issues)
    return pages, articles, notes, issues, boundaries


def prepare_review(output: Path, articles: list[dict], pages: list[dict], notes: list[dict]) -> None:
    samples = []
    for edition in sorted({a["edition"] for a in articles}):
        group = [a for a in articles if a["edition"] == edition]
        for reason, article in [("court", min(group, key=lambda a: len(a["text"]))),
                                ("long", max(group, key=lambda a: len(a["text"]))),
                                ("plusieurs_pages", next(a for a in group if len(a["page_map"]) > 1))]:
            samples.append({"reason": reason, "article_id": article["id"], "edition": edition,
                            "pdf_pages": [p["pdf_page"] for p in article["page_map"]]})
    write_json(output / "review_checklist.json", {
        "schema_version": "1.0", "all_issues_reviewed": False, "issue_resolution_notes": None,
        "samples": samples,
        "articles": [{"article_id": a["id"], "status": "pending", "reviewer": None, "reviewed_at": None,
                      "checks": {"boundaries": False, "main_text": False, "title_hierarchy_subdivisions": False,
                                 "notes_and_links": False, "pagination": False}, "comments": None}
                     for a in articles]})
    esc = html.escape
    chunks = ["<!doctype html><html lang='fr'><meta charset='utf-8'><title>Revue CGI à effectuer</title>",
              "<style>body{max-width:1100px;margin:2em auto;font:16px system-ui}pre{white-space:pre-wrap}"
              "td,th{padding:.3em;border:1px solid #bbb}table{border-collapse:collapse}summary{cursor:pointer}</style>",
              "<h1>Extraction provisoire : revue manuelle requise</h1>",
              "<p>Les notes et les alertes ne sont pas des preuves validées. Comparer chaque article au PDF.</p>"]
    for article in articles:
        link = (ROOT / article["source"]["file"]).as_uri() + f"#page={article['pdf_page_start']}"
        chunks += [f"<h2>{esc(article['id'])} — {esc(article['title'])}</h2>",
                   f"<a href='{esc(link, quote=True)}'>Ouvrir le PDF à la page {article['pdf_page_start']}</a>",
                   f"<pre>{esc(str(article['page_map']))}\n{esc(str(article['hierarchy']))}</pre>", "<table>"]
        for row in article["lines"]:
            chunks.append(f"<tr><td>{esc(row['id'])}<br>{esc('/'.join(row['subdivision_path']))}</td>"
                          f"<td>{esc(row['text'])}</td></tr>")
        chunks.append("</table><h3>Notes rattachées ou candidates</h3>")
        for note in notes:
            if note["article_id"] == article["id"] or article["id"] in note["candidate_article_ids"]:
                chunks.append(f"<pre>{esc(note['id'])} [{esc(note['attachment_status'])}]\n{esc(note['text'])}</pre>")
    chunks.append("<h2>Audit des pages : toutes les lignes et leur classement</h2>")
    for page in pages:
        chunks.append(f"<details><summary>{page['edition']} PDF {page['pdf_page']} / imprimée {page['printed_page']}</summary><pre>")
        chunks.append(esc("\n".join(f"{r['id']} [{r['role']}] {r['raw_text']}" for r in page["lines"])))
        chunks.append("</pre></details>")
    chunks.append("</html>")
    safe_write(output / "review.html", "\n".join(chunks))


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editions", type=int, nargs="+", choices=[2023, 2025], default=[2023, 2025])
    args = parser.parse_args()
    try:
        import pymupdf
    except ImportError as exc:
        raise SystemExit("PyMuPDF manque. Consulter requirements.txt ; aucune installation automatique.") from exc
    config = read_json(ROOT / "config/extraction_scope.json")
    if config["scope"] != {"first_article": 1, "last_article": 20,
                           "include_intercalated_articles": True, "stop_before_article": "21"}:
        raise ValueError("Ce parser est volontairement limité aux articles 1 à 20, intercalaires inclus.")
    editions = sorted(set(args.editions))
    output = new_run("extractions")
    metadata = {"schema_version": "1.0", "status": "started", "structural_checks_passed": False,
                "editions": editions, "config": config, "config_sha256": sha256(ROOT / "config/extraction_scope.json"),
                "code_sha256": code_hashes(), "engine": "PyMuPDF", "engine_version": pymupdf.VersionBind,
                "sources": [], "artifact_sha256": {}}
    write_json(output / "run.json", metadata)
    all_pages, all_articles, all_notes, all_issues, all_boundaries = [], [], [], [], []
    try:
        for edition in editions:
            settings = config["editions"][str(edition)]
            source = source_provenance(edition, settings)
            metadata["sources"].append(source)
            with pymupdf.open(ROOT / settings["file"]) as doc:
                if doc.needs_pass or doc.is_repaired or doc.page_count != settings["expected_pdf_pages"]:
                    raise ValueError(f"PDF chiffré, réparé ou nombre de pages inattendu : {edition}")
                cover = doc[settings["cover_pdf_page"] - 1].get_text()
                if not re.search(rf"\b{edition}\b", cover) or "IMPOTS" not in folded(cover):
                    raise ValueError(f"Édition non confirmée en page de titre : {edition}")
                results = extract_edition(doc, edition, settings, config["layout"], source)
                for target, rows in zip((all_pages, all_articles, all_notes, all_issues, all_boundaries), results):
                    target.extend(rows)
        for filename, rows in [("pages.jsonl", all_pages), ("articles.jsonl", all_articles),
                               ("notes.jsonl", all_notes), ("issues.jsonl", all_issues), ("boundaries.jsonl", all_boundaries)]:
            write_jsonl(output / filename, rows)
            metadata["artifact_sha256"][filename] = sha256(output / filename)
        prepare_review(output, all_articles, all_pages, all_notes)
        metadata.update(status="extracted_pending_manual_review", structural_checks_passed=True,
                        article_count=len(all_articles), issue_count=len(all_issues))
        write_json(output / "run.json", metadata)
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", metadata)
        raise
    print(output)


if __name__ == "__main__":
    main()
