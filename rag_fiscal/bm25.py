"""Baseline BM25 locale, sans réseau, après revue du corpus. Aucune évaluation implicite."""

from __future__ import annotations

import argparse
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .common import (ROOT, code_hashes, configure_console, inside_project, new_run,
                     read_json, reviewed_articles, sha256, write_json, write_jsonl)

TOKENIZER_VERSION = "nfkd_lower_alnum_v1"


def tokenize(text: str) -> list[str]:
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def make_passages(articles: list[dict], max_words: int, overlap_lines: int) -> list[dict]:
    """Fenêtres de lignes à l'intérieur d'une subdivision ; jamais entre deux éditions/articles."""
    passages = []
    for article in articles:
        groups = []
        for row in article["lines"]:
            if row["role"] != "main":
                continue
            if not groups or groups[-1][-1]["subdivision_path"] != row["subdivision_path"]:
                groups.append([])
            groups[-1].append(row)
        for group in groups:
            start = 0
            while start < len(group):
                end, count = start, 0
                while end < len(group):
                    words = len(tokenize(group[end]["text"]))
                    if end > start and count + words > max_words:
                        break
                    count += words
                    end += 1
                rows = group[start:end]
                text = "\n".join(row["text"] for row in rows)
                passage_id = f"{article['id']}:passage-{sum(p['article_id'] == article['id'] for p in passages) + 1:04d}"
                passages.append({"id": passage_id, "article_id": article["id"], "edition": article["edition"],
                                 "article": article["article"], "title": article["title"],
                                 "hierarchy": article["hierarchy"], "subdivision_path": rows[0]["subdivision_path"],
                                 "text": text, "line_ids": [r["id"] for r in rows],
                                 "pdf_page_start": rows[0]["pdf_page"], "pdf_page_end": rows[-1]["pdf_page"],
                                 "printed_page_start": rows[0]["printed_page"], "printed_page_end": rows[-1]["printed_page"],
                                 "page_map": list({r["pdf_page"]: {"pdf_page": r["pdf_page"], "printed_page": r["printed_page"]}
                                                   for r in rows}.values()), "source": article["source"]})
                if end == len(group):
                    break
                start = max(start + 1, end - overlap_lines)
    return passages


def build(args) -> None:
    source_run = inside_project(args.run)
    meta, articles = reviewed_articles(source_run, allow_agent_review=args.allow_agent_review)
    technical = read_json(source_run / "review_checklist.json").get("review_kind") == "agent_assisted_technical"
    if set(meta["editions"]) != {2023, 2025}:
        raise ValueError("Cette baseline compare le corpus 2023 ET 2025 ; extraire et revoir les deux éditions.")
    if args.max_words < 30 or not 0 <= args.overlap_lines <= 5:
        raise ValueError("max-words >= 30 et overlap-lines entre 0 et 5 requis.")
    passages = make_passages(articles, args.max_words, args.overlap_lines)
    if not passages:
        raise ValueError("Aucun passage indexable.")
    postings = defaultdict(list)
    lengths = []
    for index, passage in enumerate(passages):
        # Même titre ajouté à chaque passage, sans injecter l'édition dans les tokens.
        terms = tokenize(f"Article {passage['article']} {passage['title']}\n{passage['text']}")
        lengths.append(len(terms))
        for term, frequency in Counter(terms).items():
            postings[term].append([index, frequency])
    output = new_run("indexes")
    write_jsonl(output / "passages.jsonl", passages)
    payload = {"schema_version": "1.0", "status": "built_from_technically_reviewed_corpus" if technical else "built_from_manually_reviewed_corpus",
               "review_kind": "agent_assisted_technical" if technical else "human",
               "human_review_status": "pending" if technical else "declared_completed",
               "method": "BM25_log1p_IDF", "k1": 1.5, "b": 0.75, "tokenizer": TOKENIZER_VERSION,
               "corpus_run": str(source_run.relative_to(ROOT)), "corpus_run_sha256": sha256(source_run / "run.json"),
               "review_sha256": sha256(source_run / "review_checklist.json"), "code_sha256": code_hashes(),
               "passages_sha256": sha256(output / "passages.jsonl"),
               "chunking": {"max_words": args.max_words, "overlap_lines": args.overlap_lines},
               "lengths": lengths, "average_length": sum(lengths) / len(lengths),
               "postings": dict(postings), "documents": passages}
    write_json(output / "index.json", payload)
    print(output / "index.json")


def search(args) -> None:
    started = time.perf_counter()
    path = inside_project(args.index)
    index = read_json(path)
    if index["tokenizer"] != TOKENIZER_VERSION:
        raise ValueError("Version du tokenizer incompatible.")
    if index["status"] not in {"built_from_manually_reviewed_corpus", "built_from_technically_reviewed_corpus"}:
        raise ValueError("Index non validé.")
    run = ROOT / index["corpus_run"]
    reviewed_articles(run, allow_agent_review=index.get("review_kind") == "agent_assisted_technical")
    if (sha256(run / "run.json") != index["corpus_run_sha256"]
            or sha256(run / "review_checklist.json") != index["review_sha256"]):
        raise ValueError("Corpus/revue modifié depuis l'indexation : reconstruire l'index.")
    if sha256(path.parent / "passages.jsonl") != index["passages_sha256"]:
        raise ValueError("Passages modifiés depuis l'indexation.")
    if not args.query.strip() or args.top_k < 1:
        raise ValueError("Question non vide et top-k positif requis.")
    loaded = time.perf_counter()
    documents = index["documents"]
    eligible = {i for i, doc in enumerate(documents) if args.edition_filter == "off" or doc["edition"] == args.edition}
    scores = defaultdict(float)
    n = len(documents)
    for term in set(tokenize(args.query)):
        hits = index["postings"].get(term, [])
        idf = math.log(1 + (n - len(hits) + 0.5) / (len(hits) + 0.5))
        for doc_id, frequency in hits:
            if doc_id not in eligible:
                continue
            length_factor = 1 - index["b"] + index["b"] * index["lengths"][doc_id] / index["average_length"]
            scores[doc_id] += idf * frequency * (index["k1"] + 1) / (frequency + index["k1"] * length_factor)
    ranked = sorted(scores, key=lambda i: (-scores[i], documents[i]["id"]))[:args.top_k]
    results = []
    for rank, doc_id in enumerate(ranked, 1):
        doc = documents[doc_id]
        results.append({"rank": rank, "score": scores[doc_id], "matches_requested_edition": doc["edition"] == args.edition,
                        **doc})
    finished = time.perf_counter()
    payload = {"query": args.query, "requested_edition": args.edition, "edition_filter": args.edition_filter,
               "review_kind": index.get("review_kind", "human"), "human_review_status": index.get("human_review_status"),
               "index_sha256": sha256(path), "retrieval_ms": (finished - loaded) * 1000,
               "load_and_integrity_ms": (loaded - started) * 1000, "eligible_passages": len(eligible),
               "results": results, "status": "retrieval_only_no_answer_or_evaluation"}
    output = new_run("searches")
    write_json(output / "results.json", payload)
    print(output / "results.json")


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--run", type=Path, required=True)
    builder.add_argument("--max-words", type=int, default=220)
    builder.add_argument("--overlap-lines", type=int, default=2)
    builder.add_argument("--allow-agent-review", action="store_true", help="Autoriser explicitement un corpus revu techniquement par IA pour une baseline expérimentale")
    query = commands.add_parser("search")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--edition", type=int, choices=[2023, 2025], required=True)
    query.add_argument("--edition-filter", choices=["on", "off"], default="on")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    {"build": build, "search": search}[args.command](args)


if __name__ == "__main__":
    main()
