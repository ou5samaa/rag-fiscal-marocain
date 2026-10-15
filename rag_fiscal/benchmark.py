"""Valider la structure des annotations et la présence textuelle des preuves, sans retrieval."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .common import ROOT, configure_console, read_json, reviewed_articles


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate(data: dict, articles: list[dict]) -> list[str]:
    errors = []
    questions = data.get("questions", [])
    if not questions:
        return ["Aucune question annotée."]
    if data.get("status") != "annotated":
        errors.append("Le benchmark n'est pas déclaré annotated après sa revue.")
    ids, families = set(), {}
    corpus = {(a["edition"], a["article"]): a for a in articles}
    for q in questions:
        name = q.get("id")
        if not isinstance(name, str) or not name or name in ids:
            errors.append(f"Identifiant absent/dupliqué : {name}")
        ids.add(name)
        required = ("question", "expected_answer", "family_id", "reviewer", "reviewed_at")
        if q.get("status") != "verified" or any(not isinstance(q.get(k), str) or not q[k].strip() for k in required):
            errors.append(f"{name} : annotation/revue incomplète.")
        if q.get("edition") not in {2023, 2025}:
            errors.append(f"{name} : édition invalide.")
        if q.get("split") not in {"dev", "test"}:
            errors.append(f"{name} : split attendu dev ou test.")
        if q.get("change_status") not in {"modified", "unchanged"}:
            errors.append(f"{name} : comparaison entre éditions non annotée.")
        family = q.get("family_id")
        if isinstance(family, str) and family:
            if family in families and families[family] != q.get("split"):
                errors.append(f"{name} : fuite dev/test pour la famille {family}.")
            families[family] = q.get("split")
        evidence = q.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{name} : preuves absentes.")
            continue
        for proof in evidence:
            if not isinstance(proof, dict):
                errors.append(f"{name} : preuve invalide.")
                continue
            edition, label = proof.get("edition"), proof.get("article")
            if not isinstance(label, str) or edition not in {2023, 2025}:
                errors.append(f"{name} : identifiant de preuve invalide.")
                continue
            article = corpus.get((edition, label))
            if edition != q.get("edition") or article is None:
                errors.append(f"{name} : preuve hors édition/article du corpus.")
                continue
            digest = proof.get("source_sha256")
            if not isinstance(digest, str) or digest.lower() != article["source"]["sha256"].lower():
                errors.append(f"{name} : empreinte de source incorrecte.")
            path = proof.get("subdivision_path")
            if not isinstance(path, list) or not all(isinstance(p, str) for p in path):
                errors.append(f"{name} : subdivision_path doit être une liste de libellés.")
                continue
            pdf_pages = proof.get("pdf_pages")
            if not isinstance(pdf_pages, list) or not pdf_pages or not all(type(p) is int for p in pdf_pages):
                errors.append(f"{name} : pages PDF manquantes/invalides.")
                continue
            rows = [r for r in article["lines"] if r["role"] == "main"
                    and r["subdivision_path"][:len(path)] == path and r["pdf_page"] in pdf_pages]
            quote = proof.get("quote")
            # Exige une occurrence dans des lignes contiguës, même subdivision ou descendants.
            runs, current = [], []
            selected = {r["id"] for r in rows}
            for row in article["lines"]:
                if row["id"] in selected:
                    current.append(row["text"])
                elif current:
                    runs.append(normalized(" ".join(current)))
                    current = []
            if current:
                runs.append(normalized(" ".join(current)))
            if not isinstance(quote, str) or not quote.strip() or not any(normalized(quote) in run for run in runs):
                errors.append(f"{name} : citation introuvable dans la subdivision et les pages indiquées.")
            available = {r["pdf_page"]: r["printed_page"] for r in article["lines"]}
            if not set(pdf_pages).issubset(available):
                errors.append(f"{name} : page hors article.")
            if proof.get("printed_pages") != [available.get(p) for p in pdf_pages]:
                errors.append(f"{name} : correspondance pages PDF/imprimées incorrecte.")
    return errors


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "benchmark/questions.json")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--allow-agent-review", action="store_true")
    args = parser.parse_args()
    _, articles = reviewed_articles(args.run, allow_agent_review=args.allow_agent_review)
    errors = validate(read_json(args.questions), articles)
    if errors:
        raise SystemExit("Annotations non prêtes :\n- " + "\n- ".join(errors))
    print("Structure et citations textuelles contrôlées. Le soutien de la réponse par les preuves reste une vérification humaine.")


if __name__ == "__main__":
    main()
