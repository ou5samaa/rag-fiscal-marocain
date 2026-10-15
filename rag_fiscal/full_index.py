"""Index BM25, vecteurs E5 et Chroma pour le corpus candidat des quatre CGI.

La configuration active data/current_run.json n'est jamais modifiée. Les
comparaisons Chroma sont des contrôles techniques, pas un nouveau benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .bm25 import TOKENIZER_VERSION, make_passages, tokenize
from .common import ROOT, configure_console, read_json, read_jsonl, safe_write, sha256, write_json, write_jsonl
from .compare_retrievers import bm25_scores, model_files, rank_scores, rrf_scores
from .full_corpus import CONFIG, DEFAULT_OUTPUT, candidate_path


def verified_extraction(output: Path):
    output = candidate_path(output)
    report = read_json(output / "extraction_diagnostics.json")
    if not report["ready_for_candidate_index"] or report["config_sha256"] != sha256(CONFIG):
        raise ValueError("Corpus candidat non complet ou diagnostic périmé.")
    if report["source_audit_sha256"] != sha256(output / "source_audit.json"):
        raise ValueError("Audit des sources modifié.")
    for name, digest in report["artifact_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Extraction modifiée : {name}")
    for entry in read_json(output / "source_audit.json")["editions"]:
        source = entry["source"]
        if sha256(ROOT / source["file"]) != source["sha256"]:
            raise ValueError("PDF modifié depuis l'extraction.")
    return report, read_jsonl(output / "articles.jsonl")


def _same_chunk(p: dict, q: dict) -> bool:
    return all(p[k] == q[k] for k in ("edition", "article", "title", "subdivision_path", "text",
                                      "pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end"))


def verify_old_chunking(passages: list[dict]) -> dict:
    old = read_json(ROOT / "data/index/index.json")
    current = read_json(CONFIG)["chunking"]
    if current != old["chunking"]:
        raise ValueError("Le découpage candidat diffère de la configuration active.")
    previous = [p for p in old["documents"] if p["edition"] in {2023, 2025}]
    subset = [p for p in passages if p["edition"] in {2023, 2025} and int(p["article"].split()[0]) <= 20]
    if len(previous) != len(subset) or any(not _same_chunk(p, q) for p, q in zip(previous, subset)):
        raise ValueError("Régression du découpage sur la tranche 2023/2025 déjà indexée.")
    return {"active_passages_compared": len(previous), "same_chunking_and_text": True,
            "active_chunking": current}


def build_bm25(output: Path) -> dict:
    output = candidate_path(output)
    extraction, articles = verified_extraction(output)
    chunking = read_json(CONFIG)["chunking"]
    passages = make_passages(articles, **chunking)
    if not passages or len({p["id"] for p in passages}) != len(passages):
        raise ValueError("Passages absents ou identifiants dupliqués.")
    regression = verify_old_chunking(passages)
    postings = defaultdict(list)
    lengths = []
    for position, passage in enumerate(passages):
        terms = tokenize(f"Article {passage['article']} {passage['title']}\n{passage['text']}")
        lengths.append(len(terms))
        for term, frequency in Counter(terms).items():
            postings[term].append([position, frequency])
    folder = output / "index"
    write_jsonl(folder / "passages.jsonl", passages)
    payload = {"schema_version": "1.0", "status": "candidate_full_corpus_not_active",
               "method": "BM25_log1p_IDF", "k1": 1.5, "b": 0.75, "tokenizer": TOKENIZER_VERSION,
               "extraction_diagnostics_sha256": sha256(output / "extraction_diagnostics.json"),
               "passages_sha256": sha256(folder / "passages.jsonl"),
               "chunking": chunking, "lengths": lengths,
               "average_length": sum(lengths) / len(lengths),
               "postings": dict(postings), "documents": passages}
    write_json(folder / "index.json", payload)
    result = {"status": "candidate_bm25_built", "index_sha256": sha256(folder / "index.json"),
              "passages_sha256": payload["passages_sha256"], "documents": len(passages),
              "by_edition": dict(Counter(str(p["edition"]) for p in passages)),
              "vocabulary_size": len(postings), "chunking": chunking,
              "regression": regression, "source_articles": extraction["article_count"]}
    write_json(output / "index_summary.json", result)
    return result


def verified_index(output: Path):
    output = candidate_path(output)
    summary = read_json(output / "index_summary.json")
    path = output / "index/index.json"
    if (summary["status"] != "candidate_bm25_built"
            or sha256(path) != summary["index_sha256"]
            or sha256(output / "index/passages.jsonl") != summary["passages_sha256"]):
        raise ValueError("Index candidat absent ou modifié.")
    index = read_json(path)
    if index["chunking"] != read_json(CONFIG)["chunking"]:
        raise ValueError("Découpage candidat modifié.")
    return summary, index


def _local_model(device: str = "cpu"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    protocol = read_json(ROOT / "data/search_reference/protocol.json")
    path = ROOT / protocol["model_dir"]
    if model_files(path) != protocol["model_files"]:
        raise ValueError("Empreintes du modèle E5 local différentes de la référence.")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponible pour le calcul GPU régulé.")
    model = SentenceTransformer(str(path), device=device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 512
    model.eval()
    return model, protocol


def _gpu_memory_percent() -> float:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    used, total = (int(x.strip()) for x in result.stdout.splitlines()[0].split(",")[:2])
    return 100 * used / total


def _enable_cuda_runtime(output: Path) -> None:
    cuda_runtime = output / "cuda_runtime"
    if (cuda_runtime / "torch" / "__init__.py").is_file():
        sys.path.insert(0, str(cuda_runtime))
        return
    import torch
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise ValueError("Installer une version CUDA de PyTorch compatible avec ce GPU.")


def build_embeddings(output: Path, device: str = "cpu") -> dict:
    output = candidate_path(output)
    if device == "cuda_paced":
        _enable_cuda_runtime(output)
    import numpy as np
    import torch

    summary, index = verified_index(output)
    if device not in {"cpu", "cuda_paced"}:
        raise ValueError("Appareil E5 inconnu.")
    model, protocol = _local_model("cuda" if device == "cuda_paced" else "cpu")
    docs = index["documents"]
    inputs = [f"passage: Article {d['article']} {d['title']}\n{d['text']}" for d in docs]
    lengths = []
    for start in range(0, len(inputs), 128):
        lengths.extend(len(ids) for ids in model.tokenizer(inputs[start:start + 128], truncation=False)["input_ids"])
    over = []
    for i, count in enumerate(lengths):
        if count > 512:
            retained = model.tokenizer(inputs[i], truncation=True, max_length=512)["input_ids"]
            if len(retained) != 512:
                raise ValueError("Politique de borne E5 incohérente.")
            over.append({"passage_id": docs[i]["id"], "tokens": count,
                         "tokens_removed_from_model_input": count - 512,
                         "full_input_sha256": hashlib.sha256(inputs[i].encode("utf-8")).hexdigest(),
                         "retained_token_ids_sha256": hashlib.sha256(
                             bytes().join(int(t).to_bytes(4, "little") for t in retained)).hexdigest()})
    token_audit = {"model_limit": 512, "input_format": "passage: Article {article} {title}\\n{text}",
                   "passages": len(docs), "max_tokens": max(lengths),
                   "over_limit_count": len(over), "over_limit": over,
                   "policy": "full_passage_and_bm25_unchanged; explicit_model_limit_512_for_listed_inputs"}
    write_json(output / "index/token_audit.json", token_audit)
    started = time.perf_counter()
    parts = []
    peak_vram = 0.0
    chunk_size = 128
    part_folder = output / ("index/embedding_parts_cuda" if device == "cuda_paced"
                            else "index/embedding_parts")
    for start in range(0, len(inputs), chunk_size):
        end = min(start + chunk_size, len(inputs))
        part_path = part_folder / f"{start:06d}_{end:06d}.npy"
        if part_path.exists():
            chunk = np.load(part_path, allow_pickle=False)
            if chunk.shape != (end - start, 384) or not np.isfinite(chunk).all():
                raise ValueError(f"Lot E5 existant invalide : {part_path.name}")
        else:
            if device == "cuda_paced":
                before = _gpu_memory_percent()
                if before > 75:
                    raise ValueError(f"VRAM déjà occupée à {before:.1f} % ; attendre avant reprise.")
            chunk = model.encode(inputs[start:end], batch_size=8 if device == "cuda_paced" else 16,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)
            if device == "cuda_paced":
                torch.cuda.synchronize()
                after = _gpu_memory_percent()
                peak_vram = max(peak_vram, before, after)
                if after >= 90:
                    raise ValueError(f"VRAM observée à {after:.1f} % ; calcul arrêté.")
                time.sleep(0.4)
            if chunk.shape != (end - start, 384) or not np.isfinite(chunk).all():
                raise ValueError("Lot E5 invalide.")
            buf = io.BytesIO()
            np.save(buf, chunk, allow_pickle=False)
            safe_write(part_path, buf.getvalue())
        parts.append(chunk)
        if (start // chunk_size) % 8 == 0 or end == len(inputs):
            print(f"E5 {end}/{len(inputs)} passages ; VRAM échantillonnée max {peak_vram:.1f} %", flush=True)
        write_json(output / "index/embedding_progress.json", {
            "index_sha256": summary["index_sha256"], "completed": end,
            "total": len(inputs), "device": device, "sampled_peak_vram_percent": peak_vram})
    vectors = np.concatenate(parts, axis=0)
    if vectors.shape != (len(docs), 384) or not np.isfinite(vectors).all():
        raise ValueError("Vecteurs E5 invalides.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Normalisation E5 incorrecte.")
    buf = io.BytesIO()
    np.save(buf, vectors, allow_pickle=False)
    safe_write(output / "index/passage_embeddings.npy", buf.getvalue())
    report = {"status": "candidate_e5_built", "index_sha256": summary["index_sha256"],
              "vectors_sha256": sha256(output / "index/passage_embeddings.npy"),
              "token_audit_sha256": sha256(output / "index/token_audit.json"),
              "model_id": protocol["model_id"], "model_revision": protocol["model_revision"],
              "model_dir": protocol["model_dir"], "device": device,
              "batch_size": 8 if device == "cuda_paced" else 16,
              "sampled_peak_vram_percent": peak_vram,
              "embedding_seconds": time.perf_counter() - started, "documents": len(docs)}
    write_json(output / "index/vector_summary.json", report)
    return report


def verified_vectors(output: Path):
    import numpy as np

    summary, index = verified_index(output)
    report = read_json(output / "index/vector_summary.json")
    path = output / "index/passage_embeddings.npy"
    if (report["status"] != "candidate_e5_built"
            or report["index_sha256"] != summary["index_sha256"]
            or sha256(path) != report["vectors_sha256"]
            or sha256(output / "index/token_audit.json") != report["token_audit_sha256"]):
        raise ValueError("Vecteurs candidats absents ou modifiés.")
    vectors = np.load(path, allow_pickle=False)
    if vectors.shape != (len(index["documents"]), 384):
        raise ValueError("Dimensions des vecteurs incorrectes.")
    return summary, index, report, vectors


def chroma_collection(output: Path, name: str, create: bool = False):
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    import chromadb
    from chromadb.config import Settings

    store = output / "chroma"
    store.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store), settings=Settings(anonymized_telemetry=False))
    if create:
        return client.get_or_create_collection(name, embedding_function=None,
                                        configuration={"hnsw": {"space": "ip", "ef_construction": 512,
                                                                "ef_search": 512, "num_threads": 1}},
                                        metadata={"purpose": "full_cgi_candidate", "model": "multilingual-e5-small"})
    return client.get_collection(name, embedding_function=None)


def build_chroma(output: Path) -> dict:
    output = candidate_path(output)
    summary, index, vectors_report, vectors = verified_vectors(output)
    docs = index["documents"]
    name = "cgi_full_" + summary["index_sha256"][:8] + "_" + vectors_report["vectors_sha256"][:8]
    manifest_path = output / "chroma_manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["index_sha256"] == summary["index_sha256"] and old["vectors_sha256"] == vectors_report["vectors_sha256"]:
            collection = chroma_collection(output, old["collection"])
            if collection.count() != len(docs):
                raise ValueError("Collection candidate incomplète.")
            return old
        if old["index_sha256"] != summary["index_sha256"]:
            raise ValueError("Une autre collection candidate existe déjà dans ce dossier.")
        # La première collection peut avoir été construite avec les vecteurs CPU.
        # On la conserve pour comparaison et construit une collection liée au nouvel hash.
        write_json(output / "chroma_manifest_cpu_reference.json", old)
    collection = chroma_collection(output, name, create=True)
    for start in range(0, len(docs), 256):
        group = docs[start:start + 256]
        existing = collection.get(ids=[d["id"] for d in group], include=["documents", "metadatas"])
        found = {key: (document, metadata) for key, document, metadata in zip(
            existing["ids"], existing["documents"], existing["metadatas"])}
        missing = []
        for i, d in enumerate(group):
            if d["id"] in found:
                text, meta = found[d["id"]]
                if (text != d["text"] or meta["source_sha256"] != d["source"]["sha256"]
                        or meta["index_position"] != start + i):
                    raise ValueError(f"Passage Chroma déjà présent mais discordant : {d['id']}")
            else:
                missing.append((i, d))
        if missing:
            collection.add(ids=[d["id"] for _, d in missing],
                           embeddings=vectors[[start + i for i, _ in missing]].tolist(),
                           documents=[d["text"] for _, d in missing],
                           metadatas=[{"edition": d["edition"], "article": d["article"],
                                       "source_sha256": d["source"]["sha256"],
                                       "pdf_start": d["pdf_page_start"], "pdf_end": d["pdf_page_end"],
                                       "index_position": start + i} for i, d in missing])
        if start % 2048 == 0:
            print(f"Chroma: {start + len(group)}/{len(docs)}", flush=True)
    if collection.count() != len(docs):
        raise ValueError("Chroma ne contient pas tous les passages candidats.")
    for position in sorted({0, len(docs) // 4, len(docs) // 2, 3 * len(docs) // 4, len(docs) - 1}):
        row = collection.get(ids=[docs[position]["id"]], include=["documents", "metadatas", "embeddings"])
        if (row["documents"][0] != docs[position]["text"]
                or row["metadatas"][0]["source_sha256"] != docs[position]["source"]["sha256"]):
            raise ValueError("Échantillon Chroma discordant.")
    report = {"status": "candidate_chroma_built", "collection": name,
              "count": len(docs), "index_sha256": summary["index_sha256"],
              "vectors_sha256": vectors_report["vectors_sha256"],
              "store": str((output / "chroma").relative_to(ROOT)).replace("\\", "/"),
              "space": "ip", "model": vectors_report["model_id"]}
    write_json(manifest_path, report)
    return report


def compare_preselection(output: Path, device: str = "cpu") -> dict:
    import numpy as np

    output = candidate_path(output)
    if device == "cuda_paced":
        _enable_cuda_runtime(output)
    import torch
    summary, index, _, vectors = verified_vectors(output)
    manifest = read_json(output / "chroma_manifest.json")
    if manifest["index_sha256"] != summary["index_sha256"]:
        raise ValueError("Chroma lié à un autre index.")
    collection = chroma_collection(output, manifest["collection"])
    if collection.count() != len(index["documents"]):
        raise ValueError("Chroma incomplet.")
    model, _ = _local_model("cuda" if device == "cuda_paced" else "cpu")
    docs = index["documents"]
    ids = {d["id"]: i for i, d in enumerate(docs)}
    benchmark = read_json(ROOT / "benchmark/questions.json")["questions"]
    selected = []
    for year in (2023, 2025):
        subset = [q for q in benchmark if q["edition"] == year]
        selected.extend((year, q["question"], "existing_question") for q in subset[::max(1, len(subset) // 8)][:8])
    for year in (2024, 2026):
        subset = [d for d in docs if d["edition"] == year]
        selected.extend((year, f"Article {d['article']} {d['title']}", "technical_title_probe")
                        for d in subset[::max(1, len(subset) // 4)][:4])
    depths = (64, 128, 256, 512)
    records = []
    for edition, question, source in selected:
        if device == "cuda_paced" and _gpu_memory_percent() > 75:
            raise ValueError("VRAM déjà occupée ; comparaison suspendue.")
        query = "query: " + question
        if len(model.tokenizer(query, truncation=False)["input_ids"]) > 512:
            raise ValueError("Sonde technique dépassant la fenêtre E5.")
        vector = model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)[0]
        if device == "cuda_paced":
            torch.cuda.synchronize()
            if _gpu_memory_percent() >= 90:
                raise ValueError("VRAM >= 90 % ; comparaison arrêtée.")
            time.sleep(0.4)
        eligible = [i for i, d in enumerate(docs) if d["edition"] == edition]
        exact_scores = {i: float(vectors[i] @ vector) for i in eligible}
        exact_rank = rank_scores(exact_scores, docs)
        lexical = rank_scores(bm25_scores(index, question, set(eligible)), docs)
        exact_hybrid = rank_scores(rrf_scores([lexical, exact_rank], 60), docs)[:5]
        result = {"edition": edition, "query": question, "probe_type": source,
                  "eligible": len(eligible), "exact_dense_top5": [docs[i]["id"] for i in exact_rank[:5]],
                  "exact_hybrid_top5": [docs[i]["id"] for i in exact_hybrid], "depths": {}}
        for depth in depths:
            started = time.perf_counter()
            found = collection.query(query_embeddings=[vector.tolist()], n_results=min(depth, len(eligible)),
                                     where={"edition": {"$eq": edition}}, include=["distances"])
            elapsed = (time.perf_counter() - started) * 1000
            candidate_ids = found["ids"][0]
            if (len(candidate_ids) != len(set(candidate_ids))
                    or any(x not in ids or docs[ids[x]]["edition"] != edition for x in candidate_ids)):
                raise ValueError("Résultats Chroma étrangers ou dupliqués.")
            candidate_scores = {ids[x]: exact_scores[ids[x]] for x in candidate_ids}
            candidate_rank = rank_scores(candidate_scores, docs)
            hybrid_rank = rank_scores(rrf_scores([lexical, candidate_rank], 60), docs)[:5]
            result["depths"][str(depth)] = {"chroma_ms": elapsed,
                "dense_top1_exact": candidate_rank[:1] == exact_rank[:1],
                "dense_top5_exact": candidate_rank[:5] == exact_rank[:5],
                "dense_top5_recall": len(set(candidate_rank[:5]) & set(exact_rank[:5])) / 5,
                "hybrid_top5_exact": hybrid_rank == exact_hybrid,
                "hybrid_top5_recall": len(set(hybrid_rank) & set(exact_hybrid)) / 5}
        records.append(result)
        print(f"Comparaison Chroma: {edition} {source}", flush=True)
    aggregate = {}
    for depth in depths:
        rows = [r["depths"][str(depth)] for r in records]
        aggregate[str(depth)] = {"queries": len(rows),
            "dense_top1_exact": sum(r["dense_top1_exact"] for r in rows),
            "dense_top5_exact": sum(r["dense_top5_exact"] for r in rows),
            "hybrid_top5_exact": sum(r["hybrid_top5_exact"] for r in rows),
            "mean_dense_top5_recall": statistics.mean(r["dense_top5_recall"] for r in rows),
            "mean_hybrid_top5_recall": statistics.mean(r["hybrid_top5_recall"] for r in rows),
            "median_chroma_ms": statistics.median(r["chroma_ms"] for r in rows)}
    report = {"status": "technical_preselection_comparison_only", "index_sha256": summary["index_sha256"],
              "collection": manifest["collection"], "probe_count": len(records),
              "query_embedding_device": device,
              "probe_policy": "Questions existantes 2023/2025 + titres de passages 2024/2026; aucune preuve juridique annotée pour les titres.",
              "exact_reference": "NumPy float32 dot product over every eligible vector",
              "approximate": "Chroma HNSW edition-filtered bounded query, exact rescoring within returned candidates",
              "aggregate": aggregate, "queries": records}
    write_json(output / "preselection_comparison.json", report)
    return report


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bm25", "embeddings", "chroma", "compare"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda_paced"], default="cpu",
                        help="Pour embeddings/compare ; cuda_paced utilise de petits lots et surveille la VRAM.")
    args = parser.parse_args()
    tasks = {"bm25": build_bm25, "chroma": build_chroma, "compare": compare_preselection}
    result = (build_embeddings(args.output, args.device) if args.command == "embeddings"
              else compare_preselection(args.output, args.device) if args.command == "compare"
              else tasks[args.command](args.output))
    print(result["status"], args.output)


if __name__ == "__main__":
    main()
