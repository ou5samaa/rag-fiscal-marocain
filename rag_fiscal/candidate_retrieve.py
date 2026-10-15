"""Recherche sur le CGI complet candidat, isolée de la configuration active."""
from __future__ import annotations

import re
import time
from collections import Counter

from .common import read_json
from .compare_retrievers import bm25_scores, rank_scores, rrf_scores
from .full_corpus import DEFAULT_OUTPUT
from .full_index import (_enable_cuda_runtime, _gpu_memory_percent, chroma_collection,
                         verified_extraction, verified_index, verified_vectors)


class CandidateRetriever:
    scope = "full_cgi_candidate"
    editions = (2023, 2024, 2025, 2026)
    dense_candidates = 64

    def __init__(self, backend="chroma"):
        if backend not in {"chroma", "numpy"}:
            raise ValueError("Backend inconnu.")
        self.root = DEFAULT_OUTPUT
        _, self._articles = verified_extraction(self.root)
        self.summary, self.index = verified_index(self.root)
        self.documents = self.index["documents"]
        self.backend = backend
        self.model = None
        self.embeddings = None
        self.collection = None
        self._context_builder = None
        if set(int(y) for y in self.summary["by_edition"]) != set(self.editions):
            raise ValueError("Index candidat incomplet : éditions attendues absentes.")

    def load_dense(self):
        if self.model is not None:
            return
        import numpy as np

        _, _, vectors_report, vectors = verified_vectors(self.root)
        if vectors_report["device"] != "cuda_paced":
            raise ValueError("Les vecteurs candidats doivent provenir du calcul GPU validé.")
        _enable_cuda_runtime(self.root)
        import torch

        if torch.version.cuda is None or not torch.cuda.is_available():
            raise ValueError("PyTorch CUDA candidat indisponible dans ce processus.")
        from .full_index import _local_model

        self.model, _ = _local_model("cuda")
        self.embeddings = vectors
        if self.backend == "chroma":
            manifest = read_json(self.root / "chroma_manifest.json")
            if (manifest["index_sha256"] != self.summary["index_sha256"]
                    or manifest["vectors_sha256"] != vectors_report["vectors_sha256"]):
                raise ValueError("Chroma candidat lié à un autre index ou modèle.")
            self.collection = chroma_collection(self.root, manifest["collection"])
            if self.collection.count() != len(self.documents):
                raise ValueError("Chroma candidat incomplet.")
            self.positions = {d["id"]: i for i, d in enumerate(self.documents)}

    def context_builder(self):
        if self._context_builder is None:
            from .context import ContextBuilder

            self._context_builder = ContextBuilder(articles=self._articles)
        return self._context_builder

    @staticmethod
    def requested_article(question):
        if re.search(r"\barticles\b", question, flags=re.IGNORECASE):
            return None
        matches = re.findall(r"\barticle\s+(premier|1er|\d{1,3}(?:\s*(?:bis|ter|quater))?)\b",
                             question, flags=re.IGNORECASE)
        if re.search(r"\barticle\s+(?:premier|1er|\d{1,3}(?:\s*(?:bis|ter|quater))?)\s+(?:et|ou)\s+\d+",
                     question, flags=re.IGNORECASE):
            return None
        labels = {"1" if match.lower() in {"premier", "1er"}
                  else re.sub(r"\s+", " ", match.lower().strip()) for match in matches}
        return next(iter(labels)) if len(labels) == 1 else None

    def _dense_scores(self, question, edition, eligible, article=None):
        self.load_dense()
        import torch
        query = "query: " + question
        if len(self.model.tokenizer(query, truncation=False)["input_ids"]) > 512:
            raise ValueError("Question trop longue pour E5 ; aucune troncature effectuée.")
        if _gpu_memory_percent() > 75:
            raise ValueError("VRAM déjà occupée à plus de 75 % ; attendre avant la recherche.")
        vector = self.model.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)[0]
        torch.cuda.synchronize()
        if _gpu_memory_percent() >= 90:
            raise ValueError("VRAM à 90 % ou plus ; recherche interrompue.")
        if self.backend == "numpy":
            return {i: float(self.embeddings[i] @ vector) for i in eligible}
        where = {"edition": {"$eq": edition}}
        if article is not None:
            where = {"$and": [where, {"article": {"$eq": article}}]}
        result = self.collection.query(query_embeddings=[vector.tolist()],
                                       n_results=min(self.dense_candidates, len(eligible)),
                                       where=where,
                                       include=["distances"])
        ids = result["ids"][0]
        if (len(ids) != len(set(ids)) or
                any(key not in self.positions or self.positions[key] not in eligible for key in ids)):
            raise ValueError("Chroma a renvoyé une mauvaise édition ou un identifiant dupliqué.")
        return {self.positions[key]: float(self.embeddings[self.positions[key]] @ vector)
                for key in ids}

    def search(self, question, edition, method="hybrid_rrf", top_k=5, edition_filter="on"):
        if edition not in self.editions:
            raise ValueError("Édition absente du corpus candidat.")
        if edition_filter != "on":
            raise ValueError("Le filtre d'édition est obligatoire.")
        if method not in {"bm25", "dense_e5", "hybrid_rrf"}:
            raise ValueError("Moteur inconnu.")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Question requise, de 1 à 2000 caractères.")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k doit être compris entre 1 et 20.")
        started = time.perf_counter()
        requested = self.requested_article(question)
        if requested and not any(d["edition"] == edition and d["article"].lower() == requested
                                 for d in self.documents):
            requested = None
        eligible = {i for i, d in enumerate(self.documents)
                    if d["edition"] == edition and (requested is None or d["article"].lower() == requested)}
        lexical = bm25_scores(self.index, question, eligible) if method != "dense_e5" else {}
        dense = self._dense_scores(question, edition, eligible, requested) if method != "bm25" else {}
        if method == "bm25":
            scores = lexical
        elif method == "dense_e5":
            scores = dense
        else:
            scores = rrf_scores([rank_scores(lexical, self.documents),
                                 rank_scores(dense, self.documents)], 60)
        full_ranking = rank_scores(scores, self.documents)
        if method == "hybrid_rrf" and top_k == 5:
            # Les longues dispositions transitoires ne doivent pas occuper à elles
            # seules toute la fenêtre de preuve. Garder aussi le meilleur résultat
            # lexical, utile lorsque la fusion le place juste derrière le top 5.
            counts = Counter()
            ranked = []
            for i in full_ranking:
                article_key = self.documents[i]["article"]
                if counts[article_key] < 2:
                    ranked.append(i)
                    counts[article_key] += 1
                if len(ranked) == top_k:
                    break
            lexical_ranked = rank_scores(lexical, self.documents)
            if lexical_ranked and lexical_ranked[0] not in ranked:
                lexical_winner = lexical_ranked[0]
                article_key = self.documents[lexical_winner]["article"]
                same_article = [i for i, item in enumerate(ranked)
                                if self.documents[item]["article"] == article_key]
                replace_at = same_article[-1] if same_article else len(ranked) - 1
                ranked[replace_at] = lexical_winner
        else:
            ranked = full_ranking[:top_k]
        return {"question": question, "edition": edition, "method": method,
                "edition_filter": "on", "scope": self.scope,
                "requested_article_filter": requested,
                "index_sha256": self.summary["index_sha256"],
                "passages_sha256": self.summary["passages_sha256"],
                "vector_backend": self.backend,
                "dense_candidate_limit": self.dense_candidates if method != "bm25" else None,
                "ranking_policy": "article_diversity_2_plus_lexical_winner" if method == "hybrid_rrf" and top_k == 5 else "score_order",
                "retrieval_wall_ms": (time.perf_counter() - started) * 1000,
                "results": [dict(self.documents[i], rank=n, score=scores[i], source_id=f"S{n}")
                            for n, i in enumerate(ranked, 1)]}
