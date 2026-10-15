"""Recherche réutilisable sur les passages et vecteurs figés, sans reconstruire le corpus."""
from __future__ import annotations
import os
import time
from .common import ROOT,read_json,sha256,reviewed_articles
from .compare_retrievers import eligible_ids,bm25_scores,rank_scores,rrf_scores,model_files


class Retriever:
    def __init__(self, backend=None):
        state=read_json(ROOT/'data/current_run.json')
        self.experiment=(ROOT/state['comparison_evaluation']).parent
        self.protocol=read_json(self.experiment/'protocol.json')
        self.evaluation=read_json(self.experiment/'evaluation.json')
        self.index_path=ROOT/state['index']
        if sha256(self.index_path)!=self.protocol['index_sha256']:
            raise ValueError('Index différent de la comparaison retenue.')
        if sha256(self.index_path.parent/'passages.jsonl')!=self.protocol['passages_sha256']:
            raise ValueError('Passages altérés.')
        self.index=read_json(self.index_path);self.documents=self.index['documents']
        if self.index['chunking']!={'max_words':220,'overlap_lines':2}:
            raise ValueError('Découpage différent de la référence autorisée.')
        reviewed_articles(ROOT/self.index['corpus_run'],allow_agent_review=True)
        self.model=None;self.embeddings=None
        config=ROOT/'config/system.json'
        self.backend=backend or (read_json(config).get('vector_backend','numpy') if config.exists() else 'numpy')
        if self.backend not in {'numpy','chroma'}:raise ValueError('Backend inconnu.')
        self.store=None

    def load_dense(self):
        if self.model is not None:return
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        path=ROOT/self.protocol['model_dir']
        if model_files(path)!=self.protocol['model_files']:raise ValueError('Poids E5 modifiés.')
        vectors=self.experiment/'passage_embeddings.npy'
        if sha256(vectors)!=self.evaluation['embeddings_sha256']:raise ValueError('Vecteurs modifiés.')
        if self.evaluation['index_sha256']!=sha256(self.index_path):raise ValueError('Vecteurs associés à un autre index.')
        self.embeddings=np.load(vectors,allow_pickle=False)
        if self.embeddings.shape!=(len(self.documents),384) or not np.isfinite(self.embeddings).all():raise ValueError('Vecteurs invalides.')
        torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
        self.model=SentenceTransformer(str(path),device='cpu',local_files_only=True,trust_remote_code=False)
        self.model.max_seq_length=512;self.model.eval()

    def search(self,question,edition,method='hybrid_rrf',top_k=5,edition_filter='on'):
        if edition not in {2023,2025}:raise ValueError('Seules les éditions 2023 et 2025 sont indexées.')
        if method not in {'bm25','dense_e5','hybrid_rrf'}:raise ValueError('Moteur inconnu.')
        if edition_filter!='on':raise ValueError('Le filtre d’édition est obligatoire.')
        if not isinstance(question,str) or not question.strip() or len(question)>2000:raise ValueError('Question requise, de 1 à 2000 caractères.')
        if not 1<=top_k<=20:raise ValueError('top_k doit être compris entre 1 et 20.')
        started=time.perf_counter();eligible=eligible_ids(self.documents,edition,edition_filter)
        scores={};dense_scores={}
        if method in {'bm25','hybrid_rrf'}:scores=bm25_scores(self.index,question,eligible)
        if method in {'dense_e5','hybrid_rrf'}:
            self.load_dense()
            query='query: '+question
            if len(self.model.tokenizer(query,truncation=False)['input_ids'])>512:
                raise ValueError('Question trop longue pour E5 ; aucune troncature effectuée.')
            vector=self.model.encode([query],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
            if self.backend=='chroma':
                if self.store is None:
                    from .vector_store import ChromaStore
                    self.store=ChromaStore(self.documents,sha256(self.index_path))
                dense_scores=self.store.scores(vector,eligible,edition,edition_filter)
            else:dense_scores={i:float(self.embeddings[i]@vector) for i in eligible}
        if method=='dense_e5':scores=dense_scores
        elif method=='hybrid_rrf':scores=rrf_scores([rank_scores(scores,self.documents),rank_scores(dense_scores,self.documents)],60)
        ranked=rank_scores(scores,self.documents)[:top_k]
        return {'question':question,'edition':edition,'method':method,'edition_filter':edition_filter,
                'index_sha256':sha256(self.index_path),'passages_sha256':self.protocol['passages_sha256'],
                'vector_backend':self.backend,'retrieval_wall_ms':(time.perf_counter()-started)*1000,
                'results':[dict(self.documents[i],rank=n,score=scores[i],source_id=f'S{n}') for n,i in enumerate(ranked,1)]}
