"""Comparaison figée BM25 / E5 local / RRF, sans modifier les passages."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
import statistics
import time

from .common import ROOT, read_json, write_json, safe_write, sha256, run_id, reviewed_articles, configure_console
from .bm25 import tokenize
from .benchmark import validate
from .evaluate import canonical_target, evaluate_query, percentile

MODEL_ID='intfloat/multilingual-e5-small'
MODEL_REVISION='614241f622f53c4eeff9890bdc4f31cfecc418b3'


def eligible_ids(documents, edition, mode):
    return {i for i,p in enumerate(documents) if mode=='off' or p['edition']==edition}


def rank_scores(scores, documents):
    return sorted(scores,key=lambda i:(-scores[i],documents[i]['id']))


def bm25_scores(index, query, eligible):
    scores=defaultdict(float);n=len(index['documents'])
    for term in set(tokenize(query)):
        hits=index['postings'].get(term,[])
        idf=math.log(1+(n-len(hits)+.5)/(len(hits)+.5))
        for i,tf in hits:
            if i in eligible:
                factor=1-index['b']+index['b']*index['lengths'][i]/index['average_length']
                scores[i]+=idf*tf*(index['k1']+1)/(tf+index['k1']*factor)
    return dict(scores)


def rrf_scores(rankings, k=60):
    if k<1:raise ValueError('RRF k positif requis.')
    scores=defaultdict(float)
    for ranking in rankings:
        if len(ranking)!=len(set(ranking)):raise ValueError('Doublon dans une liste de rangs.')
        for rank,i in enumerate(ranking,1):scores[i]+=1/(k+rank)
    return dict(scores)


def describe(rows):
    keys=['recall_at_5','hit_at_1','reciprocal_rank','reciprocal_rank_at_5','top1_wrong_edition',
          'wrong_edition_top5_fraction','union_recall_at_5','index_oracle_recall']
    out={'questions':len(rows),'families':len({r['family_id'] for r in rows})}
    out.update({k:statistics.mean(r[k] for r in rows) for k in keys})
    for k in ['ranking_ms','query_encoding_ms','estimated_online_ms']:
        values=[r[k] for r in rows]
        out[k]={'median':statistics.median(values),'p95':percentile(values,.95)}
    return out


def model_files(path):
    return {p.relative_to(path).as_posix():sha256(p) for p in sorted(path.rglob('*'))
            if p.is_file() and '.cache' not in p.parts}


def main():
    configure_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--experiment',type=Path)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models/multilingual-e5-small')
    args=parser.parse_args()
    # L'inférence est strictement locale. Aucun téléchargement à cet endroit.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    model_path=args.model_dir.resolve()
    state=read_json(ROOT/'data/current_run.json')
    if args.command=='prepare':
        exp=ROOT/'data/experiments'/run_id();exp.mkdir(parents=True)
        baseline=ROOT/state['evaluation']
        old=read_json(baseline);index_path=ROOT/state['index'];index=read_json(index_path)
        questions=read_json(baseline.parent/'benchmark_snapshot.json')
        if len(index['documents'])!=511 or index['chunking']!={'max_words':220,'overlap_lines':2}:
            raise ValueError('Les passages ne correspondent pas à la référence autorisée.')
        provenance=read_json(model_path/'download_manifest.json')
        if provenance['revision']!=MODEL_REVISION or provenance['model_id']!=MODEL_ID:
            raise ValueError('Provenance du modèle inattendue.')
        write_json(exp/'benchmark_snapshot.json',questions)
        import importlib.metadata as metadata
        protocol={'status':'frozen_before_dense_results','frozen_at':datetime.now(timezone.utc).isoformat(),
                  'baseline_evaluation':baseline.relative_to(ROOT).as_posix(),'baseline_sha256':sha256(baseline),
                  'index':index_path.relative_to(ROOT).as_posix(),'index_sha256':sha256(index_path),
                  'passages_sha256':sha256(index_path.parent/'passages.jsonl'),
                  'benchmark_sha256':sha256(exp/'benchmark_snapshot.json'),
                  'code_sha256':{f:sha256(ROOT/f) for f in ['rag_fiscal/compare_retrievers.py','rag_fiscal/bm25.py','rag_fiscal/evaluate.py']},
                  'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'model_dir':model_path.relative_to(ROOT).as_posix(),
                  'model_files':model_files(model_path),'device':'cpu','torch_threads':4,'precision':'float32',
                  'packages':{n:metadata.version(n) for n in ['torch','sentence-transformers','transformers','numpy','huggingface-hub']},
                  'passage_input':'passage: Article {article} {title}\\n{text}',
                  'query_input':'query: {question_exacte}','normalize_embeddings':True,
                  'max_model_tokens':512,'truncation_policy':'abort_if_any_input_exceeds_limit',
                  'methods':['bm25','dense_e5','hybrid_rrf'],'filters':['on','off'],'top_k_evaluation':5,
                  'dense_search':'exact_dot_product_all_eligible_passages','rrf_k':60,'rrf_weights':[1,1],
                  'rrf_depth':'entire eligible lists; BM25 positive lexical hits only',
                  'tie_break':'ascending immutable passage id','tuning':'none',
                  'repetitions_ranking':5,'query_encoding_runs':1,
                  'latency_policy':'Warm model and index; encode each query once, reuse for both modes. Ranking median of five repetitions. Hybrid online estimate sums encoding, BM25, dense and fusion; no disk or model loading.',
                  'test_policy':'Same 30 exposed questions. Exploratory paired comparison, not an independent held-out test.',
                  'chunking_policy':'Indexed passages unchanged; chunking changes require separate validation.',
                  'model_card':'https://huggingface.co/intfloat/multilingual-e5-small',
                  'hardware':{'ram_bytes':16869351424,'gpu':'RTX 4060 Laptop 8188 MiB (not used for this CPU experiment)'}}
        if protocol['index_sha256']!=old['index_sha256'] or protocol['benchmark_sha256']!=old['benchmark_sha256']:
            raise ValueError('Différence avec le benchmark ou l’index de référence.')
        write_json(exp/'protocol.json',protocol)
        print(exp);return
    if args.experiment is None:raise ValueError('--experiment requis.')
    exp=args.experiment.resolve();p=read_json(exp/'protocol.json')
    if (exp/'execution.json').exists():raise ValueError('Expérience déjà commencée ; conserver ses résultats.')
    for f,h in p['code_sha256'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Code modifié après gel : '+f)
    index_path=ROOT/p['index'];baseline_path=ROOT/p['baseline_evaluation']
    for file,h in [(index_path,p['index_sha256']),(index_path.parent/'passages.jsonl',p['passages_sha256']),
                   (baseline_path,p['baseline_sha256']),(exp/'benchmark_snapshot.json',p['benchmark_sha256'])]:
        if sha256(file)!=h:raise ValueError('Artefact modifié : '+str(file))
    if model_files(ROOT/p['model_dir'])!=p['model_files']:raise ValueError('Modèle modifié après gel.')
    index=read_json(index_path);docs=index['documents'];questions=read_json(exp/'benchmark_snapshot.json')
    _,articles=reviewed_articles(ROOT/index['corpus_run'],allow_agent_review=True)
    errors=validate(questions,articles)
    if errors:raise ValueError(errors)
    corpus={(a['edition'],a['article']):a for a in articles}
    targets={q['id']:[canonical_target(e,corpus[e['edition'],e['article']]) for e in q['evidence']] for q in questions['questions']}
    old=read_json(baseline_path);old_rows={(r['question_id'],r['filter']):r for r in old['per_query']}
    start=time.perf_counter()
    model=SentenceTransformer(str(ROOT/p['model_dir']),device='cpu',local_files_only=True,trust_remote_code=False)
    model.max_seq_length=512;model.eval()
    load_ms=(time.perf_counter()-start)*1000
    passages=[f"passage: Article {d['article']} {d['title']}\n{d['text']}" for d in docs]
    queries=['query: '+q['question'] for q in questions['questions']]
    lens=[len(t) for t in model.tokenizer(passages+queries,truncation=False)['input_ids']]
    audit={'passage_lengths':lens[:len(docs)],'query_lengths':lens[len(docs):],
           'max_passage':max(lens[:len(docs)]),'max_query':max(lens[len(docs):]),
           'over_limit':[i for i,n in enumerate(lens) if n>512],'limit':512}
    write_json(exp/'token_audit.json',audit)
    if audit['over_limit']:raise ValueError('Entrées dépassant 512 tokens. Aucune troncature ni modification du chunking exécutée.')
    execution={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256(exp/'protocol.json')}
    write_json(exp/'execution.json',execution)
    try:
        start=time.perf_counter()
        embeddings=model.encode(passages,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
        embedding_ms=(time.perf_counter()-start)*1000
        assert embeddings.shape==(511,384) and np.isfinite(embeddings).all()
        assert np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5)
        buffer=io.BytesIO();np.save(buffer,embeddings,allow_pickle=False);safe_write(exp/'passage_embeddings.npy',buffer.getvalue())
        model.encode(['query: vérification technique'],normalize_embeddings=True,show_progress_bar=False)
        records=[];query_vectors=[]
        for q,query in zip(questions['questions'],queries):
            start=time.perf_counter()
            vector=model.encode([query],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
            encoding_ms=(time.perf_counter()-start)*1000;query_vectors.append(vector)
            payload={'question_id':q['id'],'question':q['question'],'edition':q['edition'],'modes':{}}
            for mode in p['filters']:
                eligible=eligible_ids(docs,q['edition'],mode)
                samples={m:[] for m in p['methods']}
                for _ in range(p['repetitions_ranking']):
                    start=time.perf_counter();bs=bm25_scores(index,q['question'],eligible);br=rank_scores(bs,docs)
                    samples['bm25'].append((time.perf_counter()-start)*1000)
                    start=time.perf_counter();ds={i:float(embeddings[i]@vector) for i in eligible};dr=rank_scores(ds,docs)
                    samples['dense_e5'].append((time.perf_counter()-start)*1000)
                    start=time.perf_counter();hs=rrf_scores([br,dr],p['rrf_k']);hr=rank_scores(hs,docs)
                    samples['hybrid_rrf'].append((time.perf_counter()-start)*1000)
                # Vérifier l'identité de la référence BM25, sur tout le classement enregistré.
                previous=old_rows[q['id'],mode];previous_path=ROOT/previous['results_file']
                if sha256(previous_path)!=previous['results_sha256']:raise ValueError('Ancien résultat altéré.')
                prior=read_json(previous_path)['results']
                if [docs[i]['id'] for i in br]!=[r['id'] for r in prior]:raise ValueError('Divergence BM25 : '+q['id']+'/'+mode)
                payload['modes'][mode]={}
                for method,scores,ranking in [('bm25',bs,br),('dense_e5',ds,dr),('hybrid_rrf',hs,hr)]:
                    full=[{**docs[i],'rank':rank,'score':scores[i]} for rank,i in enumerate(ranking,1)]
                    row=evaluate_query(q,targets[q['id']],full,docs)
                    ranking_ms=statistics.median(samples[method]);enc=0 if method=='bm25' else encoding_ms
                    online=ranking_ms+enc
                    if method=='hybrid_rrf':online+=statistics.median(samples['bm25'])+statistics.median(samples['dense_e5'])
                    row.update(method=method,filter=mode,ranking_ms=ranking_ms,query_encoding_ms=enc,
                               estimated_online_ms=online,results_file=(exp/'results'/f"{q['id']}.json").relative_to(ROOT).as_posix())
                    records.append(row)
                    payload['modes'][mode][method]={'top5':full[:5],
                        'ranking':[{'id':docs[i]['id'],'score':scores[i]} for i in ranking],
                        'timings':{'ranking_repetitions_ms':samples[method],'encoding_ms':enc,'estimated_online_ms':online}}
            write_json(exp/'results'/f"{q['id']}.json",payload)
            print(q['id']+' : trois méthodes, deux modes terminés.',flush=True)
        buffer=io.BytesIO();np.save(buffer,np.asarray(query_vectors),allow_pickle=False);safe_write(exp/'query_embeddings.npy',buffer.getvalue())
        summaries={}
        for method in p['methods']:
            summaries[method]={}
            for mode in p['filters']:
                group=[r for r in records if r['method']==method and r['filter']==mode]
                out={'all':describe(group)}
                for field in ['split','change_status','edition','question_style']:
                    out[field]={str(v):describe([r for r in group if r[field]==v]) for v in sorted({r[field] for r in group},key=str)}
                summaries[method][mode]=out
        write_json(exp/'evaluation.json',{'status':'completed_exploratory_comparison','protocol_sha256':sha256(exp/'protocol.json'),
            'index_sha256':p['index_sha256'],'benchmark_sha256':p['benchmark_sha256'],'model_load_ms':load_ms,
            'passage_embedding_ms':embedding_ms,'embeddings_sha256':sha256(exp/'passage_embeddings.npy'),
            'query_embeddings_sha256':sha256(exp/'query_embeddings.npy'),'ranking_count':len(records),
            'bm25_full_rankings_match_previous':True,'summaries':summaries,'per_query':records,
            'result_sha256':{f.name:sha256(f) for f in sorted((exp/'results').glob('*.json'))}})
        execution['status']='completed'
    except Exception as exc:
        execution.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        execution['finished_at']=datetime.now(timezone.utc).isoformat();write_json(exp/'execution.json',execution)
    print(exp/'evaluation.json')


if __name__=='__main__':main()
