"""Évaluation reproductible des preuves canoniques sur l'index BM25 existant."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .benchmark import normalized, validate
from .common import (ROOT, configure_console, inside_project, read_json, reviewed_articles,
                     sha256, write_json)


def canonical_target(proof, article):
    """Localiser une preuve dans les lignes sources, indépendamment des chunks."""
    selected = [r for r in article['lines'] if r['role'] == 'main'
                and r['pdf_page'] in proof['pdf_pages']
                and r['subdivision_path'][:len(proof['subdivision_path'])] == proof['subdivision_path']]
    runs, current = [], []
    ids = {r['id'] for r in selected}
    for row in article['lines']:
        if row['id'] in ids:
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    found = []
    quote = normalized(proof['quote'])
    for rows in runs:
        text, spans = '', []
        for row in rows:
            if text:
                text += ' '
            start = len(text)
            text += normalized(row['text'])
            spans.append((start, len(text), row['id']))
        pos = text.find(quote)
        while pos >= 0:
            needed = [rid for a,b,rid in spans if b > pos and a < pos + len(quote)]
            found.append(needed)
            pos = text.find(quote, pos+1)
    if len(found) != 1:
        raise ValueError(f"Preuve absente/ambiguë : {proof['edition']}/{proof['article']}")
    return {'proof':proof, 'line_ids':found[0]}


def same_source(passage, target):
    proof = target['proof']
    return (passage['edition'] == proof['edition'] and passage['article'] == proof['article']
            and passage['source']['sha256'].lower() == proof['source_sha256'].lower())


def strict_match(passage, target):
    return (same_source(passage, target)
            and set(target['line_ids']).issubset(passage['line_ids'])
            and normalized(target['proof']['quote']) in normalized(passage['text']))


def evaluate_query(question, targets, results, all_documents, k=5):
    """Unité pertinente = preuve annotée, jamais simple correspondance d'article."""
    if not targets:
        raise ValueError('Preuves requises.')
    top = results[:k]
    ranks = [next((r['rank'] for r in results if strict_match(r,t)), None) for t in targets]
    recalled = sum(rank is not None and rank <= k for rank in ranks)
    first = min((rank for rank in ranks if rank is not None), default=None)
    union_hits, coverages, oracle_ranks = [], [], []
    for target in targets:
        covered = set().union(*(set(p['line_ids']) for p in top if same_source(p,target)))
        needed = set(target['line_ids'])
        union_hits.append(needed.issubset(covered))
        coverages.append(len(needed & covered)/len(needed))
        oracle_ranks.append(any(strict_match(p,target) for p in all_documents))
    wrong = sum(p['edition'] != question['edition'] for p in top)
    top1 = top[0] if top else None
    return {'question_id':question['id'], 'family_id':question['family_id'],
            'split':question['split'], 'change_status':question['change_status'],
            'edition':question['edition'], 'annotation_batch':question.get('annotation_batch'),
            'question_style':question.get('question_style'),
            'evidence_count':len(targets), 'proof_first_ranks':ranks,
            'first_relevant_rank':first, 'recall_at_5':recalled/len(targets),
            'success_at_5':int(recalled == len(targets)),
            'hit_at_1':int(first == 1), 'reciprocal_rank':1/first if first else 0.0,
            'reciprocal_rank_at_5':1/first if first and first<=k else 0.0,
            'union_recall_at_5':sum(union_hits)/len(targets),
            'line_coverage_at_5':statistics.mean(coverages),
            'index_oracle_recall':sum(oracle_ranks)/len(targets),
            'top1_wrong_edition':int(top1 is not None and top1['edition'] != question['edition']),
            'wrong_edition_top5_fraction':wrong/len(top) if top else 0.0,
            'any_wrong_edition_top5':int(wrong>0), 'empty_results':int(not top),
            'top1_article':top1['article'] if top1 else None,
            'top1_edition':top1['edition'] if top1 else None,
            'top5_ids':[p['id'] for p in top]}


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values)-1)*q
    lo,hi = math.floor(pos),math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def aggregate(rows):
    fields = ['recall_at_5','success_at_5','hit_at_1','reciprocal_rank','reciprocal_rank_at_5',
              'union_recall_at_5','line_coverage_at_5','index_oracle_recall',
              'top1_wrong_edition','wrong_edition_top5_fraction','any_wrong_edition_top5','empty_results']
    out = {'questions':len(rows), 'families':len({r['family_id'] for r in rows})}
    out.update({key:statistics.mean(r[key] for r in rows) for key in fields})
    for key in ['retrieval_ms','load_and_integrity_ms','subprocess_wall_ms']:
        values = [r[key] for r in rows]
        out[key] = {'median':statistics.median(values), 'p95':percentile(values,.95),
                    'min':min(values),'max':max(values)}
    return out


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    exp = inside_project(args.experiment)
    protocol = read_json(exp/'protocol.json')
    if (exp/'evaluation.json').exists() or (exp/'execution.json').exists():
        raise ValueError('Expérience déjà commencée. Utiliser un nouveau dossier et un protocole figé.')
    index_path = ROOT/protocol['index']
    if sha256(index_path) != protocol['index_sha256']:
        raise ValueError('Index changé depuis le gel.')
    if sha256(exp/'benchmark_snapshot.json') != protocol['benchmark_sha256']:
        raise ValueError('Questions changées depuis le gel.')
    if sha256(Path(__file__)) != protocol['evaluator_sha256']:
        raise ValueError('Évaluateur changé depuis le gel.')
    if sha256(ROOT/'rag_fiscal/bm25.py') != protocol['bm25_code_sha256']:
        raise ValueError('Moteur changé depuis le gel.')
    index = read_json(index_path)
    if index['chunking'] != {'max_words':220,'overlap_lines':2}:
        raise ValueError('Découpage différent de la configuration retenue.')
    _, articles = reviewed_articles(ROOT/index['corpus_run'],allow_agent_review=True)
    data = read_json(exp/'benchmark_snapshot.json')
    errors = validate(data,articles)
    if errors:
        raise ValueError(errors)
    if len(data['questions']) != 30 or protocol['k'] != 5 or protocol['filters'] != ['on','off']:
        raise ValueError('Protocole attendu : 30 questions, k=5 et deux modes.')
    corpus = {(a['edition'],a['article']):a for a in articles}
    targets = {q['id']:[canonical_target(p,corpus[p['edition'],p['article']]) for p in q['evidence']]
               for q in data['questions']}
    # Préparer toutes les cibles et vérifier leur unicité AVANT les recherches.
    write_json(exp/'canonical_targets.json',targets)
    execution = {'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
                 'protocol_sha256':sha256(exp/'protocol.json'), 'runs':[]}
    write_json(exp/'execution.json',execution)
    records = []
    try:
        for n,q in enumerate(data['questions']):
            # Alterner le premier mode pour réduire le biais d'ordre des mesures de temps.
            for mode in (['on','off'] if n%2==0 else ['off','on']):
                command = [sys.executable,'-m','rag_fiscal.bm25','search','--index',str(index_path),
                           '--edition',str(q['edition']),'--edition-filter',mode,'--query',q['question'],
                           '--top-k',str(len(index['documents']))]
                start = time.perf_counter()
                proc = subprocess.run(command,cwd=ROOT,capture_output=True,encoding='utf-8',check=True,
                                      env={**os.environ,'PYTHONHASHSEED':'0','PYTHONIOENCODING':'utf-8'})
                wall = (time.perf_counter()-start)*1000
                result_path = inside_project(Path(proc.stdout.strip()))
                result = read_json(result_path)
                if result['index_sha256'] != protocol['index_sha256']:
                    raise ValueError('Résultat associé à un autre index.')
                ranks = result['results']
                if [r['rank'] for r in ranks] != list(range(1,len(ranks)+1)):
                    raise ValueError('Rangs non consécutifs.')
                record = evaluate_query(q,targets[q['id']],ranks,index['documents'])
                record.update(filter=mode, results_file=result_path.relative_to(ROOT).as_posix(),
                              results_sha256=sha256(result_path), returned_passages=len(ranks),
                              retrieval_ms=result['retrieval_ms'],
                              load_and_integrity_ms=result['load_and_integrity_ms'], subprocess_wall_ms=wall)
                records.append(record)
                execution['runs'].append({'question_id':q['id'],'filter':mode,
                                          'results_file':record['results_file'],'sha256':record['results_sha256']})
            print(f"{q['id']} : recherches avec/sans filtre terminées.",flush=True)
        summaries = {}
        for mode in protocol['filters']:
            group = [r for r in records if r['filter']==mode]
            summaries[mode] = {'all':aggregate(group)}
            for field in ['split','change_status','edition','annotation_batch','question_style']:
                summaries[mode][field] = {str(value):aggregate([r for r in group if r[field]==value])
                                          for value in sorted({r[field] for r in group},key=str)}
            summaries[mode]['split_change'] = {f'{s}/{c}':aggregate(sub)
                for s in ['dev','test'] for c in ['modified','unchanged']
                if (sub := [r for r in group if r['split']==s and r['change_status']==c])}
        if sha256(index_path)!=protocol['index_sha256'] or sha256(index_path.parent/'passages.jsonl')!=protocol['passages_sha256']:
            raise ValueError('Index ou passages modifiés pendant les recherches.')
        payload = {'status':'completed_frozen_baseline','protocol_sha256':sha256(exp/'protocol.json'),
                   'benchmark_sha256':protocol['benchmark_sha256'],'index_sha256':protocol['index_sha256'],
                   'human_review_status':'pending','question_count':30,'search_count':len(records),
                   'test_results_now_exposed':True,'parameters_tuned_after_results':False,
                   'summaries':summaries,'per_query':records}
        write_json(exp/'evaluation.json',payload)
        execution.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        execution.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(exp/'execution.json',execution)
    print(exp/'evaluation.json')


if __name__=='__main__':
    main()
