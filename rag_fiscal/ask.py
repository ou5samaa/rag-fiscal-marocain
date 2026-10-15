"""Poser une question au corpus local : recherche seule ou recherche + réponse sourcée."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import re
from .common import ROOT,configure_console,write_json,safe_write,run_id,sha256,read_json
from .generate import Generator,edition_conflict
from .retrieve import Retriever


def underspecified_rate(question):
    """Une demande de taux sans objet fiscal identifiable appelle une précision."""
    text=re.sub(r'[?!.,;:]+','',question.casefold()).strip()
    text=re.sub(r'\b(?:selon|dans|du|le|la|les|cgi|marocain|de|lédition|édition|en|2023|2024|2025|2026)\b',' ',text)
    text=' '.join(text.split())
    return bool(re.fullmatch(r"quel(?:le)?(?:s)? (?:est|sont) (?:le |la |les )?taux (?:applicable|applicables|prévu|prévus)?",text))


def focus_explicit_exclusion(question, passages):
    """Privilégier la disposition qui porte littéralement l'exclusion demandée.

    Si aucun passage ne remplit ce critère strict, conserver les résultats initiaux.
    Cette sélection agit sur le contexte, jamais sur le classement évalué.
    """
    match=re.search(r'quelle exclusion est prévue pour les\s+([^?]+?)\s+dans la disposition',question,re.I)
    if not match:return passages
    from .bm25 import tokenize
    nouns=[token for token in tokenize(match.group(1)) if len(token)>=5]
    if not nouns:return passages
    head=nouns[0]
    chosen=[p for p in passages if re.search(r"[àa] l.exclusion des?\s+"+re.escape(head)+r"\b",
                                          p['text'],re.I)]
    return chosen or passages


def answer_question(retriever,question,edition,method='hybrid_rrf',generator=None,search_only=False,edition_filter='on',context_mode='children',generation_mode='baseline',citation_mode='lines',citation_block_policy='legacy'):
    if edition_filter!='on':raise ValueError('Le filtre d’édition est obligatoire.')
    if not question.strip():raise ValueError('Question vide.')
    if edition_conflict(question,edition):
        return {'question':question,'edition':edition,'status':'edition_conflict','answer':'L’édition mentionnée dans la question ne correspond pas à celle sélectionnée. Reformulez ou changez --edition.','citations':[]}
    if not search_only and underspecified_rate(question):
        return {'question':question,'edition':edition,'status':'abstain',
                'answer':'Précisez l’impôt, l’opération ou l’article concerné pour déterminer le taux dans cette édition.',
                'citations':[],'reason':'question_underspecified'}
    start=time.perf_counter()
    retrieval=retriever.search(question,edition,method,5,edition_filter)
    result={'question':question,'edition':edition,'retrieval':retrieval,
            'scope':retrieval.get('scope','active'),'review_kind':'agent_assisted_technical',
            'human_review_status':'pending','semantic_support_verified':False}
    if search_only:result.update(status='retrieval_only',answer='',citations=[])
    elif not retrieval['results']:result.update(status='abstain',answer='Aucun passage retrouvé pour cette question.',citations=[])
    else:
        if generation_mode=='baseline':
            generator=generator or Generator()
            passages=retrieval['results']
        else:
            from .context import ContextBuilder
            from .generate_spans import SpanGenerator
            generator=generator or SpanGenerator(generation_mode)
            builder=getattr(retriever,'_context_builder',None)
            if builder is None:
                builder=(retriever.context_builder() if hasattr(retriever,'context_builder')
                         else ContextBuilder());retriever._context_builder=builder
            anchors=focus_explicit_exclusion(question,retrieval['results'])
            result['context_focus']={'kind':'explicit_exclusion_match' if len(anchors)<len(retrieval['results']) else 'none',
                                     'anchor_ids':[p['id'] for p in anchors]}
            context=builder.build(anchors,context_mode,lambda ps:generator.fits(question,edition,ps),generator.tokens)
            result['context']=context;passages=context['passages']
        generation=generator.generate(question,edition,passages)
        if generation_mode!='baseline' and citation_mode=='blocks':
            from .answer_checks import finalize_generation
            generation=finalize_generation(generation,passages,question,policy=citation_block_policy)
        if generation_mode!='baseline' and generation['validated']['status']=='rejected':
            # Reprise unique et traçable après un rejet : retirer les articles
            # concurrents, sans changer le classement ni inventer une preuve.
            first_article=retrieval['results'][0]['article']
            focused=[p for p in retrieval['results'] if p['article']==first_article]
            if len(focused)<len(retrieval['results']):
                result['recovery']={'policy':'first_ranked_article_after_rejection',
                                    'first_article':first_article,
                                    'initial_validation_error':generation.get('validation_error'),
                                    'initial_status':generation['validated']['status']}
                result['generation_attempts']=[generation]
                try:
                    retry_context=builder.build(focused,context_mode,
                        lambda ps:generator.fits(question,edition,ps),generator.tokens)
                    retry_generation=generator.generate(question,edition,retry_context['passages'])
                    if citation_mode=='blocks':
                        retry_generation=finalize_generation(retry_generation,retry_context['passages'],
                                                            question,policy=citation_block_policy)
                    result['generation_attempts'].append(retry_generation)
                    result['recovery']['retry_status']=retry_generation['validated']['status']
                    result['recovery']['retry_validation_error']=retry_generation.get('validation_error')
                    if retry_generation['validated']['status']=='answer':
                        context=retry_context;passages=context['passages'];generation=retry_generation
                        result['context']=context;result['recovery']['selected']='retry'
                    else:result['recovery']['selected']='initial_rejection'
                except Exception as exc:
                    result['recovery'].update(selected='initial_rejection',
                                              retry_error=f'{type(exc).__name__}: {exc}')
        result.update(generation['validated']);result['generation']=generation
        result['prompt_sha256']=hashlib.sha256(generation['prompt'].encode()).hexdigest()
    result['total_wall_ms']=(time.perf_counter()-start)*1000
    modules=['ask.py','retrieve.py','generate.py','generate_spans.py','context.py','vector_store.py','citation_blocks.py']
    if getattr(retriever,'scope',None)=='full_cgi_candidate':modules.extend(['candidate_retrieve.py','full_corpus.py','full_index.py'])
    result['code_sha256']={n:sha256(ROOT/'rag_fiscal'/n) for n in modules}
    return result


def render_answer(result):
    lines=[f"# CGI {result['edition']} - {result['question']}",'',result.get('answer','')]
    if result['status']=='retrieval_only':
        for p in result['retrieval']['results']:
            lines.extend(['',f"## {p['source_id']} - article {p['article']}, rang {p['rank']}",p['text']])
    for citation in result.get('citations',[]):
        source=citation['source'];pdf=(ROOT/source['file']).as_posix()
        lines.extend(['',f"## {citation['id']} - CGI {citation['edition']}, article {citation['article']}",
            'Subdivision : '+('/'.join(citation['subdivision_path']) or 'sans subdivision'),
            f"Pages du passage : PDF {citation['pdf_page_start']}-{citation['pdf_page_end']} ; imprimées {citation['printed_page_start']}-{citation['printed_page_end']}.",
            '> '+' '.join(citation['quote'].split()),f"[PDF local](<{pdf}>) · [Source]({source['source_url']})"])
    if result['status'] in {'answer','rejected','abstain'}:
        lines.extend(['','Réponse expérimentale : les contrôles de citations vérifient leur présence dans les passages, pas à eux seuls la justesse de chaque affirmation.'])
    return '\n'.join(lines)+'\n'


def main():
    configure_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question',required=True)
    parser.add_argument('--edition',type=int,choices=[2023,2024,2025,2026],required=True)
    parser.add_argument('--scope',choices=['active','full_cgi_candidate'],
                        default=read_json(ROOT/'data/current_run.json').get('scope','active'))
    parser.add_argument('--method',choices=['bm25','dense_e5','hybrid_rrf'],default='hybrid_rrf')
    parser.add_argument('--search-only',action='store_true')
    parser.add_argument('--edition-filter',choices=['on'],default='on')
    config_path=ROOT/'config/system.json'
    config=read_json(config_path) if config_path.exists() else {}
    parser.add_argument('--vector-backend',choices=['chroma','numpy'],default=config.get('vector_backend','numpy'))
    parser.add_argument('--context-mode',choices=['children','neighbors','parents'],default=config.get('context_mode','children'))
    parser.add_argument('--generation-mode',choices=['baseline','spans','extractive'],default=config.get('generation_mode','baseline'))
    parser.add_argument('--citation-mode',choices=['lines','blocks'],default=config.get('citation_mode','lines'))
    parser.add_argument('--citation-block-policy',choices=['legacy','paragraphs','verified_blocks'],default=config.get('citation_block_policy','legacy'))
    args=parser.parse_args()
    if args.scope=='active' and args.edition not in {2023,2025}:
        parser.error('L’édition 2024 ou 2026 nécessite le corpus à quatre éditions.')
    output=(ROOT/'data/candidates/full_cgi/answers' if args.scope=='full_cgi_candidate'
            else ROOT/'data/answers')/run_id()
    try:
        if args.scope=='full_cgi_candidate':
            from .candidate_retrieve import CandidateRetriever
            retriever=CandidateRetriever(backend=args.vector_backend)
        else:retriever=Retriever(backend=args.vector_backend)
        result=answer_question(retriever,args.question,args.edition,args.method,
                              search_only=args.search_only,edition_filter=args.edition_filter,
                              context_mode=args.context_mode,generation_mode=args.generation_mode,citation_mode=args.citation_mode,citation_block_policy=args.citation_block_policy)
    except Exception as exc:
        write_json(output/'failure.json',{'status':'technical_error','question':args.question,'edition':args.edition,'error':f'{type(exc).__name__}: {exc}'})
        raise SystemExit(f'Échec technique : {exc}\nJournal : {output}')
    write_json(output/'result.json',result);safe_write(output/'answer.md',render_answer(result))
    from .tracking import log_answer
    try:
        tracking_id=log_answer(result,output)
        write_json(output/'tracking.json',{'mlflow_run_id':tracking_id})
    except Exception as exc:
        write_json(output/'tracking_failure.json',{'error':str(exc)})
        print('Réponse conservée ; échec du suivi MLflow, voir tracking_failure.json.')
    print(render_answer(result));print('Dossier : '+str(output))


if __name__=='__main__':
    # Reproduire l'ordre des additions flottantes BM25 de la référence, même depuis une nouvelle console.
    if os.environ.get('PYTHONHASHSEED')!='0':
        raise SystemExit(subprocess.call([sys.executable,'-B','-m','rag_fiscal.ask',*sys.argv[1:]],env={**os.environ,'PYTHONHASHSEED':'0'}))
    main()
