"""Post-traitement des réponses citées ; aucun appel au modèle ni preuve nouvelle.

Les contrôles sont structurels. Ils ne certifient pas le sens des affirmations.
"""
from copy import deepcopy
import re
import unicodedata
from .benchmark import normalized
from .citation_blocks import expand_citation_blocks, _bounds, _metadata


def flat_list_introductions(response, passages):
    """Une liste à puces peut conserver le chemin de son introduction.

    Chercher en arrière seulement à travers des items de cette même liste.
    Un paragraphe ordinaire ou un changement de chemin coupe la recherche.
    """
    sources={p['source_id']:p for p in passages}
    for citation in list(response['citations']):
        if citation.get('construction')=='separate_governing_introduction': continue
        rows=sources[citation['source_id']]['lines']
        lo=citation['start_line']-1
        if not re.match(r'^\s*[-•▪\uf0a7]',rows[lo]['text']): continue
        path=rows[lo]['subdivision_path']; cursor=lo
        while cursor>0 and rows[cursor-1]['subdivision_path']==path:
            end=cursor; start,_=_bounds(rows,end-1,end)
            if start==end: break
            text=' '.join(r['text'] for r in rows[start:end])
            if text.rstrip().endswith(':'):
                if len(text.split())>80: break
                ids=[r['id'] for r in rows[start:end]]
                intro=next((c for c in response['citations'] if c['source_id']==citation['source_id'] and c['line_ids']==ids),None)
                if intro is None:
                    intro=deepcopy(citation); _metadata(intro,rows,start,end)
                    intro.update(id=f'C{len(response["citations"])+1}',
                        construction='separate_governing_introduction',derived_from=citation['id'],
                        original_model_line_ids=[],model_selected_line_ids=[],model_selected_quote='',model_selections=[])
                    response['citations'].append(intro)
                for claim in response['claims']:
                    if citation['id'] in claim['citations'] and intro['id'] not in claim['citations']:
                        claim['citations'].append(intro['id'])
                break
            if not re.match(r'^\s*[-•▪\uf0a7]',rows[start]['text']): break
            cursor=start


def positional_proof_error(response, passages, question):
    """Garde limitée aux demandes de voisinage immédiatement après une durée.

    Il faut citer la condition de durée ET le début du paragraphe suivant,
    dans le même passage. Une preuve antérieure seule est insuffisante.
    Ce garde rejette une réponse ; il ne produit aucune conclusion juridique.
    """
    def folded(text):
        return ''.join(c for c in unicodedata.normalize('NFKD',normalized(text).lower()) if not unicodedata.combining(c))
    query=folded(question)
    if 'immediatement apres' not in query or 'duree' not in query: return None
    sources={p['source_id']:p for p in passages}
    citations=[c for c in response['citations'] if c.get('construction')!='separate_governing_introduction']
    for c in citations:
        text=folded(c['quote'])
        if not re.search(r'\b(?:pendant|duree)\b',text): continue
        rows=sources[c['source_id']]['lines']
        for n,row in enumerate(rows):
            if row['id'] not in c['line_ids'] or not re.search(r'\b(?:pendant|duree)\b',folded(row['text'])): continue
            _,end=_bounds(rows,n,n+1)
            if end>=len(rows): continue
            next_id=rows[end]['id']
            if any(d['source_id']==c['source_id'] and next_id in d['line_ids'] for d in citations): return None
    return 'Le voisinage qui suit la condition de durée n’est pas cité ; sa portée ne peut pas être contrôlée.'


def numeric_proof_error(response):
    """Refuser un chiffre ajouté sans présence dans ses propres preuves.

    Les séparateurs de milliers sont neutralisés. Le numéro d'article et
    l'édition des citations sont admis comme métadonnées sourcées. Les nombres
    écrits uniquement en lettres ne sont pas couverts par ce contrôle.
    """
    def numbers(text):
        pattern=r'\d+(?:[ .\u00a0]\d{3})*(?:[.,]\d+)?'
        return {re.sub(r'\D','',n) for n in re.findall(pattern,text)}
    citations={c['id']:c for c in response['citations']}
    for claim in response['claims']:
        allowed=set()
        for ident in claim['citations']:
            c=citations[ident]
            allowed.update(numbers(c['quote']))
            allowed.update(numbers(str(c.get('article',''))))
            allowed.update(numbers(str(c.get('edition',''))))
        missing=numbers(claim['text'])-allowed
        if missing:return 'Chiffre absent des preuves de son affirmation : '+', '.join(sorted(missing))+'.'
    return None


def condition_proof_error(response):
    """Une attestation affirmée doit figurer dans les citations de l'affirmation."""
    citations={c['id']:c for c in response['citations']}
    for claim in response['claims']:
        if re.search(r'\battestation\b',normalized(claim['text']),re.I):
            proof=' '.join(citations[i]['quote'] for i in claim['citations'])
            if not re.search(r'\battestation\b',normalized(proof),re.I):
                return 'L’attestation mentionnée n’apparaît pas dans les preuves citées.'
    return None


def exclusion_scope_error(response,question):
    """Garde lexicale pour une question nommant expressément l'objet exclu."""
    match=re.search(r'quelle exclusion est prévue pour les (.+?) dans la disposition',question,re.I)
    if not match:return None
    from .bm25 import tokenize
    stop={'autres','produits','similaires','source','dans','pour','participations'}
    subject={t for t in tokenize(match.group(1)) if len(t)>=5 and t not in stop}
    if len(subject)<2:return None
    citations={c['id']:c for c in response['citations']}
    for claim in response['claims']:
        for ident in claim['citations']:
            quote=citations[ident]['quote']
            excluded=re.search(r'\b[àa] l.exclusion (?:des?|d[eu])\s+(.+)',quote,re.I|re.S)
            if excluded and len(subject & set(tokenize(excluded.group(1)[:250])))>=2:
                return None
    return 'Les citations ne montrent pas que l’exclusion porte sur l’objet demandé.'


def finalize_generation(generation, passages, question, policy='verified_blocks'):
    if policy!='verified_blocks': return expand_citation_blocks(generation,passages,policy=policy)
    result=expand_citation_blocks(generation,passages,policy='paragraphs')
    response=result['validated']
    if response['status']!='answer': return result
    flat_list_introductions(response,passages)
    # Supprimer uniquement les répétitions strictes après normalisation.
    unique={}; claims=[]
    for claim in response['claims']:
        key=normalized(claim['text'])
        if key in unique:
            unique[key]['citations']=list(dict.fromkeys(unique[key]['citations']+claim['citations']))
        else:
            unique[key]=claim; claims.append(claim)
    response['claims']=claims
    response['answer']='\n'.join(c['text']+' '+''.join(f'[{i}]' for i in c['citations']) for c in claims)
    error=(numeric_proof_error(response) or condition_proof_error(response)
           or exclusion_scope_error(response,question)
           or positional_proof_error(response,passages,question))
    if error:
        result['rejected_candidate']=deepcopy(response)
        result['validated']={'status':'rejected','answer':'Les preuves citées sont insuffisantes pour vérifier la réponse produite. Consultez les passages sources.', 'claims':[], 'citations':[]}
        result['citation_checks']='failed';result['validation_error']=error
    result['citation_block_policy']='verified_blocks'
    return result
