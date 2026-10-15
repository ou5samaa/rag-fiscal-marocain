"""Restitution structurelle des citations, sans nouvelle inférence ni jugement juridique.

Conserve la sélection originale du modèle. Étend dans le contexte déjà fourni
au bloc contigu de même subdivision et à son introduction présente. Ne récupère
aucun texte nouveau, ne modifie aucune affirmation et n'invente aucun lien de sens.
"""
from copy import deepcopy
import re


def expand_citation_blocks(generation, passages, policy='legacy'):
    if policy == 'paragraphs':
        return expand_paragraph_blocks(generation, passages)
    if policy != 'legacy':
        raise ValueError('Politique de citations inconnue.')
    result=deepcopy(generation)
    if result['validated']['status']!='answer':return result
    sources={p['source_id']:p for p in passages}
    for citation in result['validated']['citations']:
        p=sources[citation['source_id']];rows=p['lines']
        original=set(citation['line_ids'])
        positions=[n for n,r in enumerate(rows) if r['id'] in original]
        if len(positions)!=len(original):raise ValueError('Lignes de citation absentes du contexte.')
        lo,hi=min(positions),max(positions)+1
        first_path=rows[lo]['subdivision_path'];last_path=rows[hi-1]['subdivision_path']
        while lo>0 and rows[lo-1]['subdivision_path']==first_path:lo-=1
        while hi<len(rows) and rows[hi]['subdivision_path']==last_path:hi+=1
        # L'introduction gouvernant une liste doit apparaître AVANT les items.
        # L'intervalle reste contigu : les items intermédiaires ne sont pas effacés.
        for depth in range(len(first_path)):
            prefix=first_path[:depth]
            prior=[n for n in range(lo) if rows[n]['subdivision_path']==prefix]
            if prior:
                start=prior[-1]
                while start>0 and rows[start-1]['subdivision_path']==prefix:start-=1
                lo=start
        selected=rows[lo:hi]
        citation['model_selected_line_ids']=citation['line_ids']
        citation['model_selected_quote']=citation['quote']
        citation['line_ids']=[r['id'] for r in selected]
        citation['quote']='\n'.join(r['text'] for r in selected)
        citation['pdf_page_start']=selected[0]['pdf_page'];citation['pdf_page_end']=selected[-1]['pdf_page']
        citation['printed_page_start']=selected[0]['printed_page'];citation['printed_page_end']=selected[-1]['printed_page']
        citation['page_map']=list({(r['pdf_page'],r['printed_page']):{'pdf_page':r['pdf_page'],
            'printed_page':r['printed_page']} for r in selected}.values())
        citation['subdivision_paths']=list(dict.fromkeys(tuple(r['subdivision_path']) for r in selected))
        citation['start_line']=lo+1;citation['end_line']=hi
        citation['construction']='structural_block_in_supplied_context'
        if citation['quote'] not in p['text']:raise ValueError('Extension non contiguë.')
    result['citation_mode']='blocks'
    result['semantic_support_verified']=False
    return result


def _ends_paragraph(row):
    # Ne pas traiter le point d'une abréviation à l'intérieur d'une ligne comme une frontière.
    return bool(re.search(r'[.;:]\s*$', row['text']))


def _starts_item(row):
    return bool(re.match(r'^\s*(?:[IVX]+\s*[.\-]|[A-Z]\s*[.\-]|\d+\s*[°)\-]|[-•])', row['text']))


def _bounds(rows, lo, hi):
    path = rows[lo]['subdivision_path']
    while lo > 0 and rows[lo-1]['subdivision_path'] == path:
        if _ends_paragraph(rows[lo-1]) or _starts_item(rows[lo]):
            break
        lo -= 1
    path = rows[hi-1]['subdivision_path']
    while hi < len(rows) and rows[hi]['subdivision_path'] == path:
        if _ends_paragraph(rows[hi-1]) or _starts_item(rows[hi]):
            break
        hi += 1
    return lo, hi


def _metadata(citation, rows, lo, hi):
    selected = rows[lo:hi]
    citation.update(line_ids=[r['id'] for r in selected], quote='\n'.join(r['text'] for r in selected),
        pdf_page_start=selected[0]['pdf_page'], pdf_page_end=selected[-1]['pdf_page'],
        printed_page_start=selected[0]['printed_page'], printed_page_end=selected[-1]['printed_page'],
        page_map=list({(r['pdf_page'], r['printed_page']):{'pdf_page':r['pdf_page'],
            'printed_page':r['printed_page']} for r in selected}.values()),
        subdivision_paths=list(dict.fromkeys(tuple(r['subdivision_path']) for r in selected)),
        start_line=lo+1, end_line=hi, construction='paragraph_block_in_supplied_context')


def expand_paragraph_blocks(generation, passages):
    """Paragraphes bornés et introductions séparées : aucun item frère importé.

    Les lignes sélectionnées sont toujours conservées. Plafond de 180 mots pour
    les ajouts au bloc principal ; une sélection initiale plus longue n'est jamais tronquée.
    L'introduction est une citation distincte, pas un assemblage artificiel de texte.
    """
    result = deepcopy(generation)
    response = result['validated']
    if response['status'] != 'answer': return result
    sources = {p['source_id']:p for p in passages}
    additions = []
    for citation in response['citations']:
        p = sources[citation['source_id']]; rows = p['lines']
        positions = [n for n,r in enumerate(rows) if r['id'] in set(citation['line_ids'])]
        if len(positions) != len(set(citation['line_ids'])):
            raise ValueError('Lignes de citation absentes du contexte.')
        base_lo, base_hi = min(positions), max(positions)+1
        lo, hi = _bounds(rows, base_lo, base_hi)
        capped = len(' '.join(r['text'] for r in rows[lo:hi]).split()) > 180
        if capped: lo,hi = base_lo,base_hi
        citation['model_selected_line_ids'] = list(citation['line_ids'])
        citation['model_selected_quote'] = citation['quote']
        _metadata(citation,rows,lo,hi)
        citation['block_expansion_skipped_for_size'] = capped
        if citation['quote'] not in p['text']: raise ValueError('Extension non contiguë.')
        path = rows[base_lo]['subdivision_path']
        for depth in range(len(path)):
            prefix = path[:depth]
            prior = [n for n in range(lo) if rows[n]['subdivision_path'] == prefix]
            if not prior: continue
            end = prior[-1]+1
            if not rows[end-1]['text'].rstrip().endswith(':'): continue
            start,_ = _bounds(rows,end-1,end)
            # Une introduction courte, complète et réellement visible dans le contexte.
            if len(' '.join(r['text'] for r in rows[start:end]).split()) > 80: continue
            if not (end <= lo or start >= hi): continue
            intro = deepcopy(citation)
            _metadata(intro,rows,start,end)
            intro['construction'] = 'separate_governing_introduction'
            intro['id'] = f'C{len(response["citations"])+len(additions)+1}'
            intro['model_selected_line_ids'] = []; intro['model_selected_quote'] = ''
            intro['derived_from'] = citation['id']
            if intro['quote'] not in p['text']: raise ValueError('Introduction non contiguë.')
            additions.append(intro)
            for claim in response['claims']:
                if citation['id'] in claim['citations']: claim['citations'].append(intro['id'])
    response['citations'].extend(additions)
    # Plusieurs affirmations/sélections peuvent mener au même bloc. Garder une preuve
    # unique et réécrire les références, sans toucher aux affirmations du modèle.
    unique=[]; keys={}; aliases={}
    for c in response['citations']:
        key=(c['source_id'],tuple(c['line_ids']))
        if key in keys:
            target=keys[key]
            aliases[c['id']]=target['id']
            target['model_selections'].append({'line_ids':c['model_selected_line_ids'],
                                               'quote':c['model_selected_quote']})
        else:
            previous=c['id'];c['id']=f'C{len(unique)+1}'
            aliases[previous]=c['id']
            c['model_selections']=[{'line_ids':c['model_selected_line_ids'], 'quote':c['model_selected_quote']}]
            keys[key]=c;unique.append(c)
    for c in unique:
        if 'derived_from' in c:c['derived_from']=aliases[c['derived_from']]
    for claim in response['claims']:
        claim['citations']=list(dict.fromkeys(aliases[i] for i in claim['citations']))
    response['citations']=unique
    response['answer'] = '\n'.join(c['text']+' '+''.join(f'[{i}]' for i in c['citations']) for c in response['claims'])
    result['citation_mode'] = 'blocks'
    result['citation_block_policy'] = 'paragraphs'
    result['semantic_support_verified'] = False
    return result
