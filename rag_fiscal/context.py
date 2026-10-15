"""Contexte borné construit depuis les lignes sources, sans changer les enfants indexés."""
import re
from .common import ROOT, read_json, reviewed_articles


def candidate_roles(text):
    """Repérage lexical explicite, pas une qualification juridique validée."""
    roles = []
    if re.search(r'exception|toutefois|cependant|ne s.appliqu|à l.exclusion|except', text, re.I):
        roles.append('exception_candidate')
    if re.search(r'condition|sous réserve|lorsque|à compter|subordonné|si elles|si le', text, re.I):
        roles.append('condition_candidate')
    if not roles:
        roles.append('rule_or_other_unclassified')
    return roles


class ContextBuilder:
    def __init__(self, articles=None):
        if articles is None:
            state = read_json(ROOT / 'data/current_run.json')
            _, articles = reviewed_articles(ROOT / state['extraction'], allow_agent_review=True)
        self.articles = {a['id']: a for a in articles}
        self.article_by_key = {(a['edition'], a['article']): a for a in articles}
        self.rows = {a['id']: [r for r in a['lines'] if r['role'] == 'main'] for a in articles}
        self.positions = {aid: {r['id']: n for n, r in enumerate(rows)} for aid, rows in self.rows.items()}

    def annotations(self):
        output = []
        for aid, rows in self.rows.items():
            for lo, hi in self.runs(rows):
                text = ' '.join(r['text'] for r in rows[lo:hi])
                output.append({'article_id': aid, 'line_ids': [r['id'] for r in rows[lo:hi]],
                    'subdivision_path': rows[lo]['subdivision_path'], 'candidate_roles': candidate_roles(text),
                    'list_introduction_candidate': text.rstrip().endswith(':'),
                    'classification': 'lexical_heuristic_not_legal_validation'})
        return output

    @staticmethod
    def runs(rows):
        start = 0
        for n in range(1, len(rows) + 1):
            if n == len(rows) or rows[n]['subdivision_path'] != rows[start]['subdivision_path']:
                yield start, n
                start = n

    def _passages(self, selected, anchors):
        passages = []
        # L'ordre des articles suit le rang du premier enfant retrouvé ; à l'intérieur,
        # ordre source. Aucune citation ne peut franchir une lacune de lignes.
        for aid in dict.fromkeys(p['article_id'] for p in anchors):
            positions = sorted(selected.get(aid, set()))
            groups = []
            for pos in positions:
                if not groups or pos != groups[-1][-1] + 1: groups.append([])
                groups[-1].append(pos)
            a = self.articles[aid]
            for group in groups:
                rows = [self.rows[aid][n] for n in group]
                page_map = list({(r['pdf_page'], r['printed_page']):
                    {'pdf_page': r['pdf_page'], 'printed_page': r['printed_page']} for r in rows}.values())
                paths = [r['subdivision_path'] for r in rows]
                common = list(paths[0])
                while common and any(p[:len(common)] != common for p in paths): common.pop()
                structure = [dict(start_line=lo+1, end_line=hi,
                    subdivision_path=rows[lo]['subdivision_path'],
                    article_line_start=group[lo]+1, article_line_end=group[hi-1]+1)
                    for lo, hi in self.runs(rows)]
                passages.append({'id': f'{aid}:context:{group[0]}-{group[-1]}', 'article_id': aid,
                    'source_id': f'S{len(passages)+1}', 'edition': a['edition'], 'article': a['article'],
                    'title': a.get('title', ''),
                    'subdivision_path': common, 'text': '\n'.join(r['text'] for r in rows),
                    'line_ids': [r['id'] for r in rows], 'lines': rows, 'source': a['source'],
                    'structure': structure, 'article_line_start': group[0]+1,
                    'article_line_end': group[-1]+1,
                    'pdf_page_start': rows[0]['pdf_page'], 'pdf_page_end': rows[-1]['pdf_page'],
                    'printed_page_start': rows[0]['printed_page'], 'printed_page_end': rows[-1]['printed_page'],
                    'page_map': page_map, 'candidate_roles': candidate_roles(' '.join(r['text'] for r in rows)),
                    'retrieved_children': [p['id'] for p in anchors if p['article_id'] == aid
                        and set(p['line_ids']) & {r['id'] for r in rows}]})
        return passages

    def _introductions(self, aid, start, path):
        rows = self.rows[aid]; indices = set()
        for depth in range(len(path)):
            prefix = path[:depth]
            # Dernier bloc ancêtre AVANT l'item : son introduction gouverne la liste.
            candidates = [(lo, hi) for lo, hi in self.runs(rows)
                          if hi <= start and rows[lo]['subdivision_path'] == prefix]
            if candidates:
                lo, hi = candidates[-1]
                indices.update(range(lo, hi))
        return indices

    def build(self, anchors, mode, fits, token_count):
        if mode not in {'children', 'neighbors', 'parents'}: raise ValueError(mode)
        selected = {}
        for p in anchors:
            if p['article_id'] not in self.rows: raise ValueError('Article source absent.')
            if self.articles[p['article_id']]['edition'] != p['edition']: raise ValueError('Édition incohérente.')
            selected.setdefault(p['article_id'], set()).update(self.positions[p['article_id']][lid] for lid in p['line_ids'])
        passages = self._passages(selected, anchors)
        if not fits(passages): raise ValueError('Les enfants seuls dépassent le budget ; aucune troncature.')
        decisions = []
        if mode != 'children':
            for p in anchors:
                aid = p['article_id']; rows = self.rows[aid]
                pos = [self.positions[aid][lid] for lid in p['line_ids']]
                lo, hi = min(pos), max(pos) + 1; path = rows[lo]['subdivision_path']
                extra = self._introductions(aid, lo, path)
                if mode == 'neighbors':
                    # Voisins bornés aux subdivisions touchées, sans changer le retrieval.
                    left, right = max(0, lo-4), min(len(rows), hi+4)
                    extra.update(i for i in range(left, right) if rows[i]['subdivision_path'] == path)
                else:
                    # Plus petit parent structurel, jamais un long article entier.
                    parent = path[:-1]
                    parent_ids = {i for i, r in enumerate(rows) if r['subdivision_path'][:len(parent)] == parent}
                    text = ' '.join(rows[i]['text'] for i in sorted(parent_ids))
                    if token_count(text) <= 1000:
                        extra.update(parent_ids)
                    else:
                        # Bloc contigu de même subdivision, sinon contexte local borné.
                        block = next((set(range(a,b)) for a,b in self.runs(rows) if a <= lo < b), set())
                        if token_count(' '.join(rows[i]['text'] for i in sorted(block))) <= 1000:
                            extra.update(block)
                        else:
                            extra.update(range(max(0,lo-4), min(len(rows),hi+4)))
                candidate = {a: set(ids) for a, ids in selected.items()}
                candidate[aid].update(extra)
                extended = self._passages(candidate, anchors)
                accepted = fits(extended)
                decisions.append({'child': p['id'], 'candidate_added_lines': len(extra-selected[aid]),
                                  'accepted': accepted, 'reason': 'within_budget' if accepted else 'budget_expansion_skipped'})
                if accepted: selected, passages = candidate, extended
        return {'mode': mode, 'passages': passages, 'decisions': decisions,
                'unique_lines': sum(len(ids) for ids in selected.values()),
                'indexed_children_unchanged': True, 'semantic_labels': 'heuristic_candidates_only'}

    def oracle(self, question, targets):
        selected = {}; anchors = []
        for target in targets:
            aid = self.article_by_key[question['edition'],target['proof']['article']]['id']
            selected.setdefault(aid, set()).update(self.positions[aid][lid] for lid in target['line_ids'])
            anchors.append({'id': 'oracle', 'article_id': aid, 'line_ids': target['line_ids']})
        return self._passages(selected, anchors)
