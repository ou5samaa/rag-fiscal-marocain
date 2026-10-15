"""Chroma persistant, vecteurs E5 figés ; classement exact après requête exhaustive.

Les 511 enfants restent identiques. Le mode exhaustif conserve la profondeur RRF
de référence ; aucune promesse de performance ANN n'est faite à cette échelle.
"""
import os
from .common import ROOT, read_json, write_json, sha256

STORE = ROOT / 'data/chroma'
NAME = 'cgi_e5_511_v1'


def collection(create=False):
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(STORE), settings=Settings(anonymized_telemetry=False))
    if create:
        return client.create_collection(NAME, embedding_function=None,
            configuration={'hnsw': {'space': 'ip', 'ef_construction': 512,
                                   'ef_search': 512, 'num_threads': 1}},
            metadata={'purpose': 'frozen_511_children', 'embedding_model': 'multilingual-e5-small'})
    return client.get_collection(NAME, embedding_function=None)


def build():
    import numpy as np
    state = read_json(ROOT / 'data/current_run.json')
    index_path = ROOT / state['index']
    index = read_json(index_path)
    experiment = (ROOT / state['comparison_evaluation']).parent
    evaluation = read_json(experiment / 'evaluation.json')
    vectors_path = experiment / 'passage_embeddings.npy'
    if sha256(vectors_path) != evaluation['embeddings_sha256']:
        raise ValueError('Empreinte des vecteurs incorrecte.')
    docs = index['documents']
    if len(docs) != 511 or sha256(index_path) != evaluation['index_sha256']:
        raise ValueError('Index de référence modifié.')
    manifest = STORE / 'manifest.json'
    if manifest.exists():
        store = ChromaStore(docs, sha256(index_path))
        return store.manifest
    c = collection(create=True)
    vectors = np.load(vectors_path, allow_pickle=False)
    c.add(ids=[d['id'] for d in docs], embeddings=vectors.tolist(),
          documents=[d['text'] for d in docs], metadatas=[{
              'edition': d['edition'], 'article': d['article'],
              'source_sha256': d['source']['sha256'], 'pdf_start': d['pdf_page_start'],
              'pdf_end': d['pdf_page_end'], 'index_position': i} for i, d in enumerate(docs)])
    value = {'collection': NAME, 'count': len(docs), 'index_sha256': sha256(index_path),
             'vectors_sha256': sha256(vectors_path), 'mode': 'exhaustive_query_exact_rescore',
             'space': 'ip', 'embedding_function': None,
             'reason': 'Conserver tous les candidats et les égalités du RRF de référence.'}
    write_json(manifest, value)
    return value


class ChromaStore:
    def __init__(self, documents, index_sha256):
        import numpy as np
        self.documents = documents
        self.manifest = read_json(STORE / 'manifest.json')
        if self.manifest['index_sha256'] != index_sha256:
            raise ValueError('Collection liée à un autre index.')
        self.collection = collection()
        rows = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
        positions = {d['id']: i for i, d in enumerate(documents)}
        if set(rows['ids']) != set(positions):
            raise ValueError('Collection incomplète ou étrangère.')
        # Contrôle des textes/métadonnées ET de tous les vecteurs persistés.
        state = read_json(ROOT / 'data/current_run.json')
        vectors_path = (ROOT / state['comparison_evaluation']).parent / 'passage_embeddings.npy'
        if sha256(vectors_path) != self.manifest['vectors_sha256']:
            raise ValueError('Vecteurs sources modifiés.')
        expected = np.load(vectors_path, allow_pickle=False)
        for n, ident in enumerate(rows['ids']):
            i = positions[ident]; doc = documents[i]; meta = rows['metadatas'][n]
            if (rows['documents'][n] != doc['text'] or meta['edition'] != doc['edition']
                or meta['source_sha256'] != doc['source']['sha256']
                or not np.array_equal(np.asarray(rows['embeddings'][n], dtype=np.float32), expected[i])):
                raise ValueError('Contenu Chroma altéré : ' + ident)
        self.positions = positions

    def scores(self, vector, eligible, edition, edition_filter):
        import numpy as np
        where = {'edition': {'$eq': edition}} if edition_filter == 'on' else None
        result = self.collection.query(query_embeddings=[vector.tolist()],
            n_results=len(eligible), where=where, include=['embeddings', 'distances'])
        ids = result['ids'][0]
        if set(ids) != {self.documents[i]['id'] for i in eligible}:
            raise ValueError('Chroma ne retourne pas tous les candidats éligibles.')
        # Chroma stocke et récupère les vecteurs. Recalcul float32 identique à NumPy
        # pour éviter que l'arrondi HNSW modifie les ex aequo et donc la fusion RRF.
        scores = {self.positions[ident]: float(np.asarray(emb, dtype=np.float32) @ vector)
                  for ident, emb in zip(ids, result['embeddings'][0])}
        return scores


if __name__ == '__main__':
    from .common import configure_console
    configure_console()
    print(build())
