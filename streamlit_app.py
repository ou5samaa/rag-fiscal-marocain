"""Interface locale : lancer avec scripts/start_ui.ps1."""
import json
import os
import threading
import streamlit as st
from rag_fiscal.common import ROOT, read_json
from rag_fiscal.ui_service import run_question, page_image, source_path, source_pages

st.set_page_config(page_title='CGI • Recherche fiscale', page_icon='📖', layout='wide')
st.markdown('''<style>
.stApp {background:#f7f8fb;} h1,h2,h3 {color:#16324f;}
.block-container {max-width:1200px;padding-top:2.5rem;}
[data-testid="stSidebar"] {background:#eaf0f5;}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def engine(backend, scope):
    if scope == 'full_cgi_candidate':
        from rag_fiscal.candidate_retrieve import CandidateRetriever
        return CandidateRetriever(backend=backend), threading.Lock()
    from rag_fiscal.retrieve import Retriever
    return Retriever(backend=backend), threading.Lock()


SCOPE = os.environ.get('RAG_FISCAL_SCOPE') or read_json(ROOT / 'data/current_run.json').get('scope', 'active')
if SCOPE not in {'active','full_cgi_candidate'}:
    raise ValueError('RAG_FISCAL_SCOPE inconnu.')
EDITIONS = [2023,2024,2025,2026] if SCOPE == 'full_cgi_candidate' else [2023,2025]


@st.cache_data(max_entries=24, show_spinner=False)
def preview(source_json, page):
    return page_image(json.loads(source_json), page)


def show_source(passage, key):
    start, end = passage['pdf_page_start'], passage['pdf_page_end']
    source = passage['source']
    pages = source_pages(passage)
    st.caption(f"CGI {passage['edition']} · Article {passage['article']} · "
               f"Pages PDF {start}–{end} · Pagination imprimée "
               f"{passage.get('printed_page_start', '—')}–{passage.get('printed_page_end', '—')}")
    subdivision = '/'.join(passage.get('subdivision_path', []))
    if subdivision:
        st.caption('Subdivision : ' + subdivision)
    st.text(passage.get('quote', passage.get('text', '')))
    url = source.get('source_url', '')
    if url.startswith(('https://', 'http://')):
        st.link_button('Consulter la publication source', url)
    if st.checkbox('Afficher la page originale', key=key + '_show'):
        page = st.selectbox('Page PDF citée', pages, key=key + '_page')
        try:
            st.image(preview(json.dumps(source, sort_keys=True), page),
                     caption=f'Page PDF {page} — source originale', width='stretch')
            st.download_button('Télécharger le PDF original', source_path(source).read_bytes(),
                               file_name=source_path(source).name, mime='application/pdf', key=key+'_pdf')
        except Exception as exc:
            st.error(f'Impossible d’afficher cette page : {exc}')


with st.sidebar:
    st.title('📖 CGI Maroc')
    st.caption('RECHERCHE PAR ÉDITION')
    if SCOPE == 'full_cgi_candidate':
        st.markdown('**Corpus disponible**\n\nCGI 2023, 2024, 2025 et 2026 · Français\n\nCorps du CGI indexé.')
    else:
        st.markdown('**Corpus disponible**\n\nCGI 2023 et 2025 · Français\n\nArticles 1 à 20, intercalaires inclus. Impôt sur les sociétés.')
    st.divider()
    st.markdown('**Corpus fiscal marocain · sources citées**')
    st.caption('Prototype expérimental. Une édition du CGI ne suffit pas toujours à déterminer la règle applicable à une opération datée.')
    with st.expander('À propos du moteur'):
        st.write('Recherche hybride BM25 + E5, filtre obligatoire par édition, stockage Chroma, génération locale Qwen via Ollama, suivi MLflow.')
        st.write('Configuration cible : réponse rédigée, contexte parent et citations en blocs. La configuration reste expérimentale ; les preuves insuffisantes peuvent conduire à un rejet.')

st.caption('CODE GÉNÉRAL DES IMPÔTS · ASSISTANT LOCAL')
st.title('Une question fiscale. Des sources consultables.')
st.write('Retrouvez une disposition dans l’édition choisie et examinez les passages qui soutiennent la réponse.')

with st.form('question_form'):
    left, right = st.columns([1, 2])
    edition = left.selectbox('Édition du CGI', EDITIONS)
    mode = right.radio('Mode', ['Réponse avec citations', 'Recherche de passages'], horizontal=True)
    question = st.text_area('Votre question', placeholder='Quelle est la durée maximale de l’exercice comptable ?', height=100, max_chars=2000)
    submitted = st.form_submit_button('Rechercher', type='primary')

if submitted:
    # Effacer l'ancienne réponse avant toute validation pour éviter une attribution erronée.
    st.session_state.pop('last_result', None)
    if not question.strip():
        st.warning('Saisissez une question avant de lancer la recherche.')
    else:
        try:
            search_only = mode == 'Recherche de passages'
            from rag_fiscal.generate import edition_conflict, local_api
            if edition_conflict(question, edition):
                st.warning('L’édition mentionnée dans la question diffère de celle sélectionnée.')
            else:
                if not search_only:
                    try:
                        local_api('/api/version', timeout=2)
                    except Exception:
                        raise RuntimeError('Ollama est indisponible. Dans un terminal du projet, lancez : powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\start_ollama.ps1')
                with st.spinner('Recherche dans les sources… Le premier chargement peut prendre une minute.'):
                    config = read_json(ROOT / 'config/system.json')
                    retriever, lock = engine(config['vector_backend'], SCOPE)
                    with lock:
                        result, folder, warning = run_question(retriever, question.strip(), edition, search_only)
                    st.session_state.last_result = (result, folder, warning)
        except Exception as exc:
            st.error(str(exc))

if 'last_result' in st.session_state:
    result, folder, warning = st.session_state.last_result
    st.divider()
    st.subheader(f"Résultat · CGI {result['edition']}")
    st.write(result['question'])
    st.caption(f"Durée du traitement : {result.get('total_wall_ms', 0)/1000:.1f} s · Recherche filtrée par édition")
    if warning:
        st.warning(warning)
    response, sources, details = st.tabs(['Réponse', 'Passages retrouvés', 'Traçabilité'])
    with response:
        status = result['status']
        if status == 'answer':
            st.write(result['answer'])
            st.caption('Réponse expérimentale : l’exactitude des citations ne garantit pas la justesse de leur interprétation.')
        elif status == 'retrieval_only':
            st.info('Recherche seule : les extraits sont disponibles dans « Passages retrouvés ». Aucune réponse générée.')
        else:
            st.warning(result.get('answer', 'Aucune réponse exploitable.'))
        for i, citation in enumerate(result.get('citations', [])):
            with st.expander(f"{citation['id']} · Article {citation['article']}", expanded=True):
                show_source(citation, folder + f'_citation_{i}')
    with sources:
        for i, passage in enumerate(result.get('retrieval', {}).get('results', [])):
            with st.expander(f"Rang {passage['rank']} · Article {passage['article']} — {passage.get('title', '')}"):
                show_source(passage, folder + f'_passage_{i}')
    with details:
        st.caption('Journal local : ' + folder)
        st.download_button('Télécharger le résultat complet (JSON)', json.dumps(result, ensure_ascii=False, indent=2),
                           file_name='reponse_cgi.json', mime='application/json')
        st.write('Les réponses sont enregistrées localement et suivies dans MLflow. Les questions ne sont pas envoyées à une API de génération externe.')
else:
    a, b, c = st.columns(3)
    a.info('01 · Choisissez l’édition\n\nChaque recherche porte sur une seule édition du CGI.')
    b.info('02 · Posez une question précise\n\nIndiquez la disposition ou le thème recherché dans cette édition.')
    c.info('03 · Vérifiez les preuves\n\nConsultez l’article, l’extrait cité et la page originale.')
