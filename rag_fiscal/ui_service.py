"""Adaptateur de l'interface : pipeline existante, journaux et sources locales."""
from .common import ROOT, read_json, write_json, safe_write, run_id, sha256


def run_question(retriever, question, edition, search_only=False):
    from .ask import answer_question, render_answer
    from .tracking import log_answer
    config = read_json(ROOT / 'config/system.json')
    base = ('data/candidates/full_cgi/answers' if getattr(retriever,'scope',None)=='full_cgi_candidate'
            else 'data/answers')
    folder = ROOT / base / run_id()
    try:
        result = answer_question(retriever, question, edition,
            search_only=search_only, context_mode=config['context_mode'],
            generation_mode=config['generation_mode'], citation_mode=config['citation_mode'],
            citation_block_policy=config.get('citation_block_policy','legacy'))
    except Exception as exc:
        write_json(folder / 'failure.json', {'question': question, 'edition': edition,
            'status': 'technical_error', 'error': str(exc), 'origin': 'streamlit'})
        raise RuntimeError(f'{exc}\nJournal : {folder.relative_to(ROOT)}') from exc
    result['origin'] = 'streamlit'
    write_json(folder / 'result.json', result)
    safe_write(folder / 'answer.md', render_answer(result))
    warning = None
    try:
        ident = log_answer(result, folder)
        write_json(folder / 'tracking.json', {'mlflow_run_id': ident})
    except Exception as exc:
        warning = 'Réponse conservée, mais le suivi MLflow a échoué.'
        write_json(folder / 'tracking_failure.json', {'error': str(exc)})
    return result, str(folder.relative_to(ROOT)), warning


def source_path(source):
    path = (ROOT / source['file']).resolve()
    if not path.is_relative_to((ROOT / 'sources').resolve()) or path.suffix.lower() != '.pdf':
        raise ValueError('Source PDF hors du dossier autorisé.')
    if not path.is_file() or sha256(path) != source.get('sha256'):
        raise ValueError('Empreinte du PDF source incorrecte.')
    return path


def source_pages(passage):
    """Pages PDF réellement présentes dans le passage ou le bloc cité."""
    source = passage['source']
    if source.get('edition') != passage['edition']:
        raise ValueError('La source ne correspond pas à l’édition citée.')
    start, end = passage['pdf_page_start'], passage['pdf_page_end']
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        raise ValueError('Bornes de page PDF invalides.')
    mapped = passage.get('page_map') or []
    pages = sorted({entry['pdf_page'] for entry in mapped}) if mapped else list(range(start,end+1))
    if not pages or pages[0] != start or pages[-1] != end or any(
            type(page) is not int or not start <= page <= end for page in pages):
        raise ValueError('Pagination de la citation incohérente.')
    return pages


def page_image(source, page):
    import pymupdf
    with pymupdf.open(source_path(source)) as document:
        if not 1 <= page <= len(document):
            raise ValueError('Page PDF invalide.')
        return document[page - 1].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).tobytes('png')
