"""Suivi MLflow local : SQLite et artefacts dans le projet, aucun service distant."""
import os
from contextlib import contextmanager
from .common import ROOT


def setup():
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = 'true'
    import mlflow
    folder = ROOT / 'data/mlflow'
    folder.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri('sqlite:///' + (folder / 'tracking.db').as_posix())
    name = 'rag-fiscal'
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name, artifact_location=(folder / 'artifacts').as_uri())
    mlflow.set_experiment(name)
    return mlflow


@contextmanager
def tracked(name, params):
    mlflow = setup()
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({'review_kind': 'agent_assisted_technical', 'human_review': 'pending',
                         'test_set': 'previously_exposed_exploratory'})
        yield mlflow, run.info.run_id


def log_answer(result, folder):
    with tracked('question-libre', {
        'edition': result['edition'], 'question': result['question'],
        'method': result.get('retrieval', {}).get('method', ''),
        'status': result['status'],
        'context_mode': result.get('context', {}).get('mode', 'baseline'),
        'generation_mode': result.get('generation', {}).get('generation_mode', 'baseline'),
        'scope': result.get('scope','active'),
    }) as (mlflow, ident):
        mlflow.log_metric('total_wall_ms', result.get('total_wall_ms', 0))
        mlflow.log_artifact(str(folder / 'result.json'))
        return ident
