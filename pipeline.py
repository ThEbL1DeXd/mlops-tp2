import json
import subprocess
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

p = yaml.safe_load(open('params.yaml', encoding='utf-8'))
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp2-pipeline')


def lancer(etape):
    subprocess.run([sys.executable, f'src/{etape}.py'], check=True)


refuse = False

with mlflow.start_run(run_name='pipeline'):
    with mlflow.start_run(run_name='preparer', nested=True):
        lancer('preparer')
        mlflow.log_params(p['preparation'])

    with mlflow.start_run(run_name='entrainer', nested=True):
        lancer('entrainer')
        mlflow.log_params(p['entrainement'])

    with mlflow.start_run(run_name='evaluer', nested=True):
        lancer('evaluer')
        mets = json.load(open('reports/metrics.json'))
        mlflow.log_metrics(mets)
        mlflow.log_artifact('reports/confusion.png')

    seuil = p['validation']['seuil_f1']
    mlflow.log_param('seuil_f1', seuil)
    mlflow.log_metrics(mets)

    if mets['f1'] >= seuil:
        exemple = pd.read_csv('data/prepare/test.csv').drop(columns='fraude').head(3)
        mlflow.sklearn.log_model(
            joblib.load('models/model.pkl'),
            artifact_path='model',
            input_example=exemple,
            registered_model_name='fraude-tp2',
            skops_trusted_types=['sklearn.tree._tree.Tree']
        )
        mlflow.set_tag('statut', 'accepte')
        print(f"modele accepte (F1 {mets['f1']} >= {seuil}) et enregistre")
    else:
        refuse = True
        mlflow.set_tag('statut', 'refuse')

if refuse:
    raise SystemExit(f"modele refuse F1 {mets['f1']} < seuil {seuil}")