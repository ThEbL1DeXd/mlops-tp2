import json
import os

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt

test = pd.read_csv('data/prepare/test.csv')
X, y = test.drop(columns='fraude'), test['fraude']

modele = joblib.load('models/model.pkl')
pred = modele.predict(X)

mets = {
    'f1': round(float(f1_score(y, pred)), 4),
    'precision': round(float(precision_score(y, pred)), 4),
    'recall': round(float(recall_score(y, pred)), 4)
}

os.makedirs('reports', exist_ok=True)
json.dump(mets, open('reports/metrics.json', 'w'), indent=2)

ConfusionMatrixDisplay.from_predictions(y, pred)
plt.savefig('reports/confusion.png', dpi=110, bbox_inches='tight')
print('evaluation', mets)