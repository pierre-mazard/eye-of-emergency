import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

# Charger les données
df = pd.read_csv('data\\processed\\train_optimized_v3.csv')
df = df.dropna(subset=['text_cleaned', 'keyword', 'target'])

X_text = df['text_cleaned']  # chaîne ou liste de tokens
X_keyword = df[['keyword']]
y = df['target']

# Vectorisation du texte (exemple sans tokenizer personnalisé)
tfidf = TfidfVectorizer()

# Encodage du keyword
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

# Transformer les données
X_text_tfidf = tfidf.fit_transform(X_text)
X_keyword_ohe = onehot.fit_transform(X_keyword)

# Combiner les deux matrices sparse
from scipy.sparse import hstack
X_combined = hstack([X_text_tfidf, X_keyword_ohe])

# Classifieur
clf = LogisticRegression(max_iter=1000)

# Grid des hyperparamètres (sur tfidf et clf)
param_grid = {
    'alpha': [0.5, 1.0],               # pas applicable ici, à enlever ou adapter
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}

# GridSearch sans pipeline nécessite un wrapper pour changer les params
# Donc on crée une classe qui fait fit/predict et prend les hyperparamètres

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Pour simplifier, on fait une gridsearch "manuelle"
results = []
C_values = [0.1, 1, 10]
penalties = ['l2']

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for C in C_values:
    for penalty in penalties:
        scores = []
        for train_idx, val_idx in cv.split(X_combined, y):
            X_train, X_val = X_combined[train_idx], X_combined[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = LogisticRegression(C=C, penalty=penalty, max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores.append(accuracy_score(y_val, preds))
        mean_score = np.mean(scores)
        results.append({'C': C, 'penalty': penalty, 'mean_cv_accuracy': mean_score})

# Trouver la meilleure config
best = max(results, key=lambda x: x['mean_cv_accuracy'])

print("Meilleurs paramètres :", best)
