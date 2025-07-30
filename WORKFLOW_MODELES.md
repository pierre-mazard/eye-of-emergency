# 🤖 WORKFLOW D'ENTRAÎNEMENT ET TEST DES MODÈLES - EYE OF EMERGENCY

## 📋 Vue d'ensemble

Ce guide détaille la marche à suivre pour entraîner et tester différents modèles de machine learning sur le dataset Eye of Emergency optimisé et validé.

## 🎯 Objectifs

- **Entraîner** : Plusieurs algorithmes de classification sur les tweets d'urgence
- **Comparer** : Les performances de différents modèles 
- **Optimiser** : Les hyperparamètres pour chaque modèle
- **Valider** : La robustesse et la généralisation des modèles

---

## 📁 Structure des fichiers pour l'entraînement

```
eye-of-emergency/
├── data/processed/
│   ├── train_optimized.csv     # ✅ Features engineering appliqué
│   └── test_cleaned.csv        # ✅ Données propres pour test final
├── results/models/             # 📂 Sauvegarde des modèles entraînés
├── results/figures/            # 📊 Graphiques et métriques
└── src/models.py              # 🧠 Script des modèles (à créer)
```

---

## 🚀 ÉTAPE 1 : Préparation des données

### ✅ Prérequis - Pipeline terminé
```bash
# S'assurer que le pipeline est exécuté
python run_pipeline.py
```

**Fichiers générés requis :**
- `data/processed/train_optimized.csv` (6,185 tweets avec 17 features + metadata)
- `data/processed/test_cleaned.csv` (données test propres, 6 colonnes de base)

### 📊 Chargement et préparation
```python
import pandas as pd
from src.preprocess_train import OptimizedEmergencyPreprocessor

# Chargement des données d'entraînement
train_df = pd.read_csv('data/processed/train_optimized.csv')

# IMPORTANT: Vérification des colonnes disponibles
print(f"📋 Colonnes dans train_optimized.csv: {list(train_df.columns)}")
print(f"📊 Nombre total de colonnes: {len(train_df.columns)}")

# Séparation features/target
# ✅ FEATURES NUMÉRIQUES ET CATÉGORIELLES : keyword + 16 features d'engineering
# ✅ ON GARDE : text_cleaned (utile pour analyse et débogage)
# ❌ ON EXCLUT POUR ML : id (identifiant), target (variable cible)

# D'abord, séparation pour l'entraînement ML (features numériques/catégorielles uniquement)
X_train = train_df.drop(['id', 'target', 'text_cleaned'], axis=1)
y_train = train_df['target']

# Conserver text_cleaned séparément pour analyse
text_train = train_df['text_cleaned']

print(f"📊 Forme des données d'entraînement: {X_train.shape}")
print(f"🎯 Distribution des classes: {y_train.value_counts()}")
print(f"🔧 Features ML utilisées ({X_train.shape[1]}): {list(X_train.columns)}")
print(f"📝 Text_cleaned conservé séparément: {len(text_train)} textes")

# Validation que nous avons bien toutes les features ML
expected_features = 17  # keyword + 16 features d'engineering  
if X_train.shape[1] != expected_features:
    print(f"⚠️  ATTENTION: Expected {expected_features} features, got {X_train.shape[1]}")
else:
    print(f"✅ Parfait! Nous utilisons bien les {expected_features} features ML attendues")
```

---

## 🤖 ÉTAPE 2 : Modèles à entraîner et comparer

### 🏗️ Architecture recommandée

```python
# src/models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class EmergencyModelTrainer:
    def __init__(self):
        self.models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'NaiveBayes': MultinomialNB(),
            'KNN': KNeighborsClassifier()
        }
        self.best_models = {}
        self.results = {}
```

### 📝 Modèles prioritaires
1. **Logistic Regression** - Baseline rapide et interprétable
2. **Random Forest** - Robuste et gère bien les features hétérogènes
3. **Gradient Boosting** - Performances élevées habituelles
4. **XGBoost** - État de l'art pour ce type de données
5. **SVM** - Bon pour la classification de texte
6. **Naive Bayes** - Rapide, bon pour le texte
7. **KNN** - Comparaison locale

---

## ⚙️ GESTION DES FEATURES : DIFFÉRENCE TRAIN vs TEST

### 📊 Comprendre la structure des données

**🔧 train_optimized.csv (20 colonnes) :**
```
id, keyword, target, text_cleaned, 
text_length, word_count, char_count, has_emergency_word, 
emergency_word_count, emergency_density, has_url, url_count, 
has_mention, mention_count, exclamation_count, intense_punctuation, 
avg_word_length, urgency_score, stopword_ratio, keyword_in_text
```

**🧽 test_cleaned.csv (6 colonnes) :**
```
id, keyword, location, text, target, text_cleaned
```

**⚠️ Important pour le test :**
- **`text`** : Texte original (non utilisé pour ML)
- **`text_cleaned`** : Texte nettoyé (utilisé pour feature engineering)
- **`location`** : Information géographique (non utilisée pour ML)
- **`keyword`** : Mot-clé (utilisé comme feature)
- **`target`** : Labels (pour validation finale)

### 🎯 Stratégie de feature engineering

1. **TRAIN** : Features déjà calculées et stockées (17 features ML + text_cleaned conservé)
2. **TEST** : Features calculées en temps réel lors de la prédiction  
3. **COHÉRENCE** : Mêmes 17 features ML pour train et test
4. **ANALYSE** : text_cleaned disponible pour debugging et interprétation

### ⚠️ Différences importantes TRAIN vs TEST

**TRAIN (train_optimized.csv) :**
- ✅ Features déjà calculées et sauvegardées
- ✅ Utilise `text_cleaned` pour l'analyse
- ✅ Prêt pour l'entraînement direct

**TEST (test_cleaned.csv) :**
- ❌ Pas de features d'engineering pré-calculées  
- ✅ Utilise `text_cleaned` pour générer les features en temps réel
- ✅ Contient `location` (non utilisée pour ML)
- ⚠️ **IMPORTANT** : Ne jamais utiliser `text` (raw), toujours `text_cleaned`

### ✅ Features ML utilisées (17 au total)

**Pour l'entraînement des modèles :**

1. **keyword** (categorical)
2. **text_length** (numeric) 
3. **word_count** (numeric)
4. **char_count** (numeric)
5. **has_emergency_word** (boolean)
6. **emergency_word_count** (numeric)
7. **emergency_density** (numeric)
8. **has_url** (boolean)
9. **url_count** (numeric)
10. **has_mention** (boolean)
11. **mention_count** (numeric)
12. **exclamation_count** (numeric)
13. **intense_punctuation** (numeric)
14. **avg_word_length** (numeric)
15. **urgency_score** (numeric)
16. **stopword_ratio** (numeric)
17. **keyword_in_text** (boolean)

### 📝 Données additionnelles conservées

**text_cleaned** : Texte préprocessé conservé séparément pour :
- 🔍 **Analyse des erreurs** : Comprendre pourquoi un modèle se trompe
- 🐛 **Débogage** : Vérifier la qualité du preprocessing  
- 📊 **Interprétation** : Analyser les patterns dans les prédictions
- 📈 **Visualisation** : Créer des word clouds et analyses textuelles
- 🧪 **Validation** : Vérifier la cohérence des features extraites

```python
# Exemple d'usage de text_cleaned pour analyse
def analyze_predictions(y_true, y_pred, text_cleaned, X_train):
    """Analyse des prédictions avec le texte original"""
    
    # Identifier les erreurs
    errors = y_true != y_pred
    error_texts = text_cleaned[errors]
    error_features = X_train[errors]
    
    print(f"📊 Nombre d'erreurs: {errors.sum()}")
    print(f"📝 Exemples de textes mal classés:")
    for i, text in enumerate(error_texts.head()):
        print(f"   {i+1}. {text}")
    
    return error_texts, error_features
```

---

## ⚙️ ÉTAPE 3 : Entraînement et optimisation

### 🔄 Cross-validation et grid search
```python
def train_and_optimize_models(X_train, y_train):
    """Entraîne et optimise tous les modèles"""
    
    # Hyperparamètres à tester
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1, 10, 100],
            'max_iter': [1000, 2000]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    results = {}
    for model_name, model in models.items():
        print(f"🔄 Entraînement {model_name}...")
        
        if model_name in param_grids:
            # Grid search avec cross-validation
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=5,
                scoring='f1_macro',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
        else:
            # Cross-validation simple
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            model.fit(X_train, y_train)
            best_model = model
            best_score = scores.mean()
        
        results[model_name] = {
            'model': best_model,
            'cv_score': best_score,
            'std': scores.std() if 'scores' in locals() else 0
        }
        
        # Sauvegarde du modèle
        joblib.dump(best_model, f'results/models/{model_name}_best.pkl')
        print(f"✅ {model_name}: CV Score = {best_score:.4f}")
    
    return results
```

---

## 📊 ÉTAPE 4 : Évaluation et comparaison

### 🏆 Métriques de performance
```python
def evaluate_models(results, X_train, y_train):
    """Évalue et compare tous les modèles entraînés"""
    
    evaluation_results = []
    
    for model_name, result in results.items():
        model = result['model']
        
        # Prédictions sur train (pour détecter overfitting)
        y_pred_train = model.predict(X_train)
        
        # Métriques détaillées
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(y_train, y_pred_train)
        precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_pred_train, average='macro')
        
        evaluation_results.append({
            'Model': model_name,
            'CV_Score': result['cv_score'],
            'CV_Std': result['std'],
            'Train_Accuracy': accuracy,
            'Train_Precision': precision,
            'Train_Recall': recall,
            'Train_F1': f1
        })
    
    # DataFrame des résultats
    results_df = pd.DataFrame(evaluation_results)
    results_df = results_df.sort_values('CV_Score', ascending=False)
    
    print("🏆 CLASSEMENT DES MODÈLES:")
    print(results_df.to_string(index=False))
    
    # Sauvegarde
    results_df.to_csv('results/models_comparison.csv', index=False)
    
    return results_df

def analyze_model_errors(best_model, X_train, y_train, text_train, model_name):
    """Analyse détaillée des erreurs avec text_cleaned"""
    
    y_pred = best_model.predict(X_train)
    errors = y_train != y_pred
    
    print(f"\n🔍 ANALYSE DES ERREURS - {model_name}")
    print(f"📊 Nombre total d'erreurs: {errors.sum()}/{len(y_train)} ({errors.mean()*100:.2f}%)")
    
    if errors.sum() > 0:
        print(f"\n📝 Exemples de textes mal classés:")
        error_indices = errors[errors].index[:5]  # 5 premiers exemples
        
        for i, idx in enumerate(error_indices):
            true_label = y_train.iloc[idx]
            pred_label = y_pred[idx]
            text = text_train.iloc[idx]
            
            print(f"\n   {i+1}. Vrai: {true_label}, Prédit: {pred_label}")
            print(f"      Texte: {text[:100]}...")
    
    return errors
```

### 📈 Visualisations de performance
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_df):
    """Génère les graphiques de comparaison"""
    
    # 1. Comparaison CV Score
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Model', y='CV_Score')
    plt.title('🏆 Comparaison des performances (Cross-Validation F1-Score)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/figures/models_cv_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Train vs CV Score (détection overfitting)
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['CV_Score'], results_df['Train_F1'], s=100)
    for i, model in enumerate(results_df['Model']):
        plt.annotate(model, (results_df['CV_Score'].iloc[i], results_df['Train_F1'].iloc[i]))
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.xlabel('CV F1-Score')
    plt.ylabel('Train F1-Score')
    plt.title('🎯 Détection d\'overfitting : Train vs CV Performance')
    plt.tight_layout()
    plt.savefig('results/figures/overfitting_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 🧪 ÉTAPE 5 : Test final sur données test

### ⚠️ Important : Feature engineering en temps réel sur test
```python
def test_final_model(best_model_name, test_file='data/processed/test_cleaned.csv'):
    """Test final du meilleur modèle sur données test"""
    
    # Chargement du meilleur modèle
    best_model = joblib.load(f'results/models/{best_model_name}_best.pkl')
    
    # ÉTAPE 1: Chargement données test (propres, sans feature engineering)
    test_df = pd.read_csv(test_file)
    print(f"📊 Test data shape: {test_df.shape}")
    print(f"📋 Test columns: {list(test_df.columns)}")
    
    # VALIDATION: Vérifier que les colonnes attendues sont présentes
    required_columns = ['id', 'keyword', 'text_cleaned']
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    
    if missing_columns:
        print(f"❌ ERREUR: Colonnes manquantes dans le test: {missing_columns}")
        print(f"   Colonnes disponibles: {list(test_df.columns)}")
        return None
    
    print(f"✅ Colonnes requises présentes: {required_columns}")
    
    # Vérifier la qualité des données
    empty_texts = test_df['text_cleaned'].isna().sum()
    if empty_texts > 0:
        print(f"⚠️  {empty_texts} textes vides dans text_cleaned")
    else:
        print(f"✅ Tous les {len(test_df)} textes sont disponibles")
    
    # ÉTAPE 2: Feature engineering en temps réel sur test
    # CRITIQUE: Le test n'a que 6 colonnes de base, nous devons générer les 17 features
    # On utilise 'text_cleaned' (pas 'text') car c'est le texte preprocessé
    preprocessor = OptimizedEmergencyPreprocessor()
    
    test_features = []
    print("🔄 Application du feature engineering en temps réel...")
    
    for idx, row in test_df.iterrows():
        # IMPORTANT: Utiliser text_cleaned (pas text) et keyword du fichier test
        # Structure test_cleaned.csv: id, keyword, location, text, target, text_cleaned
        
        # Extraction des 16 features d'engineering à partir du texte NETTOYÉ
        engineered_features = preprocessor.extract_optimized_features(row['text_cleaned'])
        
        # Ajout du keyword comme feature (pour correspondre au train)
        full_features = {'keyword': row['keyword']}
        full_features.update(engineered_features)
        
        test_features.append(full_features)
        
        if idx % 500 == 0:
            print(f"   Processed {idx+1}/{len(test_df)} samples...")
    
    # ÉTAPE 3: Création du DataFrame avec les mêmes colonnes que X_train
    X_test = pd.DataFrame(test_features)
    
    # VALIDATION CRITIQUE: Vérifier que les colonnes correspondent exactement
    # Charger X_train pour comparaison
    train_df = pd.read_csv('data/processed/train_optimized.csv')
    X_train_columns = train_df.drop(['id', 'target', 'text_cleaned'], axis=1).columns.tolist()
    
    print(f"🔍 VALIDATION DES FEATURES:")
    print(f"   Train features ({len(X_train_columns)}): {X_train_columns}")
    print(f"   Test features ({len(X_test.columns)}): {list(X_test.columns)}")
    
    # Réorganiser les colonnes de test pour correspondre exactement à train
    X_test = X_test[X_train_columns]
    
    if list(X_test.columns) == X_train_columns:
        print("✅ PARFAIT! Les features test correspondent exactement au train")
    else:
        print("❌ ERREUR! Mismatch entre features train et test")
        return None
    
    print(f"📊 Forme finale X_test: {X_test.shape}")
    
    # ÉTAPE 4: Prédictions finales
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)
    
    # Sauvegarde des prédictions
    test_predictions = test_df.copy()
    test_predictions['predicted_emergency'] = y_pred_test
    test_predictions['probability_emergency'] = y_proba_test[:, 1]
    
    test_predictions.to_csv('results/test_predictions.csv', index=False)
    
    print(f"✅ Prédictions test sauvegardées dans results/test_predictions.csv")
    print(f"📊 Distribution des prédictions: {pd.Series(y_pred_test).value_counts()}")
    
    return test_predictions
```

---

## 📋 ÉTAPE 6 : Workflow complet d'exécution

### 🚀 Script principal recommandé
```bash
# 1. S'assurer que le pipeline est terminé
python run_pipeline.py

# 2. Créer le script de modèles
# Copier le code ci-dessus dans src/models.py

# 3. Entraîner tous les modèles
python -c "
from src.models import *
import pandas as pd

# Chargement données avec validation complète
train_df = pd.read_csv('data/processed/train_optimized.csv')
print(f'📋 Train columns: {list(train_df.columns)}')

# Séparation features/target (features ML uniquement)
X_train = train_df.drop(['id', 'target', 'text_cleaned'], axis=1)
y_train = train_df['target']
text_train = train_df['text_cleaned']  # Conservé pour analyse

print(f'✅ Features ML shape: {X_train.shape}')
print(f'📊 Features ML list: {list(X_train.columns)}')
print(f'📝 Text data conservé: {len(text_train)} textes')

# Entraînement
trainer = EmergencyModelTrainer()
results = trainer.train_and_optimize_models(X_train, y_train)

# Évaluation
eval_results = trainer.evaluate_models(results, X_train, y_train)
trainer.plot_model_comparison(eval_results)

# Analyse d'erreurs du meilleur modèle
best_model_name = eval_results.iloc[0]['Model']
best_model = results[best_model_name]['model']
trainer.analyze_model_errors(best_model, X_train, y_train, text_train, best_model_name)

print('🎉 Entraînement et analyse terminés!')
"

# 4. Test final du meilleur modèle
python -c "
from src.models import test_final_model
test_final_model('XGBoost')  # Remplacer par le meilleur modèle
"
```

---

## 📊 Livrables attendus

### 📁 Fichiers générés
```
results/
├── models/                          # 🤖 Modèles entraînés
│   ├── LogisticRegression_best.pkl
│   ├── RandomForest_best.pkl
│   ├── XGBoost_best.pkl
│   └── ...
├── figures/                         # 📈 Visualisations
│   ├── models_cv_comparison.png
│   ├── overfitting_detection.png
│   └── confusion_matrices/
├── models_comparison.csv            # 📋 Comparaison détaillée
└── test_predictions.csv             # 🎯 Prédictions finales
```

### 📈 Métriques clés à analyser
1. **F1-Score macro** (métrique principale)
2. **Précision/Rappel** par classe
3. **Accuracy** globale  
4. **Temps d'entraînement**
5. **Robustesse** (écart-type CV)
6. **Overfitting** (écart train/validation)

---

## ✅ Checklist de validation

- [ ] Pipeline de données exécuté avec succès
- [ ] 7 modèles différents entraînés et optimisés
- [ ] Cross-validation 5-fold sur tous les modèles
- [ ] Hyperparamètres optimisés par grid search
- [ ] Métriques de performance calculées et sauvegardées
- [ ] Visualisations de comparaison générées
- [ ] Détection d'overfitting effectuée
- [ ] Test final sur données test avec feature engineering temps réel
- [ ] Prédictions finales sauvegardées
- [ ] Documentation des résultats complète

---

## 🔮 Prochaines étapes suggérées

1. **Ensemble Methods** : Combiner les meilleurs modèles
2. **Feature Selection** : Identifier les features les plus importantes
3. **Explicabilité** : SHAP values pour interpréter les prédictions
4. **Déploiement** : API REST pour prédictions en temps réel
5. **Monitoring** : Surveillance de la dérive des données

---

**📝 Note importante** : Ce workflow garantit l'intégrité des données en appliquant le feature engineering uniquement au moment des prédictions sur les données test, évitant ainsi toute fuite de données.
