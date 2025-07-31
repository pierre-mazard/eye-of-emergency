# 📁 Dossier data/ - Eye of Emergency

Ce dossier contient toutes les données du projet Eye of Emergency, organisées selon le workflow du pipeline intégré.

---

## 📂 **Structure des sous-dossiers**

### `raw/` - Données brutes et intermédiaires
Fichiers produits par les étapes 1-2 du pipeline (nettoyage et partitionnement)

| **Fichier** | **Description** | **Taille** | **Étape pipeline** |
|-------------|-----------------|------------|-------------------|
| `original_train_tweets.csv` | **Dataset source** - Données brutes originales | 7,613 tweets | Input initial |
| `original_train_tweets_with_duplicates.csv` | **Backup automatique** avant nettoyage | 7,613 tweets | Sauvegarde étape 1 |
| `original_train_tweets_cleaned.csv` | **Nettoyé des doublons** (111 suppressions) | 7,502 tweets | Sortie étape 1 |
| `train_tweets.csv` | **Partition train** après split stratifié 90/10 | 6,753 tweets | Sortie étape 2 |
| `test_tweets.csv` | **Partition test** après split stratifié 90/10 | 749 tweets | Sortie étape 2 |

### `processed/` - Données finales optimisées  
Fichiers produits par les étapes 3-5 du pipeline (preprocessing et validation)

| **Fichier** | **Description** | **Taille** | **Étape pipeline** |
|-------------|-----------------|------------|-------------------|
| `train_optimized.csv` | **🎯 DATASET FINAL TRAIN** - 16 features optimisées | 6,185 tweets × 20 colonnes | Sortie étape 3+5 |
| `test_cleaned.csv` | **🎯 DATASET FINAL TEST** - Texte nettoyé uniquement | 749 tweets × 6 colonnes | Sortie étape 4 |
| `train_optimized_with_leaks.csv` | **Backup automatique** avant nettoyage final des fuites | 6,185 tweets × 20 colonnes | Sauvegarde étape 5 |

---

## 🔄 **Workflow de génération**

### **Étapes automatiques (via `run_pipeline.py`)**

```mermaid
original_train_tweets.csv (7,613)
    ↓ Étape 1: Nettoyage doublons
original_train_tweets_cleaned.csv (7,502)
    ↓ Étape 2: Split stratifié 90/10
train_tweets.csv (6,753) + test_tweets.csv (749)
    ↓ Étape 3: Feature engineering train
train_optimized.csv (6,185 × 20)
    ↓ Étape 4: Nettoyage test
test_cleaned.csv (749 × 6)
    ↓ Étape 5: Nettoyage fuites finales
train_optimized.csv (mis à jour, 0 fuite garantie)
```

### **Commande de génération**
```bash
# Depuis la racine du projet
python run_pipeline.py
```

---

## 📊 **Datasets finaux pour ML**

### **🚂 Train : `processed/train_optimized.csv`**
- **Usage** : Entraînement des modèles de machine learning
- **Taille** : 6,185 tweets × 20 colonnes
- **Distribution** : 59.2% non-urgence, 40.8% urgence (ratio 1.45 optimal)
- **Features** : 16 optimisées + 4 métadonnées (id, keyword, target, text_cleaned)
- **Qualité** : Score d'intégrité 100/100, zéro fuite garantie

**Features ML (16) :**
```
• Urgence/Contenu:     has_emergency_word, emergency_word_count, emergency_density, urgency_score
• Structure/Liens:     has_url, url_count, has_mention, mention_count  
• Métriques textuelles: text_length, word_count, char_count, avg_word_length
• Signaux émotionnels: exclamation_count, intense_punctuation
• Analyse linguistique: stopword_ratio
• Cohérence metadata:  keyword_in_text
```

### **🧪 Test : `processed/test_cleaned.csv`**
- **Usage** : Évaluation finale des modèles entraînés
- **Taille** : 749 tweets × 6 colonnes  
- **Distribution** : 57.4% non-urgence, 42.6% urgence
- **Contenu** : Texte nettoyé uniquement (pas de features pour éviter les fuites)
- **Colonnes** : id, keyword, location, text, target, text_cleaned

---

## ✅ **Validation et intégrité**

### **Tests automatiques réussis**
- ✅ **Aucune fuite de données** entre train et test (validation automatique)
- ✅ **Distribution préservée** après chaque transformation  
- ✅ **Reproductibilité garantie** (random_state=42 fixe)
- ✅ **Backups automatiques** avant chaque modification critique

### **Métriques de qualité**
- **Réduction contrôlée** : 7,613 → 6,185 tweets (-18.7% stratégique)
- **Features optimisées** : 16 sélectionnées pour maximiser le pouvoir prédictif
- **Intégrité parfaite** : Score 100/100 avec validation automatique
- **Pipeline reproductible** : Même résultat à chaque exécution

---

## 🚀 **Utilisation pour ML**

### **Chargement des données finales**
```python
import pandas as pd

# Dataset d'entraînement (avec features)
train_df = pd.read_csv('data/processed/train_optimized.csv')
print(f"Train: {len(train_df)} tweets, {len(train_df.columns)} colonnes")

# Dataset de test (pour évaluation finale)
test_df = pd.read_csv('data/processed/test_cleaned.csv') 
print(f"Test: {len(test_df)} tweets, {len(test_df.columns)} colonnes")

# Features pour ML (excluant métadonnées)
feature_cols = [col for col in train_df.columns 
                if col not in ['id', 'keyword', 'target', 'text_cleaned']]
X_train = train_df[feature_cols]
y_train = train_df['target']
```

### **Prochaines étapes recommandées**
1. **Entraînement ML immédiat** : Utiliser `train_optimized.csv` avec les 16 features
2. **Validation croisée** : Appliquer sur les 16 features optimisées du train uniquement
3. **Évaluation finale** : 
   - Charger `test_cleaned.csv` (texte nettoyé seulement)
   - Appliquer le **même feature engineering** qu'au train au moment de la prédiction
   - ⚠️ **IMPORTANT** : Ne jamais pré-calculer les features sur le test (risque de fuite)
4. **Production** : Pipeline complet = nettoyage + feature engineering + prédiction en temps réel

### **⚠️ Point critique : Feature engineering sur le test**
```python
# ❌ INCORRECT : Features pré-calculées sur le test (risque de fuite)
# test_with_features = apply_feature_engineering(test_df)  # JAMAIS !

# ✅ CORRECT : Feature engineering au moment de la prédiction
import pandas as pd
import joblib
from src.preprocess_train import OptimizedEmergencyPreprocessor

# 1. Charger le modèle entraîné et le preprocessor
model = joblib.load('trained_model.pkl')
preprocessor = OptimizedEmergencyPreprocessor()

# 2. Charger le test (texte nettoyé uniquement)
test_df = pd.read_csv('data/processed/test_cleaned.csv')

# 3. Feature engineering + prédiction en temps réel (approche optimisée)
predictions = []
feature_names = ['text_length', 'word_count', 'char_count', 'has_emergency_word', 
                'emergency_word_count', 'emergency_density', 'has_url', 'url_count',
                'has_mention', 'mention_count', 'exclamation_count', 'intense_punctuation',
                'avg_word_length', 'urgency_score', 'stopword_ratio', 'keyword_in_text']

for _, row in test_df.iterrows():
    # Feature engineering en temps réel (pas de stockage permanent)
    features_dict = preprocessor.extract_optimized_features(row)
    # Convertir en format attendu par le modèle
    X_test = pd.DataFrame([[features_dict[col] for col in feature_names]], 
                         columns=feature_names)
    pred = model.predict(X_test)[0]
    predictions.append(pred)

# 4. Évaluation finale
y_true = test_df['target']
accuracy = (predictions == y_true).mean()
print(f"Accuracy finale: {accuracy:.3f}")
```

### **🎯 Résumé de l'approche correcte**
- ✅ **Train** : Features pré-calculées et stockées (`train_optimized.csv`)
- ✅ **Test** : Texte nettoyé uniquement (`test_cleaned.csv`) - **JAMAIS de features pré-calculées**
- ✅ **Évaluation** : Feature engineering appliqué **dynamiquement** lors des prédictions
- ✅ **Production** : Même approche = nettoyage + feature engineering + prédiction en temps réel

---

**📋 RÉSUMÉ : Datasets Eye of Emergency V3.1 prêts pour production ML avec intégrité parfaite et features optimisées**


