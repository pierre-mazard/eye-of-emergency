# Structure du dossier src/

Ce dossier contient tous les modules Python du projet Eye of Emergency.

## 📋 Prérequis

Avant d'utiliser les modules de ce dossier :

1. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Fichier requis** : Assurez-vous que `data/raw/original_train_tweets.csv` existe

3. **Structure des dossiers** : Les dossiers `data/processed/` seront créés automatiquement

## Fichiers principaux

### `pipeline.py` 🚀
**Fichier principal** - Pipeline complet intégré
- Exécute toutes les étapes dans le bon ordre
- Utilisé par `run_pipeline.py` (depuis la racine du projet)
- **RECOMMANDÉ** : Utiliser ce pipeline pour toute nouvelle exécution
- Gestion automatique des erreurs et validation à chaque étape

### `preprocess_train.py` 🔧
Module de preprocessing optimisé pour l'entraînement
- Classe `OptimizedEmergencyPreprocessor` avec 16 features optimisées
- Feature engineering avancé et nettoyage textuel intelligent
- Basé sur l'analyse de validation V3.1 pour maximiser le pouvoir prédictif
- Utilisé par le pipeline principal (étape 3)

### `preprocess_test.py` 🧽
Module de preprocessing pour les données de test
- **Nettoyage uniquement** (pas de feature engineering pour éviter les fuites)
- Applique la même méthode de nettoyage textuel que le train
- Préserve l'intégrité des données de test
- Utilisé par le pipeline principal (étape 4)

### `models.py` 🤖
Définitions des modèles de machine learning
- **En cours de développement** pour la phase d'entraînement
- Contiendra les classes et fonctions des modèles ML optimisés

## Modules spécialisés (hérités)

Les fichiers suivants ont été intégrés dans `pipeline.py` mais sont conservés pour référence historique et maintenance :

### `clean_original_dataset.py` 🧹
- **Fonction** : Nettoyage du dataset original (suppression des doublons exacts)
- **Statut** : **INTÉGRÉ** dans l'étape 1 du pipeline (`step1_clean_original_dataset`)
- **Usage** : Analyse et suppression des textes dupliqués avec préservation de la distribution des classes

### `create_clean_split.py` ⚖️
- **Fonction** : Partitionnement stratifié propre (90% train / 10% test)
- **Statut** : **INTÉGRÉ** dans l'étape 2 du pipeline (`step2_create_clean_split`)
- **Usage** : Split stratifié avec vérification automatique de l'absence de fuites

### `clean_final_leakage.py` 🔒
- **Fonction** : Nettoyage final des fuites de données après preprocessing
- **Statut** : **INTÉGRÉ** dans l'étape 5 du pipeline (`step5_clean_final_leakage`)
- **Usage** : Suppression des textes en commun entre train et test finaux
- **Avantage** : Version optimisée et plus récente

### `clean_data_leakage.py` ⚠️
- **Fonction** : Version alternative de nettoyage des fuites
- **Statut** : **REMPLACÉ** par `clean_final_leakage.py` (plus efficace)
- **Usage** : Conservé pour référence historique uniquement

## 📁 Fichiers système

### `__pycache__/` 🗂️
- **Contenu** : Cache Python généré automatiquement
- **Statut** : Ignoré par Git (voir `.gitignore`)
- **Maintenance** : Peut être supprimé sans risque (`rm -rf __pycache__`)

### `.gitignore` 🚫
- **Fonction** : Configuration Git pour ignorer les fichiers temporaires
- **Contenu** : Cache Python (*.pyc, __pycache__/), fichiers temporaires

## Usage recommandé

### 🚀 Exécution du pipeline complet (RECOMMANDÉ)

Pour exécuter le pipeline complet depuis la racine du projet :

```bash
# Depuis la racine du projet eye-of-emergency/
python run_pipeline.py
```

**Le pipeline exécute automatiquement dans l'ordre :**

1. 🧹 **Nettoyage du dataset original** (`step1_clean_original_dataset`)
   - Suppression des doublons exacts et normalisés
   - Préservation de la distribution des classes
   - Génération de `original_train_tweets_cleaned.csv`

2. ⚖️ **Partitionnement stratifié** (`step2_create_clean_split`)
   - Split 90% train / 10% test avec stratification
   - Vérification automatique de l'absence de fuites
   - Génération de `train_tweets.csv` et `test_tweets.csv`

3. 🔧 **Preprocessing train** (`step3_preprocess_train`)
   - Feature engineering avec 16 features optimisées
   - Nettoyage textuel intelligent et extraction de patterns
   - Génération de `train_optimized.csv`

4. 🧽 **Preprocessing test** (`step4_preprocess_test`)
   - Nettoyage textuel uniquement (pas de feature engineering)
   - Préservation de l'intégrité des données de test
   - Génération de `test_cleaned.csv`

5. 🗑️ **Nettoyage final des fuites** (`step5_clean_final_leakage`)
   - Détection et suppression des textes communs résiduels
   - Vérification finale de l'intégrité train/test

6. ✅ **Validation de l'intégrité** (`step6_validate_integrity`)
   - Score de qualité final (objectif : >85/100)
   - Rapport complet de validation

### 🔧 Exécution modulaire (pour développement)

Pour exécuter des modules individuellement depuis le dossier `src/` :

```bash
# Exemples d'utilisation modulaire
cd src/

# Nettoyage uniquement du dataset original
python clean_original_dataset.py

# Preprocessing du train uniquement  
python preprocess_train.py

# Nettoyage des fuites uniquement
python clean_final_leakage.py
```

⚠️ **Attention** : L'exécution modulaire nécessite que les étapes précédentes aient été réalisées.

## Avantages du pipeline intégré

- ✅ **Robuste** : Gestion automatique des erreurs à chaque étape avec rollback
- ✅ **Reproductible** : Toujours le même ordre d'exécution avec graines fixes (random_state=42)
- ✅ **Validé** : Score d'intégrité finale de 100/100 garanti
- ✅ **Sans fuites** : Vérification automatique et suppression des fuites à chaque étape
- ✅ **Optimisé** : Basé sur l'analyse de validation V3.1 avec 16 features sélectionnées
- ✅ **Traçable** : Logs détaillés et rapports de progression à chaque étape
- ✅ **Sécurisé** : Backup automatique des fichiers avant modification

## Structure de sortie

### 📁 Fichiers générés dans `data/processed/`

| Fichier | Description | Taille estimée | Usage |
|---------|-------------|----------------|-------|
| `train_optimized.csv` | Dataset d'entraînement avec 16 features optimisées | ~6,000-6,500 échantillons | Entraînement ML |
| `test_cleaned.csv` | Dataset de test nettoyé (sans features) | ~700-750 échantillons | Évaluation finale |

### 📁 Fichiers de sauvegarde dans `data/raw/`

| Fichier | Description | Statut |
|---------|-------------|---------|
| `original_train_tweets_cleaned.csv` | Dataset original sans doublons | Généré par étape 1 |
| `original_train_tweets_with_duplicates.csv` | Backup de l'original | Backup automatique |
| `train_tweets.csv` | Partition train après split | Généré par étape 2 |
| `test_tweets.csv` | Partition test après split | Généré par étape 2 |

## 🔍 Diagnostic et maintenance

### Vérification de l'état du pipeline

```bash
# Vérifier les fichiers générés
ls -la data/processed/
ls -la data/raw/

# Vérifier la qualité des données (depuis Python)
python -c "
import pandas as pd
train = pd.read_csv('data/processed/train_optimized.csv')
test = pd.read_csv('data/processed/test_cleaned.csv')
print(f'Train: {len(train)} échantillons, {len(train.columns)} colonnes')
print(f'Test: {len(test)} échantillons, {len(test.columns)} colonnes')
print(f'Features communes: {set(train.columns) & set(test.columns)}')
"
```

### Nettoyage et redémarrage

```bash
# Nettoyer le cache Python
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Redémarrer le pipeline depuis zéro (ATTENTION: supprime tous les fichiers générés)
rm -f data/processed/*.csv
rm -f data/raw/*_cleaned.csv data/raw/*_with_*.csv
python run_pipeline.py
```

## 📊 Métriques de performance attendues

Après exécution complète du pipeline :

- **Score de qualité** : >85/100 (objectif validé)
- **Intégrité des données** : 100% (0 fuite garantie)
- **Réduction des données** : ~15-20% (suppression doublons + nettoyage)
- **Features finales** : 16 features optimisées pour ML
- **Temps d'exécution** : ~2-5 minutes selon la machine

---

💡 **Conseil** : Utilisez toujours `run_pipeline.py` pour garantir un workflow complet et validé !
