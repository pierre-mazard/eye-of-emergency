# Dossier data/
Ce dossier contient toutes les données du projet Eye of Emergency :

## Sous-dossiers :

### raw/
- **disaster_tweets.csv** : Dataset brut original téléchargé depuis Kaggle
- Données non modifiées, utilisées comme source de référence

### Fichiers traités :
- **train.csv** : Dataset d'entraînement préprocessé (80% des données)
  - Colonnes : id, text, location, keyword, target
  - Utilisé pour l'entraînement des modèles

- **test.csv** : Dataset de test préprocessé (20% des données) 
  - Colonnes : id, text, location, keyword
  - Utilisé pour l'évaluation finale des modèles


