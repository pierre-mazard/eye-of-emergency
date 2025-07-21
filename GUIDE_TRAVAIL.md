# 📋 Guide de Travail - Eye of Emergency

## 🎯 Objectif
Ce guide présente toutes les étapes à suivre pour mener à bien le projet Eye of Emergency, de la veille théorique à la livraison finale du modèle.

---

## 📚 Phase 1 : Veille NLP 

### ✅ Tâches à effectuer :

1. **Compléter la section "Veille NLP" du README.md**
   - [ ] Text Mining vs Natural Language Processing
   - [ ] Sous-domaines du NLP (sentiment analysis, NER, POS tagging)
   - [ ] Applications concrètes du NLP
   - [ ] Stop-words (définition, importance, exemples)
   - [ ] Traitement des caractères spéciaux et ponctuation
   - [ ] Tokenisation et N-grams
   - [ ] Stemming vs Lemmatisation
   - [ ] Méthodes de vectorisation (Bag of Words, TF-IDF)
   - [ ] Bagging vs Boosting

2. **Créer une présentation de veille**
   - [ ] Synthèse des concepts étudiés
   - [ ] Exemples concrets d'applications
   - [ ] Justification des choix techniques pour le projet

### 📋 Livrables :
- README.md complété (section veille)
- Présentation de veille (PowerPoint/PDF)

---

## 🔧 Phase 2 : Setup et Préparation 

### ✅ Tâches à effectuer :

1. **Installation de l'environnement**
   ```bash
   # Cloner le repository
   git clone https://github.com/pierre-mazard/eye-of-emergency.git
   cd eye-of-emergency
   
   # Installer les dépendances
   pip install -r requirements.txt
   ```

2. **Téléchargement des données**
   - [ ] Télécharger le dataset "Disaster Tweets" depuis Kaggle
   - [ ] Placer le fichier dans `data/raw/disaster_tweets.csv`
   - [ ] Vérifier l'intégrité des données

3. **Configuration des outils NLP**
   - [ ] Télécharger les corpus NLTK nécessaires
   - [ ] Installer le modèle spaCy français/anglais
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   
   # Pour spaCy
   python -m spacy download en_core_web_sm
   ```

### 📋 Livrables :
- Environnement fonctionnel
- Dataset disponible dans `data/raw/`

---

## 🔍 Phase 3 : Exploration des Données 

### ✅ Tâches dans `notebooks/eye_of_emergency_analysis.ipynb` :

1. **Chargement et aperçu des données**
   - [ ] Importer les librairies nécessaires
   - [ ] Charger le dataset depuis `data/raw/disaster_tweets.csv`
   - [ ] Afficher les premières lignes et infos générales
   - [ ] Analyser la structure des données

2. **Analyse des données manquantes**
   - [ ] Identifier les valeurs manquantes par colonne
   - [ ] Visualiser les patterns de données manquantes
   - [ ] Décider des stratégies de traitement

3. **Analyse des doublons**
   - [ ] Détecter les tweets en doublon
   - [ ] Analyser l'impact des doublons
   - [ ] Décider de la stratégie de déduplication

4. **Analyse exploratoire spécifique au NLP**
   - [ ] Distribution des classes (catastrophe vs non-catastrophe)
   - [ ] Distribution des longueurs de tweets
   - [ ] Analyse des mots-clés les plus fréquents
   - [ ] Analyse des localisations
   - [ ] Nuages de mots pour chaque classe
   - [ ] Analyse des caractères spéciaux et emojis

5. **Visualisations**
   - [ ] Graphiques de distribution
   - [ ] WordClouds par classe
   - [ ] Heatmaps de corrélation
   - [ ] Sauvegarder les graphiques dans `results/figures/`

### 📋 Livrables :
- Notebook avec EDA complète
- Graphiques sauvegardés dans `results/figures/`
- Rapport d'analyse des données

---

## 🧹 Phase 4 : Preprocessing 

### ✅ Tâches dans `src/preprocessing.py` :

1. **Créer les fonctions de nettoyage**
   - [ ] Fonction de suppression des URLs
   - [ ] Fonction de suppression des mentions (@username)
   - [ ] Fonction de suppression des hashtags ou conservation
   - [ ] Fonction de suppression des caractères spéciaux
   - [ ] Fonction de conversion en minuscules
   - [ ] Fonction de suppression des emojis (optionnel)

2. **Créer les fonctions de tokenisation**
   - [ ] Fonction de tokenisation simple
   - [ ] Fonction de suppression des stop-words
   - [ ] Fonction de stemming (Porter Stemmer)
   - [ ] Fonction de lemmatisation (spaCy ou NLTK)

3. **Créer le pipeline de preprocessing**
   - [ ] Classe `TextPreprocessor` avec méthodes modulaires
   - [ ] Méthode `fit()` pour ajuster le preprocessing
   - [ ] Méthode `transform()` pour appliquer le preprocessing
   - [ ] Tests unitaires des fonctions

4. **Appliquer le preprocessing dans le notebook**
   - [ ] Importer les fonctions depuis `src/preprocessing.py`
   - [ ] Tester différentes stratégies de preprocessing
   - [ ] Comparer l'impact sur les données
   - [ ] Générer `data/train.csv` et `data/test.csv`

### 📋 Livrables :
- Module `src/preprocessing.py` complet
- Datasets préprocessés dans `data/`
- Documentation des choix de preprocessing

---

## 🤖 Phase 5 : Modélisation 

### ✅ Tâches dans `src/models.py` :

1. **Implémentation de la classe Decision Tree personnalisée**
   - [ ] Classe `CustomDecisionTree` from scratch
   - [ ] Méthodes `fit()`, `predict()`, `score()`
   - [ ] Critères de division (Gini, Entropy)
   - [ ] Gestion de l'overfitting (max_depth, min_samples_split)

2. **Préparation des autres modèles**
   - [ ] Configuration Logistic Regression
   - [ ] Configuration Random Forest
   - [ ] Configuration XGBoost
   - [ ] Configuration SVM

3. **Développement dans le notebook**
   - [ ] Import et vectorisation des textes (TF-IDF, Bag of Words)
   - [ ] Split train/validation/test
   - [ ] Entraînement des 5 modèles
   - [ ] GridSearch pour l'optimisation des hyperparamètres
   - [ ] Cross-validation

4. **Évaluation des modèles**
   - [ ] Calcul des métriques (Accuracy, Precision, Recall, F1-score)
   - [ ] Matrices de confusion
   - [ ] Courbes ROC et calcul AUC
   - [ ] Comparaison des temps d'entraînement
   - [ ] Analyse des erreurs

### 📋 Livrables :
- Module `src/models.py` avec classe Decision Tree
- 5 modèles entraînés et optimisés
- Métriques de performance détaillées

---

## 💾 Phase 6 : Sauvegarde et Documentation 

### ✅ Tâches à effectuer :

1. **Sauvegarde des modèles**
   - [ ] Sauvegarder tous les modèles dans `results/models/`
   - [ ] Sauvegarder les vectoriseurs (TF-IDF, BoW)
   - [ ] Sauvegarder le pipeline de preprocessing
   - [ ] Créer un fichier de métadonnées des modèles

2. **Sauvegarde des résultats**
   - [ ] Export des métriques en JSON/CSV
   - [ ] Sauvegarde des graphiques de performance
   - [ ] Création d'un rapport de comparaison des modèles

3. **Documentation finale**
   - [ ] Compléter le README.md avec les résultats
   - [ ] Documenter les choix techniques
   - [ ] Ajouter des instructions d'utilisation
   - [ ] Créer des exemples d'usage

### 📋 Livrables :
- Modèles sauvegardés dans `results/models/`
- Documentation complète
- README.md finalisé

---

## 📊 Phase 7 : Analyse et Conclusion 

### ✅ Tâches finales :

1. **Analyse comparative des modèles**
   - [ ] Tableau de comparaison des performances
   - [ ] Analyse des forces/faiblesses de chaque modèle
   - [ ] Justification du choix du modèle final
   - [ ] Analyse du rapport performance/complexité

2. **Validation finale**
   - [ ] Test sur le dataset de test final
   - [ ] Validation des performances en conditions réelles
   - [ ] Tests de robustesse

3. **Présentation des résultats**
   - [ ] Création d'une présentation finale
   - [ ] Démonstration du modèle en action
   - [ ] Recommandations pour la production

### 📋 Livrables :
- Présentation finale des résultats
- Modèle recommandé pour la production
- Documentation d'utilisation

---


## ✅ Checklist de Validation

### Avant de passer à la phase suivante :
- [ ] Tous les livrables de la phase sont complétés
- [ ] Le code est testé et fonctionnel
- [ ] La documentation est à jour
- [ ] Les résultats sont sauvegardés
- [ ] Review par un autre membre de l'équipe (si applicable)

### Points de contrôle qualité :
- [ ] Code propre et commenté
- [ ] Respect des conventions de nommage
- [ ] Gestion des erreurs
- [ ] Reproductibilité des résultats
- [ ] Performance acceptable

---

## 🆘 Support et Ressources

### Documentation utile :
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### En cas de blocage :
1. Consulter la documentation officielle
2. Chercher sur Stack Overflow
3. Demander de l'aide à l'équipe
4. Documenter le problème et la solution trouvée

---

## 📝 Notes Importantes

- **Versioning** : Commiter régulièrement avec des messages clairs
- **Backup** : Sauvegarder le travail quotidiennement
- **Communication** : Tenir l'équipe informée des avancées
- **Qualité** : Privilégier la qualité à la rapidité
- **Documentation** : Documenter au fur et à mesure, pas à la fin !

---
