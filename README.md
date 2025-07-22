# Eye of Emergency 🚨

Développement d'un modèle d'apprentissage automatique capable de classer des tweets signalant des catastrophes naturelles réelles pour aider les intervenants d'urgence et le public à accéder à des informations précises et fiables en période de crise.

## 📋 Table des matières

- [Contexte du projet](#contexte-du-projet)
- [Données](#données)
- [Veille NLP](#veille-nlp)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Méthodologie](#méthodologie)
- [Algorithmes utilisés](#algorithmes-utilisés)
- [Résultats et analyse](#résultats-et-analyse)
- [Conclusion](#conclusion)

## 🎯 Contexte du projet

L'objectif de ce projet est de développer un modèle d'apprentissage automatique capable de distinguer les tweets informatifs concernant des catastrophes naturelles réelles de ceux qui ne le sont pas. Cette classification automatique permettra d'aider les services d'urgence à identifier rapidement les informations fiables lors de situations de crise.

## 📊 Données

Le projet utilise le dataset **Disaster Tweets** qui contient les colonnes suivantes :

- **id** : Identifiant unique pour chaque tweet
- **text** : Contenu textuel du tweet
- **location** : Localisation d'où a été envoyé le tweet
- **keyword** : Mot-clé associé au tweet
- **target** : Variable cible indiquant si un tweet concerne une catastrophe réelle (1) ou non (0)

## 🧠 Veille NLP

### 1. Text Mining vs Natural Language Processing
- **Text Mining** : [À compléter]
- **NLP** : [À compléter]
- **Points communs** : [À compléter]
- **Différences** : [À compléter]

### 2. Sous-domaines du NLP
- **Analyse de sentiments** : [À compléter]
- **Named Entity Recognition (NER)** : [À compléter]
- **Part-of-Speech (POS) Tagging** : [À compléter]

### 3. Applications concrètes du NLP
[À compléter]

### 4. Stop-words
- **Définition** : [À compléter]
- **Importance de leur suppression** : [À compléter]
- **Exemples** : [À compléter]

### 5. Traitement des caractères spéciaux et ponctuation
[À compléter]

### 6. Tokenisation et N-grams
- **Token** : [À compléter]
- **N-gram** : [À compléter]
- **Processus de tokenisation** : [À compléter]

### 7. Stemming vs Lemmatisation
- **Stemming** : [À compléter]
- **Lemmatisation** : [À compléter]
- **Différences** : [À compléter]
- **Cas d'usage** : [À compléter]

### 8. Méthodes de vectorisation
- **Bag of Words** : [À compléter]
- **TF-IDF** : [À compléter]

## 🔧 Prérequis

- Python 3.7+
- Jupyter Notebook
- Bibliothèques Python : pandas, numpy, scikit-learn, nltk, matplotlib, seaborn ... 

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/pierre-mazard/eye-of-emergency.git
cd eye-of-emergency

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter Notebook
jupyter notebook
```

## 📁 Structure du projet

```
eye-of-emergency/
├── assets/
│   ├── images/                        # Images pour la documentation (README, etc.)
│   └── README.md                      # Documentation des ressources visuelles
├── data/
│   ├── raw/
│   │   └── disaster_tweets.csv        # Dataset brut original
│   ├── train.csv                      # Dataset d'entraînement préprocessé
│   ├── test.csv                       # Dataset de test préprocessé
│   └── README.md                      # Documentation du dossier data
├── notebooks/
│   └── eye_of_emergency_analysis.ipynb 
├── src/
│   ├── preprocessing.py
│   └── models.py
├── results/
│   ├── figures/
│   │   └── README.md                  # Documentation des visualisations
│   └── models/
│       └── README.md                  # Documentation des modèles sauvegardés
├── requirements.txt
└── README.md
```

## 🔍 Méthodologie

### 1. Veille NLP
Étude des concepts fondamentaux du traitement du langage naturel

### 2. Exploration des données
- Analyse des données manquantes et doublons
- Visualisations spécifiques au NLP (nuages de mots, distribution des longueurs, etc.)

### 3. Préprocessing
- Nettoyage des données textuelles
- Pipeline de preprocessing personnalisé

### 4. Modélisation
- Développement et comparaison de 5 modèles
- Optimisation par GridSearch

### 5. Évaluation
- Métriques d'évaluation avec optimisation du F1-score

## 🤖 Algorithmes utilisés

### Modèles de classification étudiés

1. **Régression Logistique**
   - [À compléter]

2. **Decision Tree**
   - Implémentation d'une classe Python personnalisée
   - [À compléter]

3. **Random Forest**
   - [À compléter]

4. **XGBoost**
   - [À compléter]

5. **Support Vector Machine (SVM)**
   - [À compléter]

### Bagging vs Boosting
- **Bagging** : [À compléter]
- **Boosting** : [À compléter]

## 📈 Résultats et analyse

### Analyse exploratoire des données
- **Données manquantes** : [À compléter]
- **Doublons** : [À compléter]
- **Visualisations NLP** : [À compléter - nuages de mots, distribution des longueurs, etc.]

### Performance des modèles
- **Meilleur modèle** : [À compléter]
- **Métriques d'évaluation** :
  - **Accuracy** : [À compléter]
  - **Precision** : [À compléter]
  - **Recall** : [À compléter]
  - **F1-score** : [À compléter]
- **Matrice de confusion** : [À compléter]
- **Rapport de classification** : [À compléter]

## 🎯 Conclusion

### Comparaison des modèles
[À compléter - analyse comparative des 5 modèles]

### Modèle sélectionné
[À compléter - justification du choix final]

### Efficacité et performance
[À compléter - évaluation de l'efficacité du modèle choisi] 

