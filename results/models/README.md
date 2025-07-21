# Dossier results/models/
Ce dossier contient tous les modèles entraînés et leurs artefacts :

## Structure attendue :

### Modèles sauvegardés :
- **logistic_regression_model.pkl** : Modèle de régression logistique optimisé
- **decision_tree_model.pkl** : Modèle d'arbre de décision personnalisé
- **random_forest_model.pkl** : Modèle Random Forest optimisé
- **xgboost_model.pkl** : Modèle XGBoost optimisé
- **svm_model.pkl** : Modèle Support Vector Machine optimisé

### Vectoriseurs :
- **tfidf_vectorizer.pkl** : Vectoriseur TF-IDF entraîné
- **bow_vectorizer.pkl** : Vectoriseur Bag of Words entraîné

### Métriques et résultats :
- **model_comparison.json** : Comparaison des performances de tous les modèles
- **best_model_metrics.json** : Métriques détaillées du meilleur modèle
- **hyperparameters.json** : Meilleurs hyperparamètres pour chaque modèle

### Pipeline :
- **preprocessing_pipeline.pkl** : Pipeline complet de preprocessing
- **full_pipeline.pkl** : Pipeline complet (preprocessing + modèle final)
