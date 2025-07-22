# Dossier results/models/
Ce dossier contient tous les modèles entraînés et leurs artefacts :

## Structure attendue :

### Modèles sauvegardés :
- **logistic_regression_model.joblib** : Modèle de régression logistique optimisé
- **decision_tree_model.joblib** : Modèle d'arbre de décision personnalisé
- **random_forest_model.joblib** : Modèle Random Forest optimisé
- **xgboost_model.joblib** : Modèle XGBoost optimisé
- **svm_model.joblib** : Modèle Support Vector Machine optimisé

### Vectoriseurs :
- **tfidf_vectorizer.joblib** : Vectoriseur TF-IDF entraîné
- **bow_vectorizer.joblib** : Vectoriseur Bag of Words entraîné

### Métriques et résultats :
- **model_comparison.json** : Comparaison des performances de tous les modèles
- **best_model_metrics.json** : Métriques détaillées du meilleur modèle
- **hyperparameters.json** : Meilleurs hyperparamètres pour chaque modèle

### Pipeline :
- **preprocessing_pipeline.joblib** : Pipeline complet de preprocessing
- **full_pipeline.joblib** : Pipeline complet (preprocessing + modèle final)
