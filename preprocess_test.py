"""
Script de preprocessing pour les données de test
Applique UNIQUEMENT le nettoyage textuel (pas de feature engineering)
"""

import pandas as pd
import sys
import os

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing_optimized import OptimizedEmergencyPreprocessor

def preprocess_test_data():
    """
    Preprocessing des données de test avec UNIQUEMENT le nettoyage textuel
    """
    print("🔄 Début du preprocessing des données de test...")
    
    # Initialisation du preprocessor
    preprocessor = OptimizedEmergencyPreprocessor()
    
    # Chargement des données de test
    test_path = 'data/raw/test_tweets.csv'
    print(f"📁 Chargement de {test_path}")
    
    try:
        test_df = pd.read_csv(test_path)
        print(f"✅ Données chargées: {len(test_df)} échantillons")
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier {test_path} non trouvé")
        return False
    
    # Nettoyage textuel UNIQUEMENT (pas de feature engineering)
    print("🧹 Application du nettoyage textuel...")
    test_df['text_cleaned'] = test_df['text'].apply(preprocessor.clean_text)
    
    # Vérification de la qualité du nettoyage
    empty_texts = test_df['text_cleaned'].str.strip().eq('').sum()
    if empty_texts > 0:
        print(f"⚠️  {empty_texts} textes vides après nettoyage")
    else:
        print("✅ Tous les textes ont été nettoyés avec succès")
    
    # Sauvegarde
    output_path = 'data/processed/test_cleaned.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    test_df.to_csv(output_path, index=False)
    print(f"💾 Données sauvegardées: {output_path}")
    
    # Statistiques finales
    print(f"\n📊 Statistiques finales:")
    print(f"   - Échantillons traités: {len(test_df)}")
    print(f"   - Colonnes: {list(test_df.columns)}")
    print(f"   - Longueur moyenne texte original: {test_df['text'].str.len().mean():.1f}")
    print(f"   - Longueur moyenne texte nettoyé: {test_df['text_cleaned'].str.len().mean():.1f}")
    
    return True

if __name__ == "__main__":
    success = preprocess_test_data()
    if success:
        print("\n🎉 Preprocessing du test terminé avec succès!")
    else:
        print("\n❌ Erreur lors du preprocessing du test")
        sys.exit(1)
