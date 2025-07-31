#!/usr/bin/env python3
"""
Script pour exécuter le pipeline complet Eye of Emergency
Intègre toutes les étapes dans le bon ordre : nettoyage, partitionnement, preprocessing, validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import run_complete_pipeline

def main():
    """Exécute le pipeline complet Eye of Emergency"""
    
    print("🚀 LANCEMENT DU PIPELINE COMPLET EYE OF EMERGENCY")
    print("=" * 70)
    print("🎯 Objectif: Pipeline de données propre et sans fuites")
    print("� Étapes complètes:")
    print("   1. 🧹 Nettoyage du dataset original (doublons)")
    print("   2. ⚖️  Partitionnement stratifié propre")
    print("   3. 🔧 Preprocessing train (feature engineering)")
    print("   4. 🧽 Preprocessing test (nettoyage uniquement)")
    print("   5. 🗑️  Nettoyage final des fuites")
    print("   6. ✅ Validation de l'intégrité")
    print()
    print("🏆 AVANTAGES DU PIPELINE INTÉGRÉ:")
    print("   ✅ Exécution dans le bon ordre automatiquement")
    print("   ✅ Gestion des erreurs à chaque étape")
    print("   ✅ Validation de l'intégrité finale")
    print("   ✅ Aucune fuite de données garantie")
    print("   🎯 Pipeline reproductible et robuste")
    print()
    
    # Vérification des dépendances
    print("🔍 Vérification des fichiers requis...")
    required_file = 'data/raw/original_train_tweets.csv'
    
    if not os.path.exists(required_file):
        print(f"❌ Erreur: Fichier requis non trouvé: {required_file}")
        print("➡️  Assurez-vous que le fichier original_train_tweets.csv est dans data/raw/")
        return False
    
    print(f"✅ Fichier requis trouvé: {required_file}")
    
    try:
        # Exécution du pipeline complet
        success = run_complete_pipeline()
        
        if success:
            print(f"\n🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
            print("=" * 70)
            print("📊 RÉSULTATS:")
            print("   ✅ Dataset original nettoyé des doublons")
            print("   ✅ Partitionnement stratifié sans fuites")
            print("   ✅ Train preprocessé avec feature engineering")
            print("   ✅ Test nettoyé (pas de feature engineering)")
            print("   ✅ Fuites finales supprimées")
            print("   ✅ Intégrité validée")
            print()
            print("📁 FICHIERS GÉNÉRÉS:")
            print("   📊 data/processed/train_optimized.csv")
            print("   📊 data/processed/test_cleaned.csv")
            print()
            print("🚀 ÉTAPES SUIVANTES:")
            print("   1. Vérifier les fichiers générés")
            print("   2. Lancer l'entraînement des modèles")
            print("   3. Évaluer les performances")
            print("   4. Comparer avec les versions précédentes")
            
        else:
            print(f"\n❌ ÉCHEC DU PIPELINE")
            print("➡️  Vérifiez les erreurs ci-dessus et réessayez")
            return False
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution du pipeline: {str(e)}")
        print(f"   Détail de l'erreur: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    print("\n✅ Script terminé avec succès!")
