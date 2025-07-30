"""
Script de nettoyage final des fuites de données
Supprime les échantillons en commun entre train et test après preprocessing
"""

import pandas as pd
import os

def clean_final_leakage():
    """
    Supprime les fuites finales après preprocessing
    """
    print("🧹 NETTOYAGE FINAL DES FUITES")
    print("=" * 40)
    
    # Chargement des datasets finaux
    train_path = 'data/processed/train_optimized.csv'
    test_path = 'data/processed/test_cleaned.csv'
    
    print("📁 Chargement des datasets finaux...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Train: {len(train_df)} échantillons")
    print(f"   Test: {len(test_df)} échantillons")
    
    # Identification des textes en commun
    print("\n🔍 Identification des fuites...")
    train_texts = set(train_df['text_cleaned'].str.strip().str.lower())
    test_texts = set(test_df['text_cleaned'].str.strip().str.lower())
    
    common_texts = train_texts.intersection(test_texts)
    print(f"   Textes en commun: {len(common_texts)}")
    
    if len(common_texts) == 0:
        print("✅ Aucune fuite - Aucun nettoyage nécessaire")
        return True
    
    # Suppression des fuites du train (préserver la taille du test)
    print(f"\n🗑️  Suppression des fuites du dataset d'entraînement...")
    train_texts_col = train_df['text_cleaned'].str.strip().str.lower()
    mask_clean = ~train_texts_col.isin(common_texts)
    train_df_clean = train_df[mask_clean].copy()
    
    removed = len(train_df) - len(train_df_clean)
    print(f"   Échantillons supprimés: {removed}")
    print(f"   Train final: {len(train_df_clean)} échantillons")
    
    # Vérification de la distribution
    print(f"\n📊 Distribution des classes:")
    orig_dist = train_df['target'].value_counts(normalize=True)
    clean_dist = train_df_clean['target'].value_counts(normalize=True)
    
    print(f"   Avant: Classe 0={orig_dist[0]:.1%}, Classe 1={orig_dist[1]:.1%}")
    print(f"   Après: Classe 0={clean_dist[0]:.1%}, Classe 1={clean_dist[1]:.1%}")
    
    # Backup et sauvegarde
    print(f"\n💾 Sauvegarde...")
    backup_path = train_path.replace('.csv', '_with_leaks.csv')
    train_df.to_csv(backup_path, index=False)
    print(f"   Backup: {backup_path}")
    
    train_df_clean.to_csv(train_path, index=False)
    print(f"   Train nettoyé: {train_path}")
    
    # Vérification finale
    print(f"\n🔍 Vérification finale...")
    final_train_texts = set(train_df_clean['text_cleaned'].str.strip().str.lower())
    final_common = final_train_texts.intersection(test_texts)
    
    if len(final_common) == 0:
        print("   ✅ SUCCÈS: Toutes les fuites éliminées!")
        success = True
    else:
        print(f"   ❌ {len(final_common)} fuites persistent")
        success = False
    
    print(f"\n📋 RÉSUMÉ:")
    print(f"🗑️  Supprimés: {removed}")
    print(f"📊 Train final: {len(train_df_clean)}")
    print(f"📊 Test: {len(test_df)} (inchangé)")
    print(f"✨ Intégrité: {'PARFAITE' if success else 'PROBLÉMATIQUE'}")
    
    return success

if __name__ == "__main__":
    success = clean_final_leakage()
    if success:
        print("\n🎉 Nettoyage final terminé!")
    else:
        print("\n❌ Problème persistant")
