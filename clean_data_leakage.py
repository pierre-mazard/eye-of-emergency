"""
Script de nettoyage des fuites de données
Supprime les échantillons en commun entre train et test
"""

import pandas as pd
import os

def clean_data_leakage():
    """
    Supprime les fuites de données entre train et test
    """
    print("🧹 NETTOYAGE DES FUITES DE DONNÉES")
    print("=" * 50)
    
    # Chargement des datasets
    print("📁 Chargement des datasets...")
    train_path = 'data/processed/train_optimized_v3.csv'
    test_path = 'data/processed/test_cleaned.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Train initial: {len(train_df)} échantillons")
    print(f"   Test initial: {len(test_df)} échantillons")
    
    # Identification des textes en commun
    print("\n🔍 Identification des fuites...")
    train_texts = train_df['text_cleaned'].str.strip().str.lower()
    test_texts = test_df['text_cleaned'].str.strip().str.lower()
    
    # Trouver les textes en commun
    common_texts = set(train_texts).intersection(set(test_texts))
    print(f"   Textes en commun trouvés: {len(common_texts)}")
    
    if len(common_texts) == 0:
        print("✅ Aucune fuite détectée - Aucun nettoyage nécessaire")
        return True
    
    # Supprimer les fuites du dataset d'entraînement (stratégie conservatrice)
    print(f"\n🗑️  Suppression des fuites du dataset d'entraînement...")
    mask_train_clean = ~train_texts.isin(common_texts)
    train_df_clean = train_df[mask_train_clean].copy()
    
    removed_count = len(train_df) - len(train_df_clean)
    print(f"   Échantillons supprimés du train: {removed_count}")
    print(f"   Train après nettoyage: {len(train_df_clean)} échantillons")
    
    # Vérification de la distribution des classes après nettoyage
    print(f"\n📊 Vérification de la distribution des classes:")
    
    original_dist = train_df['target'].value_counts(normalize=True)
    clean_dist = train_df_clean['target'].value_counts(normalize=True)
    
    print(f"   Distribution originale:")
    print(f"     Classe 0: {original_dist[0]:.1%}")
    print(f"     Classe 1: {original_dist[1]:.1%}")
    
    print(f"   Distribution après nettoyage:")
    print(f"     Classe 0: {clean_dist[0]:.1%}")
    print(f"     Classe 1: {clean_dist[1]:.1%}")
    
    # Calcul de l'écart
    class_0_change = abs(clean_dist[0] - original_dist[0])
    class_1_change = abs(clean_dist[1] - original_dist[1])
    
    if max(class_0_change, class_1_change) <= 0.02:  # 2% de tolérance
        print("   ✅ Distribution des classes préservée")
    else:
        print("   ⚠️  Distribution des classes légèrement modifiée")
    
    # Sauvegarde des datasets nettoyés
    print(f"\n💾 Sauvegarde des datasets nettoyés...")
    
    # Backup des anciens fichiers
    backup_train = train_path.replace('.csv', '_with_leakage.csv')
    train_df.to_csv(backup_train, index=False)
    print(f"   Backup train: {backup_train}")
    
    # Sauvegarde du nouveau dataset d'entraînement
    train_df_clean.to_csv(train_path, index=False)
    print(f"   Train nettoyé: {train_path}")
    
    # Le test reste inchangé (pas de fuites d'IDs, juste de contenus)
    print(f"   Test inchangé: {test_path}")
    
    # Vérification finale
    print(f"\n🔍 Vérification finale...")
    train_texts_final = set(train_df_clean['text_cleaned'].str.strip().str.lower())
    test_texts_final = set(test_df['text_cleaned'].str.strip().str.lower())
    final_leaks = train_texts_final.intersection(test_texts_final)
    
    if len(final_leaks) == 0:
        print("   ✅ SUCCÈS: Toutes les fuites ont été éliminées!")
        success = True
    else:
        print(f"   ❌ ÉCHEC: {len(final_leaks)} fuites persistent")
        success = False
    
    # Résumé final
    print(f"\n" + "=" * 50)
    print(f"📋 RÉSUMÉ DU NETTOYAGE")
    print(f"=" * 50)
    print(f"🗑️  Échantillons supprimés: {removed_count}")
    print(f"📊 Train final: {len(train_df_clean)} échantillons")
    print(f"📊 Test final: {len(test_df)} échantillons (inchangé)")
    print(f"🎯 Fuites éliminées: {len(common_texts) - len(final_leaks)}")
    print(f"✨ Intégrité: {'PARFAITE' if success else 'PROBLÉMATIQUE'}")
    
    return success

if __name__ == "__main__":
    success = clean_data_leakage()
    if success:
        print("\n🎉 Nettoyage terminé avec succès!")
        print("➡️  Vous pouvez maintenant relancer la validation")
    else:
        print("\n❌ Problème lors du nettoyage")
        print("➡️  Vérifiez manuellement les données")
