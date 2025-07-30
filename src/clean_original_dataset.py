"""
Script de nettoyage du dataset original
1. Analyse des doublons dans original_train_tweets.csv
2. Suppression des doublons exacts
3. Sauvegarde du fichier nettoyé
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

def analyze_and_clean_original_dataset():
    """
    Analyse et nettoie le dataset original des doublons
    """
    print("🔍 ANALYSE ET NETTOYAGE DU DATASET ORIGINAL")
    print("=" * 60)
    
    # Chargement du dataset original
    original_path = 'data/raw/original_train_tweets.csv'
    print(f"📁 Chargement de {original_path}")
    
    try:
        df_original = pd.read_csv(original_path)
        print(f"✅ Dataset chargé: {len(df_original)} échantillons")
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier {original_path} non trouvé")
        return False
    
    print(f"📊 Colonnes: {list(df_original.columns)}")
    print(f"📊 Shape: {df_original.shape}")
    
    # Analyse des valeurs manquantes
    print(f"\n📋 Analyse des valeurs manquantes:")
    missing_analysis = df_original.isnull().sum()
    for col, missing_count in missing_analysis.items():
        if missing_count > 0:
            pct = (missing_count / len(df_original)) * 100
            print(f"   {col}: {missing_count} ({pct:.1f}%)")
        else:
            print(f"   {col}: aucune valeur manquante")
    
    # Analyse de la distribution des classes AVANT nettoyage
    print(f"\n📈 Distribution des classes AVANT nettoyage:")
    class_dist_before = df_original['target'].value_counts().sort_index()
    for class_label, count in class_dist_before.items():
        pct = (count / len(df_original)) * 100
        class_name = "Non-catastrophe" if class_label == 0 else "Catastrophe"
        print(f"   Classe {class_label} ({class_name}): {count:,} ({pct:.1f}%)")
    
    # 1. ANALYSE DES DOUBLONS EXACTS (toutes colonnes)
    print(f"\n🔍 ANALYSE DES DOUBLONS COMPLETS...")
    duplicates_all = df_original.duplicated()
    duplicate_count_all = duplicates_all.sum()
    print(f"   Doublons complets (toutes colonnes): {duplicate_count_all}")
    
    # 2. ANALYSE DES DOUBLONS DE TEXTE
    print(f"\n🔍 ANALYSE DES DOUBLONS DE TEXTE...")
    
    # Doublons exacts de texte
    text_duplicates_exact = df_original['text'].duplicated()
    text_duplicate_count_exact = text_duplicates_exact.sum()
    print(f"   Doublons de texte exacts: {text_duplicate_count_exact}")
    
    # Doublons de texte (case-insensitive et stripped)
    df_original['text_normalized'] = df_original['text'].str.strip().str.lower()
    text_duplicates_normalized = df_original['text_normalized'].duplicated()
    text_duplicate_count_normalized = text_duplicates_normalized.sum()
    print(f"   Doublons de texte normalisés: {text_duplicate_count_normalized}")
    
    # Analyse des groupes de doublons
    if text_duplicate_count_normalized > 0:
        print(f"\n📊 ANALYSE DES GROUPES DE DOUBLONS:")
        duplicate_groups = df_original[df_original['text_normalized'].duplicated(keep=False)]
        group_sizes = duplicate_groups.groupby('text_normalized').size()
        
        print(f"   Nombre de groupes de doublons: {len(group_sizes)}")
        print(f"   Taille moyenne des groupes: {group_sizes.mean():.1f}")
        print(f"   Taille max des groupes: {group_sizes.max()}")
        
        # Exemples de doublons
        print(f"\n📝 Exemples de textes dupliqués:")
        largest_groups = group_sizes.nlargest(3)
        for i, (text_norm, size) in enumerate(largest_groups.items(), 1):
            original_texts = df_original[df_original['text_normalized'] == text_norm]['text'].unique()
            preview = original_texts[0][:80] + "..." if len(original_texts[0]) > 80 else original_texts[0]
            print(f"   {i}. Groupe de {size} doublons: '{preview}'")
    
    # 3. ANALYSE DE L'IMPACT SUR LES CLASSES
    if text_duplicate_count_normalized > 0:
        print(f"\n⚖️ IMPACT DES DOUBLONS SUR LES CLASSES:")
        
        # Vérifier si les doublons ont le même label
        duplicate_mask = df_original['text_normalized'].duplicated(keep=False)
        duplicate_data = df_original[duplicate_mask]
        
        # Grouper par texte et vérifier la cohérence des labels
        label_consistency = duplicate_data.groupby('text_normalized')['target'].nunique()
        inconsistent_labels = (label_consistency > 1).sum()
        
        if inconsistent_labels > 0:
            print(f"   ⚠️  ATTENTION: {inconsistent_labels} groupes ont des labels incohérents!")
            print(f"   📋 Exemples de labels incohérents:")
            inconsistent_groups = label_consistency[label_consistency > 1].head(3)
            for text_norm in inconsistent_groups.index:
                group_data = duplicate_data[duplicate_data['text_normalized'] == text_norm]
                labels = group_data['target'].unique()
                preview = group_data['text'].iloc[0][:60] + "..."
                print(f"      '{preview}' -> Labels: {labels}")
        else:
            print(f"   ✅ Tous les doublons ont des labels cohérents")
    
    # 4. NETTOYAGE DES DOUBLONS
    print(f"\n🧹 NETTOYAGE DES DOUBLONS...")
    
    # Stratégie: Garder le premier exemplaire de chaque texte normalisé
    df_cleaned = df_original.drop_duplicates(subset=['text_normalized'], keep='first')
    removed_count = len(df_original) - len(df_cleaned)
    
    print(f"   Échantillons supprimés: {removed_count}")
    print(f"   Dataset nettoyé: {len(df_cleaned)} échantillons")
    
    # Suppression de la colonne temporaire
    df_cleaned = df_cleaned.drop('text_normalized', axis=1)
    
    # Vérification de la distribution des classes APRÈS nettoyage
    print(f"\n📈 Distribution des classes APRÈS nettoyage:")
    class_dist_after = df_cleaned['target'].value_counts().sort_index()
    for class_label, count in class_dist_after.items():
        pct = (count / len(df_cleaned)) * 100
        class_name = "Non-catastrophe" if class_label == 0 else "Catastrophe"
        before_count = class_dist_before[class_label]
        change = count - before_count
        print(f"   Classe {class_label} ({class_name}): {count:,} ({pct:.1f}%) [Δ{change:+d}]")
    
    # Calcul des écarts de distribution
    pct_before = class_dist_before / len(df_original) * 100
    pct_after = class_dist_after / len(df_cleaned) * 100
    max_change = abs(pct_after - pct_before).max()
    
    if max_change <= 1.0:
        print(f"   ✅ Distribution bien préservée (écart max: {max_change:.2f}%)")
    elif max_change <= 3.0:
        print(f"   ⚠️  Distribution légèrement modifiée (écart max: {max_change:.2f}%)")
    else:
        print(f"   ❌ Distribution significativement modifiée (écart max: {max_change:.2f}%)")
    
    # 5. SAUVEGARDE
    print(f"\n💾 SAUVEGARDE...")
    
    # Backup de l'original
    backup_path = 'data/raw/original_train_tweets_with_duplicates.csv'
    if not os.path.exists(backup_path):
        df_original.drop('text_normalized', axis=1).to_csv(backup_path, index=False)
        print(f"   Backup de l'original: {backup_path}")
    
    # Sauvegarde du fichier nettoyé
    cleaned_path = 'data/raw/original_train_tweets_cleaned.csv'
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"   Dataset nettoyé: {cleaned_path}")
    
    # 6. RAPPORT FINAL
    print(f"\n" + "=" * 60)
    print(f"📋 RAPPORT DE NETTOYAGE")
    print(f"=" * 60)
    print(f"📊 Dataset original: {len(df_original)} échantillons")
    print(f"🗑️  Doublons supprimés: {removed_count}")
    print(f"✨ Dataset nettoyé: {len(df_cleaned)} échantillons")
    print(f"📉 Réduction: {(removed_count/len(df_original)*100):.1f}%")
    print(f"⚖️  Écart de distribution max: {max_change:.2f}%")
    print(f"✅ Fichier de sortie: {cleaned_path}")
    print(f"=" * 60)
    
    return True, cleaned_path, len(df_cleaned)

if __name__ == "__main__":
    success, output_path, final_count = analyze_and_clean_original_dataset()
    if success:
        print(f"\n🎉 Nettoyage terminé avec succès!")
        print(f"➡️  Fichier nettoyé prêt pour le partitionnement: {output_path}")
        print(f"📊 {final_count} échantillons uniques disponibles")
    else:
        print(f"\n❌ Erreur lors du nettoyage")
