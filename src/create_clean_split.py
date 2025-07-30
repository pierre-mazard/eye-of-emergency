"""
Script de partitionnement stratifié propre
1. Charge le dataset nettoyé
2. Effectue un partitionnement stratifié 90/10
3. Vérifie l'absence de fuites
4. Sauvegarde les partitions
"""

import pandas as pd
import numpy as np
import os

def stratified_split(df, target_col, test_size=0.1, random_state=42):
    """
    Partitionnement stratifié manuel
    """
    np.random.seed(random_state)
    
    # Séparer par classe
    class_0 = df[df[target_col] == 0].copy()
    class_1 = df[df[target_col] == 1].copy()
    
    # Mélanger chaque classe
    class_0 = class_0.sample(frac=1, random_state=random_state).reset_index(drop=True)
    class_1 = class_1.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculer les tailles pour le test
    test_size_0 = int(len(class_0) * test_size)
    test_size_1 = int(len(class_1) * test_size)
    
    # Diviser chaque classe
    test_0 = class_0[:test_size_0]
    train_0 = class_0[test_size_0:]
    
    test_1 = class_1[:test_size_1]
    train_1 = class_1[test_size_1:]
    
    # Recombiner
    train_df = pd.concat([train_0, train_1], ignore_index=True)
    test_df = pd.concat([test_0, test_1], ignore_index=True)
    
    # Mélanger les datasets finaux
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_df, test_df

def create_clean_split():
    """
    Crée un partitionnement propre du dataset nettoyé
    """
    print("🔄 PARTITIONNEMENT STRATIFIÉ PROPRE")
    print("=" * 50)
    
    # Chargement du dataset nettoyé
    cleaned_path = 'data/raw/original_train_tweets_cleaned.csv'
    print(f"📁 Chargement de {cleaned_path}")
    
    try:
        df_cleaned = pd.read_csv(cleaned_path)
        print(f"✅ Dataset chargé: {len(df_cleaned)} échantillons")
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier {cleaned_path} non trouvé")
        print("➡️  Exécutez d'abord clean_original_dataset.py")
        return False
    
    # Vérification de l'intégrité
    print(f"\n🔍 Vérification de l'intégrité du dataset:")
    print(f"   Valeurs manquantes dans 'text': {df_cleaned['text'].isnull().sum()}")
    print(f"   Valeurs manquantes dans 'target': {df_cleaned['target'].isnull().sum()}")
    print(f"   IDs uniques: {df_cleaned['id'].nunique()}")
    print(f"   Textes uniques: {df_cleaned['text'].nunique()}")
    
    # Distribution des classes
    print(f"\n📊 Distribution des classes:")
    class_dist = df_cleaned['target'].value_counts().sort_index()
    for class_label, count in class_dist.items():
        pct = (count / len(df_cleaned)) * 100
        class_name = "Non-catastrophe" if class_label == 0 else "Catastrophe"
        print(f"   Classe {class_label} ({class_name}): {count:,} ({pct:.1f}%)")
    
    # Partitionnement stratifié
    print(f"\n⚖️ Partitionnement stratifié (90% train, 10% test):")
    
    # Utilisation de notre fonction de partitionnement stratifié
    train_df, test_df = stratified_split(df_cleaned, 'target', test_size=0.1, random_state=42)
    
    print(f"   Train: {len(train_df)} échantillons ({len(train_df)/len(df_cleaned)*100:.1f}%)")
    print(f"   Test: {len(test_df)} échantillons ({len(test_df)/len(df_cleaned)*100:.1f}%)")
    
    # Vérification de la stratification
    print(f"\n📈 Vérification de la stratification:")
    
    train_dist = train_df['target'].value_counts(normalize=True).sort_index()
    test_dist = test_df['target'].value_counts(normalize=True).sort_index()
    original_dist = df_cleaned['target'].value_counts(normalize=True).sort_index()
    
    print(f"   Distribution originale:")
    for class_label, pct in original_dist.items():
        class_name = "Non-catastrophe" if class_label == 0 else "Catastrophe"
        print(f"     Classe {class_label} ({class_name}): {pct:.1%}")
    
    print(f"   Distribution train:")
    for class_label, pct in train_dist.items():
        class_name = "Non-catastrophe" if class_label == 0 else "Catastrophe"
        original_pct = original_dist[class_label]
        diff = abs(pct - original_pct)
        print(f"     Classe {class_label} ({class_name}): {pct:.1%} (Δ{diff:.2%})")
    
    print(f"   Distribution test:")
    for class_label, pct in test_dist.items():
        class_name = "Non-catastrophe" if class_label == 0 else "Catastrophe"
        original_pct = original_dist[class_label]
        diff = abs(pct - original_pct)
        print(f"     Classe {class_label} ({class_name}): {pct:.1%} (Δ{diff:.2%})")
    
    # Calcul de l'écart maximum
    max_train_diff = abs(train_dist - original_dist).max()
    max_test_diff = abs(test_dist - original_dist).max()
    
    if max_train_diff <= 0.01 and max_test_diff <= 0.01:
        print(f"   ✅ Stratification excellente (écarts < 1%)")
    elif max_train_diff <= 0.02 and max_test_diff <= 0.02:
        print(f"   ✅ Stratification très bonne (écarts < 2%)")
    else:
        print(f"   ⚠️  Stratification acceptable (écarts max: {max(max_train_diff, max_test_diff):.1%})")
    
    # VÉRIFICATION CRITIQUE: Absence de fuites par construction
    print(f"\n🔒 Vérification de l'absence de fuites:")
    
    # IDs
    train_ids = set(train_df['id'])
    test_ids = set(test_df['id'])
    common_ids = train_ids.intersection(test_ids)
    print(f"   IDs en commun: {len(common_ids)}")
    
    # Textes
    train_texts = set(train_df['text'].str.strip().str.lower())
    test_texts = set(test_df['text'].str.strip().str.lower())
    common_texts = train_texts.intersection(test_texts)
    print(f"   Textes en commun: {len(common_texts)}")
    
    if len(common_ids) == 0 and len(common_texts) == 0:
        print(f"   ✅ PARFAIT: Aucune fuite détectée par construction!")
        leak_status = "AUCUNE"
    else:
        print(f"   ❌ PROBLÈME: Fuites détectées malgré le partitionnement!")
        leak_status = "DÉTECTÉE"
    
    # Sauvegarde des nouvelles partitions
    print(f"\n💾 Sauvegarde des nouvelles partitions:")
    
    # Backup des anciennes partitions si elles existent
    old_train_path = 'data/raw/train_tweets.csv'
    old_test_path = 'data/raw/test_tweets.csv'
    
    if os.path.exists(old_train_path):
        backup_train = 'data/raw/train_tweets_old.csv'
        os.rename(old_train_path, backup_train)
        print(f"   Backup ancien train: {backup_train}")
    
    if os.path.exists(old_test_path):
        backup_test = 'data/raw/test_tweets_old.csv'
        os.rename(old_test_path, backup_test)
        print(f"   Backup ancien test: {backup_test}")
    
    # Sauvegarde des nouvelles partitions
    train_df.to_csv(old_train_path, index=False)
    test_df.to_csv(old_test_path, index=False)
    
    print(f"   ✅ Nouveau train: {old_train_path}")
    print(f"   ✅ Nouveau test: {old_test_path}")
    
    # Rapport final
    print(f"\n" + "=" * 50)
    print(f"📋 RAPPORT DE PARTITIONNEMENT")
    print(f"=" * 50)
    print(f"📊 Dataset source: {len(df_cleaned)} échantillons")
    print(f"🚂 Partition train: {len(train_df)} échantillons ({len(train_df)/len(df_cleaned)*100:.1f}%)")
    print(f"🧪 Partition test: {len(test_df)} échantillons ({len(test_df)/len(df_cleaned)*100:.1f}%)")
    print(f"⚖️  Écart de stratification max: {max(max_train_diff, max_test_diff):.2%}")
    print(f"🔒 Fuites de données: {leak_status}")
    print(f"✅ Qualité: {'EXCELLENTE' if leak_status == 'AUCUNE' else 'PROBLÉMATIQUE'}")
    print(f"=" * 50)
    
    return True

if __name__ == "__main__":
    success = create_clean_split()
    if success:
        print(f"\n🎉 Partitionnement terminé avec succès!")
        print(f"➡️  Prêt pour le preprocessing des nouvelles partitions")
    else:
        print(f"\n❌ Erreur lors du partitionnement")
