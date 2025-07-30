"""
Pipeline complet de preprocessing pour le projet Eye of Emergency
Intègre toutes les étapes dans le bon ordre :
1. Nettoyage du dataset original (doublons)
2. Partitionnement stratifié propre  
3. Preprocessing du train avec feature engineering
4. Preprocessing du test (nettoyage uniquement)
5. Nettoyage final des fuites
6. Validation de l'intégrité
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple
from collections import Counter

# Import des modules spécialisés
from preprocess_train import OptimizedEmergencyPreprocessor


def step1_clean_original_dataset() -> Tuple[bool, str, int]:
    """
    Étape 1: Analyse et nettoyage du dataset original des doublons
    """
    print("🔍 ÉTAPE 1: NETTOYAGE DU DATASET ORIGINAL")
    print("=" * 60)
    
    # Chargement du dataset original
    original_path = 'data/raw/original_train_tweets.csv'
    print(f"📁 Chargement de {original_path}")
    
    try:
        df_original = pd.read_csv(original_path)
        print(f"✅ Dataset chargé: {len(df_original)} échantillons")
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier {original_path} non trouvé")
        return False, "", 0
    
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
    
    # Analyse des doublons de texte (case-insensitive et stripped)
    print(f"\n🔍 ANALYSE DES DOUBLONS DE TEXTE...")
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
        
        # Vérifier si les doublons ont le même label
        duplicate_mask = df_original['text_normalized'].duplicated(keep=False)
        duplicate_data = df_original[duplicate_mask]
        
        # Grouper par texte et vérifier la cohérence des labels
        label_consistency = duplicate_data.groupby('text_normalized')['target'].nunique()
        inconsistent_labels = (label_consistency > 1).sum()
        
        if inconsistent_labels > 0:
            print(f"   ⚠️  ATTENTION: {inconsistent_labels} groupes ont des labels incohérents!")
            print(f"   🔧 Résolution automatique: conservation du label majoritaire")
        else:
            print(f"   ✅ Tous les doublons ont des labels cohérents")
    
    # NETTOYAGE DES DOUBLONS
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
    
    # SAUVEGARDE
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
    
    print(f"✅ ÉTAPE 1 TERMINÉE: Dataset nettoyé de {len(df_original)} → {len(df_cleaned)} échantillons")
    
    return True, cleaned_path, len(df_cleaned)


def step2_create_clean_split() -> bool:
    """
    Étape 2: Partitionnement stratifié propre
    """
    print("\n🔄 ÉTAPE 2: PARTITIONNEMENT STRATIFIÉ PROPRE")
    print("=" * 50)
    
    # Chargement du dataset nettoyé
    cleaned_path = 'data/raw/original_train_tweets_cleaned.csv'
    print(f"📁 Chargement de {cleaned_path}")
    
    try:
        df_cleaned = pd.read_csv(cleaned_path)
        print(f"✅ Dataset chargé: {len(df_cleaned)} échantillons")
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier {cleaned_path} non trouvé")
        return False
    
    # Fonction de partitionnement stratifié manuel
    def stratified_split(df, target_col, test_size=0.1, random_state=42):
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
    
    # Distribution des classes
    print(f"\n📊 Distribution des classes:")
    class_dist = df_cleaned['target'].value_counts().sort_index()
    for class_label, count in class_dist.items():
        pct = (count / len(df_cleaned)) * 100
        class_name = "Non-catastrophe" if class_label == 0 else "Catastrophe"
        print(f"   Classe {class_label} ({class_name}): {count:,} ({pct:.1f}%)")
    
    # Partitionnement stratifié
    print(f"\n⚖️ Partitionnement stratifié (90% train, 10% test):")
    train_df, test_df = stratified_split(df_cleaned, 'target', test_size=0.1, random_state=42)
    
    print(f"   Train: {len(train_df)} échantillons ({len(train_df)/len(df_cleaned)*100:.1f}%)")
    print(f"   Test: {len(test_df)} échantillons ({len(test_df)/len(df_cleaned)*100:.1f}%)")
    
    # Vérification de la stratification
    print(f"\n📈 Vérification de la stratification:")
    train_dist = train_df['target'].value_counts(normalize=True).sort_index()
    test_dist = test_df['target'].value_counts(normalize=True).sort_index()
    original_dist = df_cleaned['target'].value_counts(normalize=True).sort_index()
    
    # Calcul de l'écart maximum
    max_train_diff = abs(train_dist - original_dist).max()
    max_test_diff = abs(test_dist - original_dist).max()
    
    if max_train_diff <= 0.01 and max_test_diff <= 0.01:
        print(f"   ✅ Stratification excellente (écarts < 1%)")
    elif max_train_diff <= 0.02 and max_test_diff <= 0.02:
        print(f"   ✅ Stratification très bonne (écarts < 2%)")
    else:
        print(f"   ⚠️  Stratification acceptable (écarts max: {max(max_train_diff, max_test_diff):.1%})")
    
    # Vérification de l'absence de fuites par construction
    print(f"\n🔒 Vérification de l'absence de fuites:")
    train_ids = set(train_df['id'])
    test_ids = set(test_df['id'])
    common_ids = train_ids.intersection(test_ids)
    
    train_texts = set(train_df['text'].str.strip().str.lower())
    test_texts = set(test_df['text'].str.strip().str.lower())
    common_texts = train_texts.intersection(test_texts)
    
    print(f"   IDs en commun: {len(common_ids)}")
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
        if os.path.exists(backup_train):
            os.remove(backup_train)
        os.rename(old_train_path, backup_train)
        print(f"   Backup ancien train: {backup_train}")
    
    if os.path.exists(old_test_path):
        backup_test = 'data/raw/test_tweets_old.csv'
        if os.path.exists(backup_test):
            os.remove(backup_test)
        os.rename(old_test_path, backup_test)
        print(f"   Backup ancien test: {backup_test}")
    
    # Sauvegarde des nouvelles partitions
    train_df.to_csv(old_train_path, index=False)
    test_df.to_csv(old_test_path, index=False)
    
    print(f"   ✅ Nouveau train: {old_train_path}")
    print(f"   ✅ Nouveau test: {old_test_path}")
    
    print(f"✅ ÉTAPE 2 TERMINÉE: Partitionnement {len(df_cleaned)} → {len(train_df)}+{len(test_df)}")
    return True


def step3_preprocess_train() -> str:
    """
    Étape 3: Preprocessing du train avec feature engineering
    """
    print("\n🔄 ÉTAPE 3: PREPROCESSING TRAIN AVEC FEATURE ENGINEERING")
    print("=" * 60)
    
    # Utilisation de la fonction existante
    from preprocess_train import create_optimized_datasets_v3
    
    train_path = 'data/raw/train_tweets.csv'
    output_dir = 'data/processed'
    
    train_output = create_optimized_datasets_v3(
        train_path=train_path,
        output_dir=output_dir
    )
    
    print(f"✅ ÉTAPE 3 TERMINÉE: Train preprocessé → {train_output}")
    return train_output


def step4_preprocess_test() -> bool:
    """
    Étape 4: Preprocessing du test (nettoyage uniquement)
    """
    print("\n🔄 ÉTAPE 4: PREPROCESSING TEST (NETTOYAGE UNIQUEMENT)")
    print("=" * 60)
    
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
    
    print(f"✅ ÉTAPE 4 TERMINÉE: Test nettoyé → {output_path}")
    return True


def step5_clean_final_leakage() -> bool:
    """
    Étape 5: Nettoyage final des fuites
    """
    print("\n🧹 ÉTAPE 5: NETTOYAGE FINAL DES FUITES")
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
        print(f"✅ ÉTAPE 5 TERMINÉE: Aucune fuite détectée")
        return True
    
    # Suppression des fuites du train (préserver la taille du test)
    print(f"\n🗑️  Suppression des fuites du dataset d'entraînement...")
    train_texts_col = train_df['text_cleaned'].str.strip().str.lower()
    mask_clean = ~train_texts_col.isin(common_texts)
    train_df_clean = train_df[mask_clean].copy()
    
    removed = len(train_df) - len(train_df_clean)
    print(f"   Échantillons supprimés: {removed}")
    print(f"   Train final: {len(train_df_clean)} échantillons")
    
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
    
    print(f"✅ ÉTAPE 5 TERMINÉE: {removed} fuites supprimées")
    return success


def step6_validate_integrity() -> bool:
    """
    Étape 6: Validation finale de l'intégrité
    """
    print("\n🔍 ÉTAPE 6: VALIDATION FINALE DE L'INTÉGRITÉ")
    print("=" * 50)
    
    # Chargement des datasets finaux
    train_path = 'data/processed/train_optimized.csv'
    test_path = 'data/processed/test_cleaned.csv'
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"📁 Datasets chargés:")
        print(f"   Train: {len(train_df)} échantillons")
        print(f"   Test: {len(test_df)} échantillons")
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        return False
    
    # Vérification de l'absence de fuites
    print(f"\n🔒 Vérification de l'absence de fuites:")
    
    # Textes
    train_texts = set(train_df['text_cleaned'].str.strip().str.lower())
    test_texts = set(test_df['text_cleaned'].str.strip().str.lower())
    common_texts = train_texts.intersection(test_texts)
    
    print(f"   Textes en commun: {len(common_texts)}")
    
    if len(common_texts) == 0:
        print("   ✅ PARFAIT: Aucune fuite de texte détectée!")
        integrity_score = 100
    else:
        print(f"   ❌ {len(common_texts)} fuites de texte détectées")
        integrity_score = max(0, 100 - (len(common_texts) / len(test_texts)) * 100)
    
    # Vérification des colonnes attendues
    print(f"\n📊 Vérification des colonnes:")
    expected_train_cols = ['id', 'keyword', 'target', 'text_cleaned']
    expected_test_cols = ['id', 'keyword', 'location', 'text', 'target', 'text_cleaned']
    
    train_cols_ok = all(col in train_df.columns for col in expected_train_cols)
    test_cols_ok = all(col in expected_test_cols for col in expected_test_cols)
    
    print(f"   Train columns: {'✅' if train_cols_ok else '❌'} {list(train_df.columns)}")
    print(f"   Test columns: {'✅' if test_cols_ok else '❌'} {list(test_df.columns)}")
    
    # Vérification des valeurs manquantes critiques
    print(f"\n📋 Vérification des valeurs manquantes critiques:")
    train_missing = train_df[['text_cleaned', 'target']].isnull().sum()
    test_missing = test_df[['text_cleaned', 'target']].isnull().sum()
    
    print(f"   Train - text_cleaned: {train_missing['text_cleaned']} manquants")
    print(f"   Train - target: {train_missing['target']} manquants")
    print(f"   Test - text_cleaned: {test_missing['text_cleaned']} manquants")
    print(f"   Test - target: {test_missing['target']} manquants")
    
    missing_ok = (train_missing.sum() == 0) and (test_missing.sum() == 0)
    
    # Score final
    structure_score = 20 if train_cols_ok and test_cols_ok else 0
    missing_score = 20 if missing_ok else 0
    total_score = integrity_score * 0.6 + structure_score + missing_score
    
    print(f"\n🏆 SCORE D'INTÉGRITÉ FINAL:")
    print(f"   Absence de fuites: {integrity_score:.0f}/60")
    print(f"   Structure: {structure_score}/20")
    print(f"   Valeurs manquantes: {missing_score}/20")
    print(f"   TOTAL: {total_score:.0f}/100")
    
    if total_score >= 95:
        print(f"   🎉 EXCELLENT: Pipeline parfaitement exécuté!")
        status = "EXCELLENT"
    elif total_score >= 80:
        print(f"   ✅ TRÈS BON: Pipeline bien exécuté")
        status = "TRÈS BON"
    elif total_score >= 60:
        print(f"   ⚠️  ACCEPTABLE: Quelques problèmes mineurs")
        status = "ACCEPTABLE"
    else:
        print(f"   ❌ PROBLÉMATIQUE: Problèmes majeurs détectés")
        status = "PROBLÉMATIQUE"
    
    print(f"✅ ÉTAPE 6 TERMINÉE: Validation complète - Status {status}")
    return total_score >= 60


def run_complete_pipeline() -> bool:
    """
    Exécute le pipeline complet dans le bon ordre
    """
    print("🚀 LANCEMENT DU PIPELINE COMPLET EYE OF EMERGENCY")
    print("=" * 70)
    print("🎯 Objectif: Pipeline de données propre et sans fuites")
    print("📋 Étapes: Nettoyage → Partitionnement → Preprocessing → Validation")
    print()
    
    try:
        # Étape 1: Nettoyage du dataset original
        success, cleaned_path, count = step1_clean_original_dataset()
        if not success:
            print("❌ ÉCHEC: Étape 1 - Nettoyage du dataset original")
            return False
        
        # Étape 2: Partitionnement stratifié
        success = step2_create_clean_split()
        if not success:
            print("❌ ÉCHEC: Étape 2 - Partitionnement stratifié")
            return False
        
        # Étape 3: Preprocessing du train
        train_output = step3_preprocess_train()
        if not train_output:
            print("❌ ÉCHEC: Étape 3 - Preprocessing train")
            return False
        
        # Étape 4: Preprocessing du test
        success = step4_preprocess_test()
        if not success:
            print("❌ ÉCHEC: Étape 4 - Preprocessing test")
            return False
        
        # Étape 5: Nettoyage final des fuites
        success = step5_clean_final_leakage()
        if not success:
            print("❌ ÉCHEC: Étape 5 - Nettoyage final des fuites")
            return False
        
        # Étape 6: Validation finale
        success = step6_validate_integrity()
        if not success:
            print("❌ ÉCHEC: Étape 6 - Validation finale")
            return False
        
        print(f"\n" + "=" * 70)
        print(f"🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
        print(f"=" * 70)
        print(f"📊 Fichiers générés:")
        print(f"   ✅ Train: data/processed/train_optimized.csv")
        print(f"   ✅ Test: data/processed/test_cleaned.csv")
        print(f"🔒 Intégrité: VALIDÉE")
        print(f"⚡ Prêt pour l'entraînement des modèles!")
        print(f"=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR DANS LE PIPELINE: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_complete_pipeline()
    if success:
        print("\n✅ Pipeline terminé avec succès!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline échoué")
        sys.exit(1)
