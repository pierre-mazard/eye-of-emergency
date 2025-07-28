"""
Preprocessing pipeline pour le projet Eye of Emergency
Nettoyage et préparation des données textuelles pour la classification de tweets de catastrophes
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Tuple, List, Dict, Any
from collections import Counter
import unicodedata
import warnings
warnings.filterwarnings('ignore')

class EmergencyTweetPreprocessor:
    """
    Classe principale pour le preprocessing des tweets de catastrophes
    """
    
    def __init__(self, remove_location: bool = True, handle_duplicates: bool = True):
        """
        Initialise le preprocessor
        
        Args:
            remove_location: Si True, supprime la colonne location (recommandé)
            handle_duplicates: Si True, gère les doublons et conflits
        """
        self.remove_location = remove_location
        self.should_handle_duplicates = handle_duplicates
        self.stop_words = self._get_stop_words()
        self.emergency_keywords = self._get_emergency_keywords()
        
    def _get_stop_words(self) -> set:
        """Retourne les stop words en gardant ceux importants pour les urgences"""
        basic_stops = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'you', 'your', 'i', 'me',
            'my', 'we', 'us', 'our', 'they', 'them', 'their', 'this', 'these',
            'those', 'am', 'been', 'being', 'have', 'had', 'do', 'does', 'did',
            'can', 'could', 'should', 'may', 'might', 'must', 'shall', 'but',
            'or', 'if', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        # Ne pas supprimer les mots d'urgence même s'ils ressemblent à des stop words
        return basic_stops
    
    def _get_emergency_keywords(self) -> set:
        """Retourne les mots-clés d'urgence à préserver"""
        return {
            'emergency', 'urgent', 'help', 'fire', 'flood', 'earthquake', 'storm',
            'evacuate', 'rescue', 'disaster', 'crisis', 'danger', 'warning',
            'alert', 'breaking', 'urgent', 'police', 'ambulance', 'hospital',
            'injured', 'casualties', 'victims', 'trapped', 'missing', 'found',
            'safe', 'unsafe', 'damage', 'destroyed', 'collapsed', 'explosion'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte selon les patterns identifiés dans l'analyse
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Conversion en string et nettoyage unicode
        text = str(text)
        text = unicodedata.normalize('NFKD', text)
        
        # Préservation des patterns importants avant nettoyage
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        has_url = len(urls) > 0
        
        # Remplacement des URLs par un token spécial
        text = re.sub(url_pattern, ' URL_TOKEN ', text)
        
        # Gestion des mentions (@username)
        mention_pattern = r'@[A-Za-z0-9_]+'
        mentions = re.findall(mention_pattern, text)
        has_mention = len(mentions) > 0
        text = re.sub(mention_pattern, ' MENTION_TOKEN ', text)
        
        # Gestion des hashtags (#hashtag)
        hashtag_pattern = r'#[A-Za-z0-9_]+'
        hashtags = re.findall(hashtag_pattern, text)
        has_hashtag = len(hashtags) > 0
        text = re.sub(hashtag_pattern, ' HASHTAG_TOKEN ', text)
        
        # Nettoyage des entités HTML
        text = re.sub(r'&amp;', ' and ', text)
        text = re.sub(r'&lt;', ' < ', text)
        text = re.sub(r'&gt;', ' > ', text)
        text = re.sub(r'&quot;', ' " ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        
        # Normalisation des caractères répétés (mais préservation de l'emphase)
        text = re.sub(r'([!?]){3,}', r'\1\1', text)  # Max 2 répétitions
        text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)  # Max 2 lettres répétées
        
        # Gestion des chiffres - conservation des numéros importants
        text = re.sub(r'\b\d{1,2}\b', ' NUMBER_TOKEN ', text)  # Petits nombres
        text = re.sub(r'\b\d{3,}\b', ' BIGNUMBER_TOKEN ', text)  # Grands nombres
        
        # Nettoyage de la ponctuation (en préservant certains caractères)
        text = re.sub(r'[^\w\s!?.\-]', ' ', text)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Conversion en minuscules
        text = text.lower()
        
        return text
    
    def extract_features(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extrait les features techniques identifiées dans l'analyse
        Version optimisée sans redondances et avec nouvelles features
        """
        text = str(row['text']) if pd.notna(row['text']) else ''
        keyword = row.get('keyword', None)
        
        features = {
            # Features de longueur (suppression de char_count, redondant avec text_length)
            'text_length': len(text),
            'word_count': len(text.split()) if text else 0,
            
            # Features techniques (patterns identifiés)
            'has_url': bool(re.search(r'http[s]?://', text)),
            'has_mention': bool(re.search(r'@[A-Za-z0-9_]+', text)),
            'has_hashtag': bool(re.search(r'#[A-Za-z0-9_]+', text)),
            'url_count': len(re.findall(r'http[s]?://', text)),
            'mention_count': len(re.findall(r'@[A-Za-z0-9_]+', text)),
            'hashtag_count': len(re.findall(r'#[A-Za-z0-9_]+', text)),
            
            # Features de contenu expressif
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'number_count': len(re.findall(r'\d+', text)),
            
            # Features sémantiques
            'has_emergency_word': any(word in text.lower() for word in self.emergency_keywords),
            
            # Nouvelle feature: keyword significatif (remplace has_keyword quasi-constante)
            'has_meaningful_keyword': bool(pd.notna(keyword) and str(keyword).lower().strip() not in ['', 'none', 'nan', 'null']),
            
            # Nouvelle feature: densité d'information (mots uniques / total mots)
            'word_diversity': len(set(text.lower().split())) / len(text.split()) if text.split() else 0
            
            # ❌ Suppression de 'has_keyword' : feature quasi-constante (99.2% = True)
            # ❌ Suppression de 'char_count' : redondant avec text_length
        }
        
        return features
    
    def resolve_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Résout les conflits de labels (textes identiques avec targets différents)
        Version améliorée avec stratégie intelligente
        """
        print("🔄 Résolution des conflits de labels...")
        
        # Nettoyer le texte avant la comparaison pour éviter les faux conflits
        df_temp = df.copy()
        df_temp['text_normalized'] = df_temp['text'].apply(self.clean_text)
        
        # Identifier les textes en conflit basés sur le texte nettoyé
        text_groups = df_temp.groupby('text_normalized')['target'].agg(['nunique', 'count', list])
        conflicts = text_groups[text_groups['nunique'] > 1]
        
        if len(conflicts) > 0:
            print(f"⚠️  {len(conflicts)} textes avec labels conflictuels trouvés")
            conflicts_resolved = 0
            
            for text_norm, info in conflicts.iterrows():
                # Stratégies de résolution par priorité:
                # 1. Label majoritaire
                # 2. En cas d'égalité, privilégier classe positive (catastrophe = 1)
                targets = info['list']
                target_counts = {t: targets.count(t) for t in set(targets)}
                
                if target_counts[1] >= target_counts[0]:
                    # Si égalité ou majorité pour classe 1, choisir 1
                    majority_target = 1
                    strategy = "classe positive privilégiée"
                else:
                    # Sinon, prendre le majoritaire
                    majority_target = 0
                    strategy = "majorité"
                
                # Mettre à jour tous les tweets avec ce texte normalisé
                mask = df_temp['text_normalized'] == text_norm
                df.loc[mask, 'target'] = majority_target
                conflicts_resolved += 1
                
                print(f"   Conflit résolu ({strategy}): {len(targets)} tweets → target {majority_target}")
            
            print(f"✅ {conflicts_resolved} conflits résolus avec succès")
        else:
            print("✅ Aucun conflit de label trouvé")
        
        return df
    
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les doublons selon les stratégies identifiées
        Version améliorée avec nettoyage plus agressif
        """
        print("🔄 Gestion des doublons...")
        
        initial_size = len(df)
        
        # 1. Suppression des doublons complets (toutes colonnes identiques)
        df_no_complete_dupes = df.drop_duplicates()
        complete_dupes_removed = initial_size - len(df_no_complete_dupes)
        
        if complete_dupes_removed > 0:
            print(f"🗑️  {complete_dupes_removed} doublons complets supprimés")
        
        # 2. Gestion des doublons de texte basés sur le texte nettoyé
        # Créer une version nettoyée pour identifier les vrais doublons
        df_temp = df_no_complete_dupes.copy()
        df_temp['text_normalized'] = df_temp['text'].apply(self.clean_text)
        
        # Stratégie: garder la première occurrence de chaque texte nettoyé
        initial_after_complete = len(df_temp)
        df_no_text_dupes = df_temp.drop_duplicates(subset=['text_normalized'], keep='first')
        text_dupes_removed = initial_after_complete - len(df_no_text_dupes)
        
        # Supprimer la colonne temporaire
        if 'text_normalized' in df_no_text_dupes.columns:
            df_no_text_dupes = df_no_text_dupes.drop('text_normalized', axis=1)
        
        if text_dupes_removed > 0:
            print(f"🗑️  {text_dupes_removed} doublons de texte supprimés (basé sur texte nettoyé)")
        
        total_removed = initial_size - len(df_no_text_dupes)
        print(f"✅ Dataset final: {len(df_no_text_dupes)} tweets ({total_removed} supprimés au total)")
        
        return df_no_text_dupes
    
    def process_dataset(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
        """
        Traite un dataset complet selon le pipeline identifié
        """
        print(f"\n🚀 PREPROCESSING - {dataset_name}")
        print("=" * 50)
        print(f"📊 Dataset initial: {len(df)} tweets")
        
        # Copie pour éviter les modifications inattendues
        df_processed = df.copy()
        
        # 1. Résolution des conflits de labels (train seulement)
        if 'target' in df_processed.columns and dataset_name.lower() == 'train':
            df_processed = self.resolve_conflicts(df_processed)
        
        # 2. Gestion des doublons
        if self.should_handle_duplicates:
            df_processed = self.handle_duplicates(df_processed)
        
        # 3. Suppression de la colonne location si demandé
        if self.remove_location and 'location' in df_processed.columns:
            print("🗑️  Suppression de la colonne 'location'")
            df_processed = df_processed.drop('location', axis=1)
        
        # 4. Nettoyage du texte
        print("🧹 Nettoyage du texte...")
        df_processed['text_cleaned'] = df_processed['text'].apply(self.clean_text)
        
        # 5. Extraction des features (avant suppression de 'text')
        print("⚙️  Extraction des features...")
        features_list = []
        for _, row in df_processed.iterrows():
            features = self.extract_features(row)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # 5b. Suppression de la colonne text originale (plus nécessaire pour ML)
        print("🗑️  Suppression de la colonne 'text' originale")
        df_processed = df_processed.drop('text', axis=1)
        
        # 6. Combinaison des données
        df_final = pd.concat([df_processed.reset_index(drop=True), 
                             features_df.reset_index(drop=True)], axis=1)
        
        # 7. Suppression des tweets vides après nettoyage
        empty_mask = (df_final['text_cleaned'] == '') | (df_final['text_cleaned'].isna())
        empty_count = empty_mask.sum()
        if empty_count > 0:
            print(f"🗑️  {empty_count} tweets vides après nettoyage supprimés")
            df_final = df_final[~empty_mask]
        
        print(f"✅ Dataset final: {len(df_final)} tweets")
        print(f"📊 Nouvelles colonnes: {list(features_df.columns)}")
        
        return df_final
    
    def get_preprocessing_report(self, df_original: pd.DataFrame, 
                               df_processed: pd.DataFrame) -> Dict[str, Any]:
        """
        Génère un rapport de preprocessing
        """
        report = {
            'original_size': len(df_original),
            'processed_size': len(df_processed),
            'rows_removed': len(df_original) - len(df_processed),
            'removal_percentage': (len(df_original) - len(df_processed)) / len(df_original) * 100,
            'original_columns': list(df_original.columns),
            'processed_columns': list(df_processed.columns),
            'new_features': [col for col in df_processed.columns if col not in df_original.columns],
            'text_stats': {
                'avg_length_original': df_original['text'].str.len().mean(),
                'avg_length_cleaned': df_processed['text_cleaned'].str.len().mean(),
                'avg_words_original': df_original['text'].str.split().str.len().mean(),
                'avg_words_cleaned': df_processed['text_cleaned'].str.split().str.len().mean()
            }
        }
        
        if 'target' in df_processed.columns:
            report['class_distribution'] = df_processed['target'].value_counts().to_dict()
        
        return report


def create_optimized_datasets(train_path: str, test_path: str, 
                            output_dir: str) -> Tuple[str, str]:
    """
    Fonction principale pour créer les datasets optimisés
    
    Args:
        train_path: Chemin vers le fichier train
        test_path: Chemin vers le fichier test
        output_dir: Répertoire de sortie
    
    Returns:
        Tuple des chemins des fichiers créés (train_optimized, test_optimized)
    """
    
    print("🎯 CRÉATION DES DATASETS OPTIMISÉS POUR ML")
    print("=" * 60)
    
    # Initialisation du preprocessor
    preprocessor = EmergencyTweetPreprocessor(
        remove_location=True,  # Suppression recommandée par l'analyse
        handle_duplicates=True
    )
    
    # Chargement des données
    print("📁 Chargement des données...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Train: {len(train_df)} tweets")
    print(f"✅ Test: {len(test_df)} tweets")
    
    # Preprocessing
    train_processed = preprocessor.process_dataset(train_df, "Train")
    test_processed = preprocessor.process_dataset(test_df, "Test")
    
    # Sauvegarde
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, 'train_optimized_v1.csv')
    test_output_path = os.path.join(output_dir, 'test_optimized_v1.csv')
    
    train_processed.to_csv(train_output_path, index=False)
    test_processed.to_csv(test_output_path, index=False)
    
    print(f"\n💾 Fichiers sauvegardés:")
    print(f"   Train: {train_output_path}")
    print(f"   Test: {test_output_path}")
    
    # Génération des rapports
    train_report = preprocessor.get_preprocessing_report(train_df, train_processed)
    test_report = preprocessor.get_preprocessing_report(test_df, test_processed)
    
    # Affichage du résumé
    print(f"\n📊 RÉSUMÉ DU PREPROCESSING")
    print("=" * 40)
    print(f"Train: {train_report['original_size']} → {train_report['processed_size']} "
          f"({train_report['removal_percentage']:.1f}% supprimés)")
    print(f"Test:  {test_report['original_size']} → {test_report['processed_size']} "
          f"({test_report['removal_percentage']:.1f}% supprimés)")
    print(f"Features ajoutées: {len(train_report['new_features'])}")
    
    return train_output_path, test_output_path


# Fonctions utilitaires supplémentaires

def analyze_feature_importance(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Analyse l'importance des features extraites
    """
    if target_col not in df.columns:
        print("⚠️  Colonne target non trouvée")
        return pd.DataFrame()
    
    # Features numériques pour corrélation
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != target_col]
    
    correlations = []
    for feature in numeric_features:
        corr = df[feature].corr(df[target_col])
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    return corr_df


def validate_processed_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valide la qualité des données après preprocessing
    """
    validation = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'text_cleaned_empty': (df['text_cleaned'] == '').sum(),
        'duplicate_texts': df['text_cleaned'].duplicated().sum(),
        'basic_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    
    if 'target' in df.columns:
        validation['class_distribution'] = df['target'].value_counts().to_dict()
        validation['class_balance_ratio'] = df['target'].value_counts().min() / df['target'].value_counts().max()
    
    return validation
