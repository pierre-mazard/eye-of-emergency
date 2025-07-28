"""
Script pour tester le preprocessing optimisé basé sur la validation
"""

from src.preprocessing_optimized import create_optimized_datasets_v2
import os
import pandas as pd

def main():
    print('🚀 DÉBUT DU PREPROCESSING OPTIMISÉ V2.0 - EYE OF EMERGENCY')
    print('=' * 70)
    
    try:
        # Vérification des fichiers d'entrée
        train_file = 'data/raw/train_tweets.csv'
        test_file = 'data/raw/test_tweets.csv'
        
        if not os.path.exists(train_file):
            print(f'❌ Fichier non trouvé: {train_file}')
            return
        
        if not os.path.exists(test_file):
            print(f'❌ Fichier non trouvé: {test_file}')
            return
        
        print(f'✅ Fichiers d\'entrée trouvés')
        
        # Création des datasets optimisés v2
        train_path, test_path = create_optimized_datasets_v2(
            train_path=train_file,
            test_path=test_file,
            output_dir='data/processed'
        )
        
        print('\n✅ PREPROCESSING OPTIMISÉ TERMINÉ AVEC SUCCÈS!')
        print(f'📁 Fichiers créés:')
        print(f'   - {train_path}')
        print(f'   - {test_path}')
        
        # Vérification des résultats
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print(f'\n📊 RÉSULTATS FINAUX:')
            print(f'   Train: {len(train_df):,} tweets, {len(train_df.columns)} colonnes')
            print(f'   Test:  {len(test_df):,} tweets, {len(test_df.columns)} colonnes')
            
            # Affichage des colonnes finales
            print(f'   Colonnes: {list(train_df.columns)}')
            
            # Distribution des classes
            if 'target' in train_df.columns:
                class_dist = train_df['target'].value_counts()
                print(f'   Distribution des classes: {dict(class_dist)}')
                ratio = class_dist.max() / class_dist.min()
                print(f'   Ratio d\'équilibrage: {ratio:.2f}')
            
            # Comparaison avec la version précédente
            print(f'\n🔍 COMPARAISON AVEC VERSION PRÉCÉDENTE:')
            
            old_train_path = 'data/processed/train_optimized.csv'
            if os.path.exists(old_train_path):
                old_train = pd.read_csv(old_train_path)
                print(f'   V1: {len(old_train.columns)} colonnes → V2: {len(train_df.columns)} colonnes')
                print(f'   Réduction: {len(old_train.columns) - len(train_df.columns)} features supprimées')
                
                # Features supprimées
                old_features = set(old_train.columns)
                new_features = set(train_df.columns)
                removed = old_features - new_features
                added = new_features - old_features
                
                if removed:
                    print(f'   Features supprimées: {", ".join(removed)}')
                if added:
                    print(f'   Nouvelles features: {", ".join(added)}')
            
        else:
            print('❌ Erreur: Fichiers non créés correctement')
            
    except Exception as e:
        print(f'❌ Erreur pendant le preprocessing: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
