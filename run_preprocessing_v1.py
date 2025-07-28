"""
Script pour exécuter le preprocessing des données Eye of Emergency
"""

from src.preprocessing import create_optimized_datasets
import os
import pandas as pd

def main():
    print('🚀 DÉBUT DU PREPROCESSING - EYE OF EMERGENCY')
    print('=' * 60)
    
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
        
        # Création des datasets optimisés
        train_path, test_path = create_optimized_datasets(
            train_path=train_file,
            test_path=test_file,
            output_dir='data/processed'
        )
        
        print('\n✅ PREPROCESSING TERMINÉ AVEC SUCCÈS!')
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
            
            # Affichage des nouvelles colonnes
            original_cols = ['id', 'keyword', 'location', 'text', 'target']
            new_cols = [col for col in train_df.columns if col not in original_cols]
            print(f'   Nouvelles features: {len(new_cols)}')
            print(f'   Features ajoutées: {", ".join(new_cols[:5])}{"..." if len(new_cols) > 5 else ""}')
            
            # Distribution des classes
            if 'target' in train_df.columns:
                class_dist = train_df['target'].value_counts()
                print(f'   Distribution des classes: {dict(class_dist)}')
            
        else:
            print('❌ Erreur: Fichiers non créés correctement')
            
    except Exception as e:
        print(f'❌ Erreur pendant le preprocessing: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
