#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection - Remove low-importance features
Analyzes which features help vs hurt performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from config import STORAGE, LEAGUES


class FeatureSelector:
    """Select best features per league"""
    
    def __init__(self):
        self.processed_path = Path(STORAGE['processed_data_path'])
        self.results = {}
    
    def load_dataset(self):
        """Load V2 dataset"""
        datasets = list(self.processed_path.glob('training_dataset_v2_enhanced_*.csv'))
        if not datasets:
            datasets = list(self.processed_path.glob('training_dataset_enhanced_all_*.csv'))
        if not datasets:
            raise FileNotFoundError("No dataset found!")
        
        dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
        print(f"Loading: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Features: {len(df.columns)}")
        return df
    
    def analyze_league_features(self, df, league_code, league_name):
        """Analyze feature importance for a specific league"""
        print("\n" + "="*60)
        print(f"ANALYZING: {league_name}")
        print("="*60)
        
        # Filter league
        league_df = df[df['competition_code'] == league_code].copy()
        
        if len(league_df) < 100:
            print("Not enough data, skipping...")
            return None
        
        # Prepare data
        exclude_cols = ['match_id', 'season', 'competition_code', 'home_team_id', 
                       'away_team_id', 'matchday', 'result', 'home_win', 'draw', 'away_win']
        feature_cols = [col for col in league_df.columns if col not in exclude_cols]
        
        X = league_df[feature_cols]
        y = league_df['home_win']
        
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train baseline model with ALL features
        rf_all = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_all.fit(X_train_scaled, y_train)
        baseline_acc = rf_all.score(X_test_scaled, y_test)
        
        print(f"Baseline (all {len(feature_cols)} features): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_all.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(importances.head(15).to_string(index=False))
        
        print("\nBottom 15 Least Important Features:")
        print(importances.tail(15).to_string(index=False))
        
        # Try removing bottom features
        thresholds = [0.001, 0.002, 0.003, 0.005]
        
        best_threshold = None
        best_acc = baseline_acc
        best_features = feature_cols
        
        print("\nTesting feature removal thresholds:")
        for threshold in thresholds:
            # Select features above threshold
            selector = SelectFromModel(rf_all, threshold=threshold, prefit=True)
            X_train_selected = selector.transform(X_train_scaled)
            X_test_selected = selector.transform(X_test_scaled)
            
            selected_features = [f for f, i in zip(feature_cols, rf_all.feature_importances_) if i >= threshold]
            
            # Train with selected features
            rf_selected = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            rf_selected.fit(X_train_selected, y_train)
            selected_acc = rf_selected.score(X_test_selected, y_test)
            
            improvement = selected_acc - baseline_acc
            
            print(f"  Threshold {threshold:.4f}: {len(selected_features)} features → {selected_acc:.4f} ({selected_acc*100:.2f}%) [{improvement:+.4f}]")
            
            if selected_acc > best_acc:
                best_acc = selected_acc
                best_threshold = threshold
                best_features = selected_features
        
        if best_threshold:
            print(f"\n✓ IMPROVED! Using threshold {best_threshold}")
            print(f"  Old: {baseline_acc*100:.2f}% with {len(feature_cols)} features")
            print(f"  New: {best_acc*100:.2f}% with {len(best_features)} features")
            print(f"  Gain: {(best_acc - baseline_acc)*100:+.2f}%")
        else:
            print(f"\n→ No improvement from feature selection")
            print(f"  Keeping all {len(feature_cols)} features")
        
        return {
            'league_code': league_code,
            'league_name': league_name,
            'baseline_acc': baseline_acc,
            'best_acc': best_acc,
            'improvement': best_acc - baseline_acc,
            'original_features': len(feature_cols),
            'selected_features': len(best_features),
            'features_removed': len(feature_cols) - len(best_features),
            'best_threshold': best_threshold,
            'selected_feature_list': best_features,
            'feature_importances': importances
        }
    
    def select_features_all_leagues(self):
        """Run feature selection for all leagues"""
        print("="*60)
        print("FEATURE SELECTION ANALYSIS")
        print("="*60)
        
        df = self.load_dataset()
        
        results = []
        for league_key, league_info in LEAGUES.items():
            result = self.analyze_league_features(
                df,
                league_info['id'],
                league_info['name']
            )
            if result:
                results.append(result)
                self.results[league_info['id']] = result
        
        # Summary
        print("\n" + "="*60)
        print("FEATURE SELECTION SUMMARY")
        print("="*60)
        
        summary_df = pd.DataFrame([{
            'league_code': r['league_code'],
            'league_name': r['league_name'],
            'baseline_acc': r['baseline_acc'],
            'best_acc': r['best_acc'],
            'improvement': r['improvement'],
            'features': f"{r['selected_features']}/{r['original_features']}"
        } for r in results])
        
        summary_df = summary_df.sort_values('improvement', ascending=False)
        print("\n" + summary_df.to_string(index=False))
        
        total_improvement = summary_df['improvement'].sum()
        print(f"\nTotal accuracy gain: {total_improvement*100:+.2f}%")
        
        # Save selected features
        output_dir = Path('models/feature_selection')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            league_code = result['league_code']
            features_file = output_dir / f"{league_code}_selected_features.txt"
            
            with open(features_file, 'w') as f:
                for feature in result['selected_feature_list']:
                    f.write(f"{feature}\n")
            
            print(f"✓ Saved: {features_file}")
        
        return summary_df
    
    def create_optimized_dataset(self, df):
        """Create new dataset with only selected features per league"""
        print("\n" + "="*60)
        print("CREATING OPTIMIZED DATASET")
        print("="*60)
        
        # This keeps all features but saves which ones to use per league
        # The training script will load the selected features list
        
        print("✓ Feature selection complete!")
        print("\nNext step: Retrain models with selected features")
        print("Run: python league_specific_binary_training_OPTIMIZED.py")


if __name__ == "__main__":
    selector = FeatureSelector()
    summary = selector.select_features_all_leagues()