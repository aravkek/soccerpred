#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:22:14 2025

@author: aravkekane
"""

"""
League-Specific Binary Training - OPTIMIZED
Uses selected features from feature_selection.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from config import STORAGE, LEAGUES


class OptimizedLeagueTrainer:
    """Train models with optimized feature sets per league"""
    
    def __init__(self):
        self.processed_path = Path(STORAGE['processed_data_path'])
        self.models_dir = Path('models/league_specific_optimized')
        self.feature_selection_dir = Path('models/feature_selection')
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
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
        return df
    
    def load_selected_features(self, league_code):
        """Load selected features for a league"""
        features_file = self.feature_selection_dir / f"{league_code}_selected_features.txt"
        
        if not features_file.exists():
            print(f"  No feature selection file found, using all features")
            return None
        
        with open(features_file, 'r') as f:
            selected_features = [line.strip() for line in f]
        
        print(f"  Using {len(selected_features)} selected features")
        return selected_features
    
    def train_league_model(self, df, league_code, league_name):
        """Train optimized model for specific league"""
        print("\n" + "="*60)
        print(f"TRAINING: {league_name}")
        print("="*60)
        
        # Filter league
        league_df = df[df['competition_code'] == league_code].copy()
        print(f"Samples: {len(league_df)}")
        
        if len(league_df) < 100:
            print(f"⚠️  Not enough data, skipping...")
            return None
        
        # Get selected features
        selected_features = self.load_selected_features(league_code)
        
        # Prepare features
        exclude_cols = ['match_id', 'season', 'competition_code', 'home_team_id', 
                       'away_team_id', 'matchday', 'result', 'home_win', 'draw', 'away_win']
        
        if selected_features:
            feature_cols = selected_features
        else:
            feature_cols = [col for col in league_df.columns if col not in exclude_cols]
        
        X = league_df[feature_cols]
        y = league_df['home_win']
        
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"Home Win Rate: {y.mean():.2%}")
        print(f"Using {len(feature_cols)} features")
        
        # Temporal split
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_acc = rf_model.score(X_test_scaled, y_test)
        
        # Train XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_acc = xgb_model.score(X_test_scaled, y_test)
        
        print(f"\nRandom Forest: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
        print(f"XGBoost: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
        
        # Select best
        if rf_acc > xgb_acc:
            best_model = rf_model
            best_name = 'random_forest'
            best_acc = rf_acc
        else:
            best_model = xgb_model
            best_name = 'xgboost'
            best_acc = xgb_acc
        
        print(f"✓ Best: {best_name} ({best_acc*100:.2f}%)")
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d')
        model_path = self.models_dir / f"{league_code}_{best_name}_optimized_{timestamp}.pkl"
        scaler_path = self.models_dir / f"{league_code}_scaler_optimized_{timestamp}.pkl"
        features_path = self.models_dir / f"{league_code}_features_optimized_{timestamp}.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        
        return {
            'league_code': league_code,
            'league_name': league_name,
            'model': best_name,
            'accuracy': best_acc,
            'samples': len(league_df),
            'test_samples': len(X_test),
            'num_features': len(feature_cols)
        }
    
    def train_all_leagues(self):
        """Train optimized models for all leagues"""
        print("="*60)
        print("TRAINING OPTIMIZED LEAGUE-SPECIFIC MODELS")
        print("="*60)
        
        df = self.load_dataset()
        
        results = []
        for league_key, league_info in LEAGUES.items():
            result = self.train_league_model(
                df,
                league_info['id'],
                league_info['name']
            )
            if result:
                results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("OPTIMIZED MODEL PERFORMANCE")
        print("="*60)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print("\n" + results_df.to_string(index=False))
        
        print("\n✓ Optimized models saved!")
        print(f"\nBest: {results_df.iloc[0]['league_name']} ({results_df.iloc[0]['accuracy']*100:.2f}%)")
        print(f"Worst: {results_df.iloc[-1]['league_name']} ({results_df.iloc[-1]['accuracy']*100:.2f}%)")
        
        return results_df


if __name__ == "__main__":
    trainer = OptimizedLeagueTrainer()
    results = trainer.train_all_leagues()