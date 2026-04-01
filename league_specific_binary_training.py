#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train separate binary models for each league - FIXED VERSION
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

class LeagueSpecificTrainer:
    """Train separate models for each league"""
    
    def __init__(self):
        self.processed_path = Path(STORAGE['processed_data_path'])
        self.models_dir = Path('models/league_specific')
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self):
        """Load the full dataset - prioritize V2 files"""
        # Try V2 files first, then fall back to old files
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
    
    def train_league_model(self, df, league_code, league_name):
        """Train binary model for specific league"""
        print("\n" + "="*60)
        print(f"TRAINING: {league_name}")
        print("="*60)
        
        # Filter data for this league
        league_df = df[df['competition_code'] == league_code].copy()
        print(f"Samples: {len(league_df)}")
        
        if len(league_df) < 100:
            print(f"⚠️  Not enough data for {league_name}, skipping...")
            return None
        
        # Prepare features
        exclude_cols = ['match_id', 'season', 'competition_code', 'home_team_id', 
                       'away_team_id', 'matchday', 'result', 'home_win', 'draw', 'away_win']
        feature_cols = [col for col in league_df.columns if col not in exclude_cols]
        
        X = league_df[feature_cols]
        y = league_df['home_win']
        
        # Remove NaN
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
        
        print(f"\nRandom Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
        print(f"XGBoost Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
        
        # Save best model
        if rf_acc > xgb_acc:
            best_model = rf_model
            best_name = 'random_forest'
            best_acc = rf_acc
        else:
            best_model = xgb_model
            best_name = 'xgboost'
            best_acc = xgb_acc
        
        print(f"✓ Best: {best_name} ({best_acc*100:.2f}%)")
        
        # Save model, scaler, AND feature names
        timestamp = datetime.now().strftime('%Y%m%d')
        model_path = self.models_dir / f"{league_code}_{best_name}_{timestamp}.pkl"
        scaler_path = self.models_dir / f"{league_code}_scaler_{timestamp}.pkl"
        features_path = self.models_dir / f"{league_code}_features_{timestamp}.json"
        
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
        """Train models for all leagues"""
        print("\n" + "="*60)
        print("TRAINING LEAGUE-SPECIFIC BINARY MODELS")
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
        print("SUMMARY: LEAGUE-SPECIFIC PERFORMANCE")
        print("="*60)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print("\n" + results_df.to_string(index=False))
        
        print("\n✓ League-specific models saved!")
        print(f"\nBest League: {results_df.iloc[0]['league_name']} ({results_df.iloc[0]['accuracy']*100:.2f}%)")
        print(f"Worst League: {results_df.iloc[-1]['league_name']} ({results_df.iloc[-1]['accuracy']*100:.2f}%)")
        
        return results_df


if __name__ == "__main__":
    trainer = LeagueSpecificTrainer()
    results = trainer.train_all_leagues()