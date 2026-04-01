#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Learning System - FIXED VERSION
- Compares adaptive/ vs league_specific/ models
- Uses the best performer
- Handles feature mismatches properly
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

from config import STORAGE, LEAGUES
from feature_engineering import FeatureEngineer

class AdaptiveLearningSystem:
    """Self-improving prediction system"""
    
    def __init__(self):
        self.db_path = Path(STORAGE['database_path'])
        self.models_dir = Path('models/adaptive')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_log_path = self.models_dir / 'performance_log.csv'
        self.retrain_log_path = self.models_dir / 'retrain_log.csv'
        
        # Initialize performance tracking
        self._init_performance_tracking()
        
        print("Adaptive Learning System initialized")
    
    def _init_performance_tracking(self):
        """Initialize performance tracking database"""
        # Create performance log if doesn't exist
        if not self.performance_log_path.exists():
            df = pd.DataFrame(columns=[
                'prediction_id', 'timestamp', 'league_code', 'league_name',
                'match_id', 'home_team', 'away_team', 'predicted_home_win',
                'predicted_prob', 'actual_home_win', 'correct', 'model_version'
            ])
            df.to_csv(self.performance_log_path, index=False)
        
        # Create retrain log if doesn't exist
        if not self.retrain_log_path.exists():
            df = pd.DataFrame(columns=[
                'retrain_id', 'timestamp', 'league_code', 'samples_added',
                'old_accuracy', 'new_accuracy', 'improvement', 'model_source'
            ])
            df.to_csv(self.retrain_log_path, index=False)
    
    def load_best_league_model(self, league_code, X_test=None, y_test=None):
        """
        Load the BEST model for a league by comparing adaptive vs league_specific
        
        Args:
            league_code: League code
            X_test: Optional test data to evaluate models
            y_test: Optional test labels
            
        Returns:
            model, scaler, source, accuracy
        """
        print(f"\nSearching for best model for {league_code}...")
        
        # Find models in both directories
        adaptive_models = list(self.models_dir.glob(f"{league_code}_*.pkl"))
        league_specific_models = list(Path('models/league_specific').glob(f"{league_code}_*.pkl"))
        
        candidates = []
        
        # Check adaptive models
        if adaptive_models:
            latest_adaptive = max(adaptive_models, key=lambda p: p.stat().st_mtime)
            candidates.append(('adaptive', latest_adaptive))
            print(f"  Found adaptive model: {latest_adaptive.name}")
        
        # Check league_specific models
        if league_specific_models:
            latest_specific = max(league_specific_models, key=lambda p: p.stat().st_mtime)
            candidates.append(('league_specific', latest_specific))
            print(f"  Found league_specific model: {latest_specific.name}")
        
        if not candidates:
            raise FileNotFoundError(f"No model found for league {league_code}")
        
        # If we have test data, compare models
        if X_test is not None and y_test is not None:
            best_source = None
            best_model = None
            best_scaler = None
            best_accuracy = -1
            
            for source, model_path in candidates:
                try:
                    # Load model and scaler
                    scaler_path = model_path.parent / model_path.name.replace('random_forest', 'scaler').replace('xgboost', 'scaler')
                    features_path = model_path.parent / model_path.name.replace('random_forest', 'features').replace('xgboost', 'features').replace('.pkl', '.json')
                    
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    if scaler_path.exists():
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                    else:
                        scaler = None
                    
                    # Load feature order if available
                    if features_path.exists():
                        with open(features_path, 'r') as f:
                            expected_features = json.load(f)
                        
                        # Reorder X_test to match expected features
                        X_test_reordered = X_test[expected_features]
                    else:
                        X_test_reordered = X_test
                    
                    # Test the model
                    if scaler:
                        X_test_scaled = scaler.transform(X_test_reordered)
                    else:
                        X_test_scaled = X_test_reordered
                    
                    accuracy = model.score(X_test_scaled, y_test)
                    print(f"  {source} model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                        best_scaler = scaler
                        best_source = source
                
                except Exception as e:
                    print(f"  ⚠️ Could not evaluate {source} model: {e}")
                    continue
            
            if best_model:
                print(f"  ✓ Using {best_source} model ({best_accuracy*100:.2f}%)")
                return best_model, best_scaler, best_source, best_accuracy
        
        # If no test data or comparison failed, use most recent
        source, model_path = candidates[0]
        scaler_path = model_path.parent / model_path.name.replace('random_forest', 'scaler').replace('xgboost', 'scaler')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        
        print(f"  ✓ Using {source} model (no comparison available)")
        return model, scaler, source, None
    
    def get_recent_performance(self, league_code, days=30):
        """
        Get recent prediction performance for a league
        """
        if not self.performance_log_path.exists():
            return None
        
        df = pd.read_csv(self.performance_log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by league and time window
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = df[
            (df['league_code'] == league_code) &
            (df['timestamp'] >= cutoff_date) &
            (df['actual_home_win'].notna())  # Only completed matches
        ]
        
        if len(recent) == 0:
            return None
        
        accuracy = recent['correct'].mean()
        total_predictions = len(recent)
        
        return {
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'period_days': days
        }
    
    def should_retrain(self, league_code, min_new_matches=20, check_days=30):
        """
        Determine if model should be retrained
        """
        conn = sqlite3.connect(self.db_path)
        
        # Check when model was last updated
        if self.retrain_log_path.exists():
            retrain_log = pd.read_csv(self.retrain_log_path)
            league_retrains = retrain_log[retrain_log['league_code'] == league_code]
            
            if len(league_retrains) > 0:
                last_retrain = pd.to_datetime(league_retrains['timestamp'].max())
            else:
                last_retrain = datetime.now() - timedelta(days=365)  # Very old
        else:
            last_retrain = datetime.now() - timedelta(days=365)
        
        # Count new matches since last retrain
        query = """
            SELECT COUNT(*) as new_matches
            FROM matches
            WHERE competition_code = ?
            AND status = 'FINISHED'
            AND datetime(utc_date) > datetime(?)
        """
        
        result = pd.read_sql_query(query, conn, params=[league_code, last_retrain.isoformat()])
        conn.close()
        
        new_matches = result['new_matches'].values[0]
        
        # Check recent performance
        recent_perf = self.get_recent_performance(league_code, days=check_days)
        
        reasons = []
        should_retrain = False
        
        # Reason 1: Enough new data
        if new_matches >= min_new_matches:
            should_retrain = True
            reasons.append(f"{new_matches} new matches available")
        
        # Reason 2: Performance degradation
        if recent_perf and recent_perf['accuracy'] < 0.55:
            should_retrain = True
            reasons.append(f"Performance dropped to {recent_perf['accuracy']:.2%}")
        
        # Reason 3: Been too long (30+ days)
        days_since_retrain = (datetime.now() - last_retrain).days
        if days_since_retrain > 30:
            should_retrain = True
            reasons.append(f"{days_since_retrain} days since last retrain")
        
        return should_retrain, reasons
    
    def retrain_league_model(self, league_code, league_name):
        """
        Retrain model for a specific league with latest data
        """
        print("\n" + "="*60)
        print(f"RETRAINING: {league_name}")
        print("="*60)
        
        # Rebuild features with latest data
        engineer = FeatureEngineer()
        
        # Load matches
        df = engineer.load_matches(competition_code=league_code)
        df = engineer.create_basic_features(df)
        standings_df = engineer.load_standings(competition_code=league_code)
        
        print(f"Total matches: {len(df)}")
        
        # Build features for each match
        features_list = []
        
        for idx, match in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(df)}...")
            
            match_date = match['utc_date']
            home_team = match['home_team_id']
            away_team = match['away_team_id']
            season = match['season']
            competition = match['competition_code']
            matchday = match['matchday']
            
            # Skip matches without enough historical data
            historical_count = len(df[
                (df['utc_date'] < match_date) &
                (df['season'] == season) &
                (df['competition_code'] == competition)
            ])
            
            if historical_count < 10:
                continue
            
            # Build match features
            match_features = {
                'match_id': match['match_id'],
                'home_win': match['home_win']
            }
            
            # Get all features
            home_form = engineer.get_team_form_at_date(df, home_team, match_date, is_home=True, n_matches=10)
            if home_form:
                for key, value in home_form.items():
                    match_features[f'home_{key}'] = value
            
            away_form = engineer.get_team_form_at_date(df, away_team, match_date, is_home=False, n_matches=10)
            if away_form:
                for key, value in away_form.items():
                    match_features[f'away_{key}'] = value
            
            h2h_features = engineer.get_h2h_features(df, home_team, away_team, match_date)
            match_features.update(h2h_features)
            
            position_features = engineer.get_league_position_features(
                standings_df, season, competition, home_team, away_team, matchday
            )
            match_features.update(position_features)
            
            home_rest = engineer.get_rest_days_features(df, home_team, match_date)
            away_rest = engineer.get_rest_days_features(df, away_team, match_date)
            match_features['home_rest_days'] = home_rest['rest_days']
            match_features['away_rest_days'] = away_rest['rest_days']
            match_features['rest_days_diff'] = home_rest['rest_days'] - away_rest['rest_days']
            
            season_features = engineer.get_season_stage_features(matchday)
            match_features.update(season_features)
            
            # Relative strength features
            if home_form and away_form:
                match_features['attack_vs_defense'] = (
                    home_form['goals_scored_per_match'] - away_form['goals_conceded_per_match']
                )
                match_features['defense_vs_attack'] = (
                    away_form['goals_scored_per_match'] - home_form['goals_conceded_per_match']
                )
                match_features['form_diff'] = (
                    home_form['points_per_match'] - away_form['points_per_match']
                )
                match_features['momentum_diff'] = (
                    home_form['form_trend'] - away_form['form_trend']
                )
            
            features_list.append(match_features)
        
        dataset = pd.DataFrame(features_list).fillna(0)
        
        print(f"\nFeatures built: {len(dataset)} samples")
        
        # Prepare data
        exclude_cols = ['match_id', 'home_win']
        feature_cols = [col for col in dataset.columns if col not in exclude_cols]
        
        X = dataset[feature_cols]
        y = dataset['home_win']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Get old model accuracy for comparison (FIXED VERSION)
        try:
            old_model, old_scaler, old_source, old_accuracy = self.load_best_league_model(
                league_code, X_test, y_test
            )
            
            if old_accuracy is None:
                # Couldn't compare, set baseline
                old_accuracy = 0.0
                old_source = 'none'
            
            print(f"\nCurrent best model: {old_source} ({old_accuracy*100:.2f}%)")
            
        except Exception as e:
            print(f"\n⚠️ Could not load old model: {e}")
            old_accuracy = 0.0
            old_source = 'none'
        
        # Train new model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest with BETTER hyperparameters
        model = RandomForestClassifier(
            n_estimators=500,  # More trees
            max_depth=15,  # Deeper trees
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        new_accuracy = model.score(X_test_scaled, y_test)
        
        improvement = new_accuracy - old_accuracy
        
        print(f"\nOld Accuracy: {old_accuracy:.4f} ({old_accuracy*100:.2f}%)")
        print(f"New Accuracy: {new_accuracy:.4f} ({new_accuracy*100:.2f}%)")
        print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        # Save new model (always save to adaptive/)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.models_dir / f"{league_code}_random_forest_{timestamp}.pkl"
        scaler_path = self.models_dir / f"{league_code}_scaler_{timestamp}.pkl"
        features_path = self.models_dir / f"{league_code}_features_{timestamp}.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        
        print(f"\n✓ New model saved: {model_path}")
        
        # Determine which model is now best
        if new_accuracy > old_accuracy:
            print(f"✓ NEW MODEL IS BETTER! ({new_accuracy*100:.2f}% vs {old_accuracy*100:.2f}%)")
            best_source = 'adaptive'
        else:
            print(f"⚠️ Old model still better ({old_accuracy*100:.2f}% vs {new_accuracy*100:.2f}%)")
            best_source = old_source
        
        # Log retrain
        retrain_log = pd.read_csv(self.retrain_log_path) if self.retrain_log_path.exists() else pd.DataFrame()
        
        new_log = pd.DataFrame([{
            'retrain_id': len(retrain_log) + 1,
            'timestamp': datetime.now().isoformat(),
            'league_code': league_code,
            'samples_added': len(dataset) - split_idx,
            'old_accuracy': old_accuracy,
            'new_accuracy': new_accuracy,
            'improvement': improvement,
            'model_source': best_source
        }])
        
        retrain_log = pd.concat([retrain_log, new_log], ignore_index=True)
        retrain_log.to_csv(self.retrain_log_path, index=False)
        
        return {
            'league_code': league_code,
            'league_name': league_name,
            'old_accuracy': old_accuracy,
            'new_accuracy': new_accuracy,
            'improvement': improvement,
            'samples': len(dataset),
            'best_source': best_source
        }
    
    def auto_retrain_all_leagues(self):
        """Check all leagues and retrain if needed"""
        print("\n" + "="*60)
        print("ADAPTIVE RETRAINING - CHECKING ALL LEAGUES")
        print("="*60)
        
        results = []
        
        for league_key, league_info in LEAGUES.items():
            league_code = league_info['id']
            league_name = league_info['name']
            
            print(f"\n{'='*60}")
            print(f"Checking: {league_name}")
            print(f"{'='*60}")
            
            should_retrain, reasons = self.should_retrain(league_code)
            
            if should_retrain:
                print(f"✓ Retraining needed:")
                for reason in reasons:
                    print(f"  - {reason}")
                
                result = self.retrain_league_model(league_code, league_name)
                results.append(result)
            else:
                print(f"✗ No retraining needed - model is up to date")
        
        if results:
            print("\n" + "="*60)
            print("RETRAINING SUMMARY")
            print("="*60)
            
            results_df = pd.DataFrame(results)
            print("\n" + results_df.to_string(index=False))
            
            avg_improvement = results_df['improvement'].mean()
            print(f"\nAverage Improvement: {avg_improvement:+.4f} ({avg_improvement*100:+.2f}%)")
        else:
            print("\n✓ All models are up to date!")
        
        return results


if __name__ == "__main__":
    print("ADAPTIVE LEARNING SYSTEM")
    print("="*60)
    
    system = AdaptiveLearningSystem()
    
    # Auto-retrain all leagues if needed
    results = system.auto_retrain_all_leagues()