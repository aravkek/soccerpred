#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 02:08:22 2025

@author: aravkekane
"""

"""
Model Training Pipeline
Trains ensemble of ML models for match outcome prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb

from config import STORAGE, MVP_CONFIG

class ModelTrainer:
    """Trains and evaluates prediction models"""
    
    def __init__(self):
        """Initialize model trainer"""
        self.processed_path = Path(STORAGE['processed_data_path'])
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        
        print("ModelTrainer initialized")
    
    def load_dataset(self, dataset_path=None):
        """
        Load training dataset
        
        Args:
            dataset_path: Path to dataset CSV (if None, loads most recent)
            
        Returns:
            DataFrame
        """
        if dataset_path is None:
            # Find most recent dataset
            datasets = list(self.processed_path.glob('training_dataset_*.csv'))
            if not datasets:
                raise FileNotFoundError("No training dataset found. Run feature_engineering.py first!")
            dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
        
        print(f"\nLoading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples")
        
        return df
    
    def prepare_data(self, df, target='result'):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features
            target: Target column ('result' for multiclass, 'home_win' for binary)
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        print(f"\nPreparing data for target: {target}")
        
        # Define feature columns (exclude IDs and targets)
        exclude_cols = ['match_id', 'season', 'competition_code', 'home_team_id', 
                       'away_team_id', 'matchday', 'result', 'home_win', 'draw', 'away_win']
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Using {len(self.feature_columns)} features:")
        for col in self.feature_columns:
            print(f"  - {col}")
        
        X = df[self.feature_columns]
        
        # Prepare target
        if target == 'result':
            # Multiclass: H, D, A
            y = df['result']
            label_map = {'H': 0, 'D': 1, 'A': 2}
            y = y.map(label_map)
        elif target == 'home_win':
            # Binary: Home Win vs Not Home Win
            y = df['home_win']
        else:
            y = df[target]
        
        # Remove rows with missing targets
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"\nTarget distribution:")
        print(y.value_counts())
        
        # Split data - use temporal split for realistic evaluation
        # Train on older matches, test on recent matches
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        self.models['random_forest'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        print("\n" + "="*60)
        print("Training LightGBM")
        print("="*60)
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting model"""
        print("\n" + "="*60)
        print("Training Gradient Boosting")
        print("="*60)
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        
        return model
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name.upper()}")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Confusion matrix
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # Log loss (calibration metric)
            logloss = log_loss(y_test, y_pred_proba)
            print(f"\nLog Loss: {logloss:.4f}")
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'log_loss': logloss,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_model[0].upper()}")
        print(f"Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")
        print("="*60)
        
        return results
    
    def create_ensemble(self, X_test, y_test, results):
        """
        Create ensemble prediction by averaging model probabilities
        
        Args:
            X_test: Test features
            y_test: Test targets
            results: Dictionary of model results
            
        Returns:
            Ensemble accuracy
        """
        print("\n" + "="*60)
        print("ENSEMBLE MODEL (Averaging)")
        print("="*60)
        
        # Average probabilities from all models
        all_probs = np.array([results[name]['probabilities'] for name in results.keys()])
        ensemble_probs = np.mean(all_probs, axis=0)
        
        # Get predictions
        ensemble_pred = np.argmax(ensemble_probs, axis=1)
        
        # Evaluate
        accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nClassification Report:")
        print(classification_report(y_test, ensemble_pred))
        
        logloss = log_loss(y_test, ensemble_probs)
        print(f"\nLog Loss: {logloss:.4f}")
        
        return accuracy, ensemble_probs
    
    def save_models(self):
        """Save trained models and scaler"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = models_dir / f"{name}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {model_path}")
        
        # Save scaler
        scaler_path = models_dir / f"scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved: {scaler_path}")
        
        # Save feature columns
        features_path = models_dir / f"features_{timestamp}.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        print(f"Saved: {features_path}")
        
        print("\n✓ All models saved!")
    
    def run_full_training(self):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("STARTING FULL MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load dataset
        df = self.load_dataset()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target='result')
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Create ensemble
        ensemble_accuracy, ensemble_probs = self.create_ensemble(X_test, y_test, results)
        
        # Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Models trained: {len(self.models)}")
        print(f"Best individual model accuracy: {max(r['accuracy'] for r in results.values()):.4f}")
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        print("="*60)
        
        return results, ensemble_accuracy


# Main execution
if __name__ == "__main__":
    print("SOCCER PREDICTION MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer()
    results, ensemble_acc = trainer.run_full_training()
    
    print("\n✓ Ready for predictions!")