#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 02:22:16 2025

@author: aravkekane
"""

"""
Binary Model Training Pipeline
Trains models to predict Home Win (Y/N) - easier and better for betting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from config import STORAGE, MVP_CONFIG

class BinaryModelTrainer:
    """Trains binary classifiers for Home Win prediction"""
    
    def __init__(self):
        """Initialize model trainer"""
        self.processed_path = Path(STORAGE['processed_data_path'])
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        
        print("BinaryModelTrainer initialized")
    
    def load_dataset(self, dataset_path=None):
        """Load training dataset"""
        if dataset_path is None:
            # Find most recent enhanced dataset
            datasets = list(self.processed_path.glob('training_dataset_enhanced_*.csv'))
            if not datasets:
                raise FileNotFoundError("No training dataset found!")
            dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
        
        print(f"\nLoading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples")
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for BINARY classification (Home Win Y/N)
        
        Args:
            df: DataFrame with features
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nPreparing data for BINARY classification: Home Win (Y/N)")
        
        # Define feature columns
        exclude_cols = ['match_id', 'season', 'competition_code', 'home_team_id', 
                       'away_team_id', 'matchday', 'result', 'home_win', 'draw', 'away_win']
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Using {len(self.feature_columns)} features")
        
        X = df[self.feature_columns]
        y = df['home_win']  # Binary: 1=Home Win, 0=Not Home Win
        
        # Remove rows with missing targets
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"\nTarget distribution:")
        print(y.value_counts())
        print(f"Home Win Rate: {y.mean():.2%}")
        
        # Temporal split (train on older, test on recent)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\nTrain set: {len(X_train)} samples (Home Win: {y_train.mean():.2%})")
        print(f"Test set: {len(X_test)} samples (Home Win: {y_test.mean():.2%})")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest for binary classification"""
        print("\n" + "="*60)
        print("Training Random Forest (Binary)")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',  # Handle class imbalance
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
        """Train XGBoost for binary classification"""
        print("\n" + "="*60)
        print("Training XGBoost (Binary)")
        print("="*60)
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM for binary classification"""
        print("\n" + "="*60)
        print("Training LightGBM (Binary)")
        print("="*60)
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        return model
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models with binary classification metrics"""
        print("\n" + "="*60)
        print("BINARY MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name.upper()}")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of Home Win
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # ROC AUC (important for binary classification)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC AUC: {roc_auc:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Not Home Win', 'Home Win']))
            
            # Confusion matrix
            print("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
            print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        # Find best model by ROC AUC
        best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_model[0].upper()}")
        print(f"Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")
        print(f"ROC AUC: {best_model[1]['roc_auc']:.4f}")
        print("="*60)
        
        return results
    
    def create_weighted_ensemble(self, X_test, y_test, results):
        """
        Create WEIGHTED ensemble based on model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            results: Dictionary of model results
            
        Returns:
            Ensemble accuracy and ROC AUC
        """
        print("\n" + "="*60)
        print("WEIGHTED ENSEMBLE (Based on ROC AUC)")
        print("="*60)
        
        # Calculate weights based on ROC AUC scores
        total_roc_auc = sum(r['roc_auc'] for r in results.values())
        weights = {name: r['roc_auc'] / total_roc_auc for name, r in results.items()}
        
        print("\nModel Weights:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # Weighted average of probabilities
        ensemble_probs = np.zeros(len(y_test))
        for name, result in results.items():
            ensemble_probs += weights[name] * result['probabilities']
        
        # Get predictions (threshold at 0.5)
        ensemble_pred = (ensemble_probs >= 0.5).astype(int)
        
        # Evaluate
        accuracy = accuracy_score(y_test, ensemble_pred)
        roc_auc = roc_auc_score(y_test, ensemble_probs)
        
        print(f"\nWeighted Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Weighted Ensemble ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, ensemble_pred, target_names=['Not Home Win', 'Home Win']))
        
        cm = confusion_matrix(y_test, ensemble_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return accuracy, roc_auc, ensemble_probs, weights
    
    def save_models(self, weights=None):
        """Save trained models, scaler, and weights"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        models_dir = Path('models/binary')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = models_dir / f"{name}_binary_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {model_path}")
        
        # Save scaler
        scaler_path = models_dir / f"scaler_binary_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved: {scaler_path}")
        
        # Save feature columns
        features_path = models_dir / f"features_binary_{timestamp}.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        print(f"Saved: {features_path}")
        
        # Save ensemble weights if provided
        if weights:
            weights_path = models_dir / f"ensemble_weights_{timestamp}.json"
            with open(weights_path, 'w') as f:
                json.dump(weights, f)
            print(f"Saved: {weights_path}")
        
        print("\n✓ All binary models saved!")
    
    def run_full_training(self):
        """Run complete binary training pipeline"""
        print("\n" + "="*60)
        print("BINARY CLASSIFICATION TRAINING PIPELINE")
        print("="*60)
        
        # Load dataset
        df = self.load_dataset()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train models
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Create weighted ensemble
        accuracy, roc_auc, ensemble_probs, weights = self.create_weighted_ensemble(X_test, y_test, results)
        
        # Save models
        self.save_models(weights=weights)
        
        print("\n" + "="*60)
        print("BINARY TRAINING COMPLETE!")
        print("="*60)
        print(f"Models trained: {len(self.models)}")
        print(f"Best individual ROC AUC: {max(r['roc_auc'] for r in results.values()):.4f}")
        print(f"Best individual accuracy: {max(r['accuracy'] for r in results.values()):.4f}")
        print(f"Weighted ensemble accuracy: {accuracy:.4f}")
        print(f"Weighted ensemble ROC AUC: {roc_auc:.4f}")
        print("="*60)
        
        return results, accuracy, roc_auc


# Main execution
if __name__ == "__main__":
    print("BINARY HOME WIN PREDICTION MODEL")
    print("="*60)
    
    trainer = BinaryModelTrainer()
    results, acc, roc = trainer.run_full_training()
    
    print("\n✓ Ready for betting predictions!")
    
    
    
    