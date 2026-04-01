#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 22:00:11 2025

@author: aravkekane
"""

"""
Match Predictor V3 - FINAL VERSION
Integrated with:
- Kelly Criterion for bet sizing
- Monte Carlo simulation for risk assessment
- 3-way probability estimation
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

from config import STORAGE, LEAGUES, API_CONFIG
from api_client import FootballDataClient
from advanced_betting import KellyCriterion, MonteCarloSimulator

import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from feature_engineering import FeatureEngineerV2
except ImportError:
    exec(open('feature_engineering_FIXED.py').read())
    from feature_engineering import FeatureEngineerV2


class MatchPredictorV3:
    """Final predictor with advanced betting features"""
    
    def __init__(self, bankroll=1000, max_bet_pct=0.05):
        """
        Args:
            bankroll: Your total bankroll
            max_bet_pct: Maximum % to risk per bet (5% default)
        """
        self.db_path = Path(STORAGE['database_path'])
        self.api_client = FootballDataClient()
        self.feature_engineer = FeatureEngineerV2()
        
        # Betting modules
        self.kelly = KellyCriterion(bankroll=bankroll, max_bet_pct=max_bet_pct)
        self.monte_carlo = MonteCarloSimulator(n_simulations=10000)
        
        self.models_dir = Path('models/league_specific_optimized')
        if not self.models_dir.exists():
            self.models_dir = Path('models/league_specific')
        
        print("MatchPredictorV3 initialized")
        print(f"Bankroll: ${bankroll}")
        print(f"Max bet per match: {max_bet_pct*100}% (${bankroll * max_bet_pct})")
    
    def load_league_model(self, league_code):
        """Load model"""
        model_files = list(self.models_dir.glob(f"{league_code}_*_optimized_*.pkl"))
        model_files = [f for f in model_files if 'scaler' not in f.name and 'features' not in f.name]
        
        if not model_files:
            model_files = list(self.models_dir.glob(f"{league_code}_*.pkl"))
            model_files = [f for f in model_files if 'scaler' not in f.name and 'features' not in f.name]
        
        if not model_files:
            raise FileNotFoundError(f"No model found for {league_code}")
        
        model_file = max(model_files, key=lambda p: p.stat().st_mtime)
        model_type = 'random_forest' if 'random_forest' in model_file.name else 'xgboost'
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        scaler_file = model_file.parent / model_file.name.replace(model_type, 'scaler')
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        
        features_file = model_file.parent / model_file.name.replace(model_type, 'features').replace('.pkl', '.json')
        if features_file.exists():
            with open(features_file, 'r') as f:
                feature_names = json.load(f)
        else:
            feature_names = None
        
        return model, scaler, feature_names
    
    def get_upcoming_matches(self, league_code, days_ahead=10):
        """Get upcoming matches"""
        try:
            response = self.api_client.get_matches(league_code, status='SCHEDULED')
            matches = response.get('matches', [])
            
            if not matches:
                return []
            
            now = datetime.now()
            future_date = now + timedelta(days=days_ahead)
            
            upcoming = []
            for match in matches:
                match_date = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00')).replace(tzinfo=None)
                if now <= match_date <= future_date:
                    upcoming.append(match)
            
            return upcoming
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def generate_match_features(self, match, league_code):
        """Generate features"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM matches 
            WHERE competition_code = ? AND status = 'FINISHED'
            ORDER BY utc_date
        """
        df = pd.read_sql_query(query, conn, params=[league_code])
        conn.close()
        
        df['utc_date'] = pd.to_datetime(df['utc_date']).dt.tz_localize(None)
        df = self.feature_engineer.create_basic_features(df)
        standings_df = self.feature_engineer.load_standings(competition_code=league_code)
        
        home_team_id = match['homeTeam']['id']
        away_team_id = match['awayTeam']['id']
        match_date = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00')).replace(tzinfo=None)
        season = match['season']['id']
        matchday = match['matchday'] if match.get('matchday') else 20
        
        match_features = {}
        
        home_form = self.feature_engineer.get_team_form_at_date_v2(df, home_team_id, match_date, is_home=True, n_matches=10)
        if home_form:
            for key, value in home_form.items():
                match_features[f'home_{key}'] = value
        
        away_form = self.feature_engineer.get_team_form_at_date_v2(df, away_team_id, match_date, is_home=False, n_matches=10)
        if away_form:
            for key, value in away_form.items():
                match_features[f'away_{key}'] = value
        
        h2h_features = self.feature_engineer.get_h2h_features(df, home_team_id, away_team_id, match_date)
        match_features.update(h2h_features)
        
        position_features = self.feature_engineer.get_league_position_features(standings_df, season, league_code, home_team_id, away_team_id, matchday)
        match_features.update(position_features)
        
        home_rest = self.feature_engineer.get_rest_days_features(df, home_team_id, match_date)
        away_rest = self.feature_engineer.get_rest_days_features(df, away_team_id, match_date)
        match_features['home_rest_days'] = home_rest['rest_days']
        match_features['away_rest_days'] = away_rest['rest_days']
        match_features['rest_days_diff'] = home_rest['rest_days'] - away_rest['rest_days']
        
        season_features = self.feature_engineer.get_season_stage_features(matchday)
        match_features.update(season_features)
        
        home_opp_strength = self.feature_engineer.get_recent_opponent_strength(df, standings_df, home_team_id, match_date, n_matches=5)
        away_opp_strength = self.feature_engineer.get_recent_opponent_strength(df, standings_df, away_team_id, match_date, n_matches=5)
        match_features['home_opponent_strength'] = home_opp_strength
        match_features['away_opponent_strength'] = away_opp_strength
        match_features['opponent_strength_diff'] = home_opp_strength - away_opp_strength
        
        if home_form and away_form:
            match_features['attack_vs_defense'] = home_form['goals_scored_per_match'] - away_form['goals_conceded_per_match']
            match_features['defense_vs_attack'] = away_form['goals_scored_per_match'] - home_form['goals_conceded_per_match']
            match_features['form_diff'] = home_form['points_per_match'] - away_form['points_per_match']
            match_features['momentum_diff'] = home_form['form_trend'] - away_form['form_trend']
        
        # Return home/away form for Monte Carlo
        return match_features, home_form, away_form
    
    def estimate_draw_probability(self, home_win_prob, league_code):
        """Estimate 3-way probabilities"""
        draw_rates = {
            'PL': 0.26, 'PD': 0.24, 'BL1': 0.23,
            'SA': 0.27, 'FL1': 0.25, 'CL': 0.22
        }
        base_draw_rate = draw_rates.get(league_code, 0.25)
        not_home_win_prob = 1 - home_win_prob
        closeness = 1 - abs(home_win_prob - 0.5) * 2
        draw_prob = base_draw_rate * (0.7 + 0.6 * closeness)
        draw_prob = min(draw_prob, not_home_win_prob * 0.8)
        away_win_prob = not_home_win_prob - draw_prob
        
        total = home_win_prob + draw_prob + away_win_prob
        return home_win_prob / total, draw_prob / total, away_win_prob / total
    
    def predict_match(self, match, league_code, bookmaker_odds=None):
        """
        Predict match with Kelly Criterion and Monte Carlo
        
        Args:
            match: Match data
            league_code: League code
            bookmaker_odds: Dict with {'home': 2.0, 'draw': 3.5, 'away': 3.0} (optional)
        """
        model, scaler, feature_names = self.load_league_model(league_code)
        
        features_dict, home_form, away_form = self.generate_match_features(match, league_code)
        features_df = pd.DataFrame([features_dict]).fillna(0)
        
        if feature_names:
            for col in feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[feature_names]
        
        if scaler:
            features_scaled = scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        probabilities = model.predict_proba(features_scaled)[0]
        home_win_prob_binary = probabilities[1]
        
        # 3-way probabilities
        home_prob, draw_prob, away_prob = self.estimate_draw_probability(home_win_prob_binary, league_code)
        
        # Monte Carlo simulation for additional insights
        mc_result = None
        if home_form and away_form:
            try:
                mc_result = self.monte_carlo.simulate_match_poisson(
                    home_attack=home_form.get('goals_scored_per_match', 1.5),
                    home_defense=home_form.get('goals_conceded_per_match', 1.2),
                    away_attack=away_form.get('goals_scored_per_match', 1.2),
                    away_defense=away_form.get('goals_conceded_per_match', 1.5),
                    home_advantage=1.3
                )
            except:
                pass
        
        # Kelly Criterion if odds provided
        kelly_result = None
        if bookmaker_odds:
            # Determine best bet
            outcomes = {
                'home': (home_prob, bookmaker_odds.get('home', 2.0)),
                'draw': (draw_prob, bookmaker_odds.get('draw', 3.5)),
                'away': (away_prob, bookmaker_odds.get('away', 3.0))
            }
            
            best_outcome = None
            best_kelly = None
            max_ev = -999
            
            for outcome_type, (prob, odds) in outcomes.items():
                kelly = self.kelly.calculate_kelly(prob, odds)
                if kelly['expected_value'] > max_ev:
                    max_ev = kelly['expected_value']
                    best_kelly = kelly
                    best_outcome = outcome_type
            
            if max_ev > 0:  # Positive EV
                kelly_result = {
                    'bet_on': best_outcome.upper(),
                    'bet_amount': best_kelly['bet_amount'],
                    'kelly_pct': best_kelly['kelly_percentage'],
                    'expected_value': best_kelly['expected_value'],
                    'recommendation': best_kelly['recommendation']
                }
        
        return {
            'home_team': match['homeTeam']['name'],
            'away_team': match['awayTeam']['name'],
            'date': match['utcDate'],
            'home_win_prob': home_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_prob,
            'monte_carlo': mc_result,
            'kelly': kelly_result
        }
    
    def predict_league(self, league_code, league_name, bookmaker_odds_dict=None):
        """
        Predict league with optional bookmaker odds
        
        Args:
            bookmaker_odds_dict: Dict mapping match to odds
                                 e.g., {'Team A vs Team B': {'home': 2.0, 'draw': 3.5, 'away': 3.0}}
        """
        print("\n" + "="*80)
        print(f"📊 {league_name.upper()}")
        print("="*80)
        
        upcoming = self.get_upcoming_matches(league_code, days_ahead=10)
        
        if not upcoming:
            print("No upcoming matches")
            return []
        
        print(f"Found {len(upcoming)} matches\n")
        
        predictions = []
        
        for i, match in enumerate(upcoming, 1):
            try:
                match_name = f"{match['homeTeam']['name']} vs {match['awayTeam']['name']}"
                odds = bookmaker_odds_dict.get(match_name) if bookmaker_odds_dict else None
                
                prediction = self.predict_match(match, league_code, odds)
                predictions.append(prediction)
                
                # Display
                print(f"Match {i}: {prediction['home_team']} vs {prediction['away_team']}")
                print(f"  📅 {prediction['date'][:10]} {prediction['date'][11:16]}")
                print(f"  📊 Probabilities:")
                print(f"     Home: {prediction['home_win_prob']:.1%} | Draw: {prediction['draw_prob']:.1%} | Away: {prediction['away_win_prob']:.1%}")
                
                if prediction['monte_carlo']:
                    mc = prediction['monte_carlo']
                    print(f"  🎲 Monte Carlo (10k sims):")
                    print(f"     Most likely score: {mc['most_likely_score'][0]}-{mc['most_likely_score'][1]}")
                    print(f"     Over 2.5: {mc['over_2_5_prob']:.1%} | BTTS: {mc['btts_prob']:.1%}")
                
                if prediction['kelly']:
                    k = prediction['kelly']
                    print(f"  💰 Kelly Criterion:")
                    print(f"     BET {k['bet_on']} - ${k['bet_amount']} ({k['kelly_pct']}%)")
                    print(f"     EV: {k['expected_value']:.2%} | {k['recommendation']}")
                else:
                    print(f"  💡 No odds provided - add bookmaker odds for bet sizing")
                
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
        
        return predictions
    
    def predict_all_leagues(self, bookmaker_odds_all=None):
        """Predict all leagues"""
        print("="*80)
        print("⚽ SOCCER MATCH PREDICTOR V3 - WITH KELLY CRITERION + MONTE CARLO")
        print("="*80)
        print(f"Bankroll: ${self.kelly.bankroll}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        all_predictions = {}
        all_bets = []
        
        for league_key, league_info in LEAGUES.items():
            league_code = league_info['id']
            league_name = league_info['name']
            
            odds_dict = bookmaker_odds_all.get(league_code) if bookmaker_odds_all else None
            predictions = self.predict_league(league_code, league_name, odds_dict)
            all_predictions[league_code] = predictions
            
            # Collect bets
            for pred in predictions:
                if pred.get('kelly'):
                    all_bets.append({
                        'match': f"{pred['home_team']} vs {pred['away_team']}",
                        'bet_on': pred['kelly']['bet_on'],
                        'amount': pred['kelly']['bet_amount'],
                        'ev': pred['kelly']['expected_value']
                    })
        
        # Summary
        print("\n" + "="*80)
        print("💰 BETTING SUMMARY")
        print("="*80)
        
        if all_bets:
            total_staked = sum(b['amount'] for b in all_bets)
            print(f"\nTotal bets: {len(all_bets)}")
            print(f"Total staked: ${total_staked:.2f} ({total_staked/self.kelly.bankroll*100:.1f}% of bankroll)")
            print(f"\nRecommended bets:")
            for bet in all_bets:
                print(f"  ✅ {bet['match']}")
                print(f"     Bet {bet['bet_on']} - ${bet['amount']:.2f} (EV: {bet['ev']:.2%})")
        else:
            print("\n⚠️  No bets recommended")
            print("Add bookmaker odds to enable Kelly Criterion bet sizing")
        
        print("="*80)
        
        return all_predictions, all_bets


if __name__ == "__main__":
    # Example with NO odds (will show probabilities only)
    predictor = MatchPredictorV3(bankroll=1000, max_bet_pct=0.05)
    predictions, bets = predictor.predict_all_leagues()
    
    print("\n📝 To enable Kelly Criterion bet sizing:")
    print("Add bookmaker odds like this:")
    print("""
    odds = {
        'PL': {
            'Liverpool FC vs Nottingham Forest FC': {'home': 1.5, 'draw': 4.0, 'away': 7.0}
        }
    }
    predictor.predict_all_leagues(bookmaker_odds_all=odds)
    """)