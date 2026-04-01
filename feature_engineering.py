#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering - FIXED VERSION
Fixes timezone comparison bug
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from config import STORAGE, MVP_CONFIG

class FeatureEngineerV2:
    """Enhanced feature engineer with timezone fix"""
    
    def __init__(self):
        self.db_path = Path(STORAGE['database_path'])
        self.processed_path = Path(STORAGE['processed_data_path'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        print("FeatureEngineerV2 initialized - Enhanced with 5 new features!")
    
    def load_matches(self, competition_code=None, status='FINISHED'):
        """Load matches from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                match_id, season, competition_code, competition_name, matchday, status,
                utc_date, home_team_id, home_team_name, away_team_id, away_team_name,
                home_score, away_score, winner, duration
            FROM matches WHERE status = ?
        """
        
        params = [status]
        if competition_code:
            query += " AND competition_code = ?"
            params.append(competition_code)
        query += " ORDER BY utc_date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert to timezone-naive
        df['utc_date'] = pd.to_datetime(df['utc_date']).dt.tz_localize(None)
        
        print(f"Loaded {len(df)} matches")
        return df
    
    def load_standings(self, competition_code=None):
        """Load standings data"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM standings"
        params = []
        
        if competition_code:
            query += " WHERE competition_code = ?"
            params.append(competition_code)
        
        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        
        # Fix timezone if present
        if 'snapshot_date' in df.columns:
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date']).dt.tz_localize(None)
        
        return df
    
    def create_basic_features(self, df):
        """Create basic match outcome features"""
        df['total_goals'] = df['home_score'] + df['away_score']
        
        df['result'] = 'D'
        df.loc[df['winner'] == 'HOME_TEAM', 'result'] = 'H'
        df.loc[df['winner'] == 'AWAY_TEAM', 'result'] = 'A'
        
        df['home_win'] = (df['result'] == 'H').astype(int)
        df['draw'] = (df['result'] == 'D').astype(int)
        df['away_win'] = (df['result'] == 'A').astype(int)
        df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
        df['btts'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)
        df['home_clean_sheet'] = (df['away_score'] == 0).astype(int)
        df['away_clean_sheet'] = (df['home_score'] == 0).astype(int)
        
        return df
    
    def get_team_form_at_date_v2(self, df, team_id, date, is_home, n_matches=10):
        """Get team form with ENHANCED features"""
        team_matches = df[
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['utc_date'] < date)
        ].tail(n_matches).copy()
        
        if len(team_matches) == 0:
            return None
        
        points = []
        goals_for = []
        goals_against = []
        wins = []
        draws = []
        losses = []
        clean_sheets = []
        btts = []
        match_dates = []
        
        home_only_points = []
        home_only_gf = []
        home_only_ga = []
        away_only_points = []
        away_only_gf = []
        away_only_ga = []
        
        weights = np.exp(np.linspace(-2, 0, len(team_matches)))
        weights = weights / weights.sum()
        
        for idx, match in team_matches.iterrows():
            match_dates.append(match['utc_date'])
            
            if match['home_team_id'] == team_id:
                gf = match['home_score']
                ga = match['away_score']
                if match['result'] == 'H':
                    pts = 3
                    wins.append(1)
                    draws.append(0)
                    losses.append(0)
                elif match['result'] == 'D':
                    pts = 1
                    wins.append(0)
                    draws.append(1)
                    losses.append(0)
                else:
                    pts = 0
                    wins.append(0)
                    draws.append(0)
                    losses.append(1)
                clean_sheets.append(1 if ga == 0 else 0)
                home_only_points.append(pts)
                home_only_gf.append(gf)
                home_only_ga.append(ga)
            else:
                gf = match['away_score']
                ga = match['home_score']
                if match['result'] == 'A':
                    pts = 3
                    wins.append(1)
                    draws.append(0)
                    losses.append(0)
                elif match['result'] == 'D':
                    pts = 1
                    wins.append(0)
                    draws.append(1)
                    losses.append(0)
                else:
                    pts = 0
                    wins.append(0)
                    draws.append(0)
                    losses.append(1)
                clean_sheets.append(1 if ga == 0 else 0)
                away_only_points.append(pts)
                away_only_gf.append(gf)
                away_only_ga.append(ga)
            
            points.append(pts)
            goals_for.append(gf)
            goals_against.append(ga)
            btts.append(1 if (gf > 0 and ga > 0) else 0)
        
        features = {
            'matches_played': len(team_matches),
            'points_per_match': np.average(points, weights=weights),
            'goals_scored_per_match': np.average(goals_for, weights=weights),
            'goals_conceded_per_match': np.average(goals_against, weights=weights),
            'goal_diff_per_match': np.average(np.array(goals_for) - np.array(goals_against), weights=weights),
            'win_rate': np.average(wins, weights=weights),
            'draw_rate': np.average(draws, weights=weights),
            'loss_rate': np.average(losses, weights=weights),
            'clean_sheet_rate': np.average(clean_sheets, weights=weights),
            'btts_rate': np.average(btts, weights=weights),
            'form_trend': self._calculate_trend(points),
            'scoring_trend': self._calculate_trend(goals_for),
            'recent_3_points': sum(points[-3:]) if len(points) >= 3 else sum(points),
        }
        
        # NEW FEATURE 1: Home/Away Split
        if is_home and len(home_only_points) >= 3:
            features['home_form_ppg'] = np.mean(home_only_points[-5:])
            features['home_form_gf'] = np.mean(home_only_gf[-5:])
            features['home_form_ga'] = np.mean(home_only_ga[-5:])
            features['home_games_played'] = len(home_only_points)
        else:
            features['home_form_ppg'] = 0
            features['home_form_gf'] = 0
            features['home_form_ga'] = 0
            features['home_games_played'] = 0
        
        if not is_home and len(away_only_points) >= 3:
            features['away_form_ppg'] = np.mean(away_only_points[-5:])
            features['away_form_gf'] = np.mean(away_only_gf[-5:])
            features['away_form_ga'] = np.mean(away_only_ga[-5:])
            features['away_games_played'] = len(away_only_points)
        else:
            features['away_form_ppg'] = 0
            features['away_form_gf'] = 0
            features['away_form_ga'] = 0
            features['away_games_played'] = 0
        
        # NEW FEATURE 2: Scoring Momentum
        if len(goals_for) >= 6:
            recent_3_goals = np.mean(goals_for[-3:])
            previous_3_goals = np.mean(goals_for[-6:-3])
            features['scoring_momentum'] = recent_3_goals - previous_3_goals
        else:
            features['scoring_momentum'] = 0
        
        # NEW FEATURE 3: Defensive Momentum
        if len(goals_against) >= 6:
            recent_3_conceded = np.mean(goals_against[-3:])
            previous_3_conceded = np.mean(goals_against[-6:-3])
            features['defensive_momentum'] = previous_3_conceded - recent_3_conceded
        else:
            features['defensive_momentum'] = 0
        
        # NEW FEATURE 4: Fixture Congestion
        if len(match_dates) >= 2:
            days_between = [(match_dates[i] - match_dates[i-1]).days for i in range(1, len(match_dates))]
            features['avg_rest_days'] = np.mean(days_between[-5:]) if len(days_between) >= 5 else np.mean(days_between)
            recent_14_days = [m for m in match_dates if (date - m).days <= 14]
            features['games_last_14_days'] = len(recent_14_days)
            recent_7_days = [m for m in match_dates if (date - m).days <= 7]
            features['games_last_7_days'] = len(recent_7_days)
        else:
            features['avg_rest_days'] = 7
            features['games_last_14_days'] = 0
            features['games_last_7_days'] = 0
        
        # Original home/away advantage
        if is_home:
            home_matches = team_matches[team_matches['home_team_id'] == team_id]
            if len(home_matches) >= 3:
                home_results = []
                for _, m in home_matches.iterrows():
                    if m['result'] == 'H':
                        home_results.append(3)
                    elif m['result'] == 'D':
                        home_results.append(1)
                    else:
                        home_results.append(0)
                features['home_advantage'] = np.mean(home_results) / 3
        else:
            away_matches = team_matches[team_matches['away_team_id'] == team_id]
            if len(away_matches) >= 3:
                away_results = []
                for _, m in away_matches.iterrows():
                    if m['result'] == 'A':
                        away_results.append(3)
                    elif m['result'] == 'D':
                        away_results.append(1)
                    else:
                        away_results.append(0)
                features['away_performance'] = np.mean(away_results) / 3
        
        return features
    
    def _calculate_trend(self, values):
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def get_recent_opponent_strength(self, df, standings_df, team_id, date, n_matches=5):
        """NEW FEATURE 5: Strength of schedule"""
        team_matches = df[
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['utc_date'] < date)
        ].tail(n_matches)
        
        if len(team_matches) == 0:
            return 10
        
        opponent_positions = []
        
        for _, match in team_matches.iterrows():
            if match['home_team_id'] == team_id:
                opponent_id = match['away_team_id']
            else:
                opponent_id = match['home_team_id']
            
            season = match['season']
            competition = match['competition_code']
            
            opponent_standings = standings_df[
                (standings_df['season'] == season) &
                (standings_df['competition_code'] == competition) &
                (standings_df['team_id'] == opponent_id)
            ].head(1)
            
            if len(opponent_standings) > 0:
                opponent_positions.append(opponent_standings.iloc[0]['position'])
            else:
                opponent_positions.append(10)
        
        return np.mean(opponent_positions) if opponent_positions else 10
    
    def get_h2h_features(self, df, home_team_id, away_team_id, date, n_matches=10):
        """Head-to-head features"""
        h2h_matches = df[
            (((df['home_team_id'] == home_team_id) & (df['away_team_id'] == away_team_id)) |
             ((df['home_team_id'] == away_team_id) & (df['away_team_id'] == home_team_id))) &
            (df['utc_date'] < date)
        ].tail(n_matches)
        
        if len(h2h_matches) == 0:
            return {
                'h2h_matches': 0,
                'h2h_home_win_rate': 0,
                'h2h_draw_rate': 0,
                'h2h_away_win_rate': 0,
                'h2h_avg_goals': 0,
                'h2h_btts_rate': 0
            }
        
        weights = np.exp(np.linspace(-1, 0, len(h2h_matches)))
        weights = weights / weights.sum()
        
        home_wins = []
        draws = []
        away_wins = []
        total_goals = []
        btts = []
        
        for _, match in h2h_matches.iterrows():
            if match['home_team_id'] == home_team_id:
                home_wins.append(1 if match['result'] == 'H' else 0)
                away_wins.append(1 if match['result'] == 'A' else 0)
            else:
                home_wins.append(1 if match['result'] == 'A' else 0)
                away_wins.append(1 if match['result'] == 'H' else 0)
            
            draws.append(1 if match['result'] == 'D' else 0)
            total_goals.append(match['total_goals'])
            btts.append(match['btts'])
        
        return {
            'h2h_matches': len(h2h_matches),
            'h2h_home_win_rate': np.average(home_wins, weights=weights),
            'h2h_draw_rate': np.average(draws, weights=weights),
            'h2h_away_win_rate': np.average(away_wins, weights=weights),
            'h2h_avg_goals': np.average(total_goals, weights=weights),
            'h2h_btts_rate': np.average(btts, weights=weights)
        }
    
    def get_league_position_features(self, standings_df, season, competition_code, 
                                     home_team_id, away_team_id, matchday):
        """League position features"""
        home_standings = standings_df[
            (standings_df['season'] == season) &
            (standings_df['competition_code'] == competition_code) &
            (standings_df['team_id'] == home_team_id)
        ].head(1)
        
        away_standings = standings_df[
            (standings_df['season'] == season) &
            (standings_df['competition_code'] == competition_code) &
            (standings_df['team_id'] == away_team_id)
        ].head(1)
        
        features = {
            'home_position': home_standings.iloc[0]['position'] if len(home_standings) > 0 else 10,
            'away_position': away_standings.iloc[0]['position'] if len(away_standings) > 0 else 10,
            'home_points': home_standings.iloc[0]['points'] if len(home_standings) > 0 else 0,
            'away_points': away_standings.iloc[0]['points'] if len(away_standings) > 0 else 0
        }
        
        features['position_diff'] = features['home_position'] - features['away_position']
        features['points_diff'] = features['home_points'] - features['away_points']
        
        return features
    
    def get_rest_days_features(self, df, team_id, date):
        """Rest days features"""
        team_matches = df[
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['utc_date'] < date)
        ].tail(1)
        
        if len(team_matches) == 0:
            return {'rest_days': 7}
        
        last_match_date = team_matches.iloc[-1]['utc_date']
        rest_days = (date - last_match_date).days
        
        return {'rest_days': rest_days}
    
    def get_season_stage_features(self, matchday):
        """Season stage features"""
        max_matchday = 38
        progress = matchday / max_matchday
        
        return {
            'season_progress': progress,
            'early_season': 1 if matchday <= 10 else 0,
            'mid_season': 1 if 10 < matchday <= 28 else 0,
            'late_season': 1 if matchday > 28 else 0
        }
    
    def build_dataset_v2(self, df, standings_df, competition_code=None):
        """Build complete dataset with enhanced features"""
        print("\n" + "="*60)
        print("BUILDING ENHANCED DATASET WITH 5 NEW FEATURES")
        print("="*60)
        
        if competition_code:
            df = df[df['competition_code'] == competition_code].copy()
        
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
            
            historical_count = len(df[
                (df['utc_date'] < match_date) &
                (df['season'] == season) &
                (df['competition_code'] == competition)
            ])
            
            if historical_count < 10:
                continue
            
            match_features = {
                'match_id': match['match_id'],
                'season': season,
                'competition_code': competition,
                'home_team_id': home_team,
                'away_team_id': away_team,
                'matchday': matchday,
                'result': match['result'],
                'home_win': match['home_win'],
                'draw': match['draw'],
                'away_win': match['away_win']
            }
            
            home_form = self.get_team_form_at_date_v2(df, home_team, match_date, is_home=True, n_matches=10)
            if home_form:
                for key, value in home_form.items():
                    match_features[f'home_{key}'] = value
            
            away_form = self.get_team_form_at_date_v2(df, away_team, match_date, is_home=False, n_matches=10)
            if away_form:
                for key, value in away_form.items():
                    match_features[f'away_{key}'] = value
            
            h2h_features = self.get_h2h_features(df, home_team, away_team, match_date)
            match_features.update(h2h_features)
            
            position_features = self.get_league_position_features(
                standings_df, season, competition, home_team, away_team, matchday
            )
            match_features.update(position_features)
            
            home_rest = self.get_rest_days_features(df, home_team, match_date)
            away_rest = self.get_rest_days_features(df, away_team, match_date)
            match_features['home_rest_days'] = home_rest['rest_days']
            match_features['away_rest_days'] = away_rest['rest_days']
            match_features['rest_days_diff'] = home_rest['rest_days'] - away_rest['rest_days']
            
            season_features = self.get_season_stage_features(matchday)
            match_features.update(season_features)
            
            home_opponent_strength = self.get_recent_opponent_strength(
                df, standings_df, home_team, match_date, n_matches=5
            )
            away_opponent_strength = self.get_recent_opponent_strength(
                df, standings_df, away_team, match_date, n_matches=5
            )
            match_features['home_opponent_strength'] = home_opponent_strength
            match_features['away_opponent_strength'] = away_opponent_strength
            match_features['opponent_strength_diff'] = home_opponent_strength - away_opponent_strength
            
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
        
        print(f"\n✓ Dataset built: {len(dataset)} samples, {len(dataset.columns)} features")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'training_dataset_v2_enhanced_{timestamp}.csv'
        if competition_code:
            filename = f'training_dataset_v2_enhanced_{competition_code}_{timestamp}.csv'
        
        output_path = self.processed_path / filename
        dataset.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
        
        return dataset


if __name__ == "__main__":
    print("="*60)
    print("FEATURE ENGINEERING V2 - ENHANCED")
    print("="*60)
    
    engineer = FeatureEngineerV2()
    
    print("\nLoading data...")
    df = engineer.load_matches()
    df = engineer.create_basic_features(df)
    standings_df = engineer.load_standings()
    
    print("\nBuilding enhanced dataset...")
    dataset = engineer.build_dataset_v2(df, standings_df)
    
    print("\n✓ COMPLETE!")
    print(f"Total samples: {len(dataset)}")
    print(f"Total features: {len(dataset.columns)}")