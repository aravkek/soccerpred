#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 01:52:59 2025

@author: aravkekane
"""

"""
Data Collection Script
Pulls historical and current season data and stores in database
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from api_client import FootballDataClient
from config import MVP_CONFIG, STORAGE, SEASONS
import time

class DataCollector:
    """Collects and stores football data in SQLite database"""
    
    def __init__(self):
        """Initialize data collector"""
        self.client = FootballDataClient()
        self.db_path = Path(STORAGE['database_path'])
        self.raw_data_path = Path(STORAGE['raw_data_path'])
        
        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"DataCollector initialized")
        print(f"Database: {self.db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                season INTEGER,
                competition_code TEXT,
                competition_name TEXT,
                matchday INTEGER,
                status TEXT,
                utc_date TEXT,
                home_team_id INTEGER,
                home_team_name TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                home_score INTEGER,
                away_score INTEGER,
                winner TEXT,
                duration TEXT,
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT,
                short_name TEXT,
                tla TEXT,
                crest TEXT,
                founded INTEGER,
                venue TEXT,
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Standings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS standings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER,
                competition_code TEXT,
                team_id INTEGER,
                team_name TEXT,
                position INTEGER,
                played_games INTEGER,
                won INTEGER,
                draw INTEGER,
                lost INTEGER,
                points INTEGER,
                goals_for INTEGER,
                goals_against INTEGER,
                goal_difference INTEGER,
                form TEXT,
                snapshot_date TEXT,
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(season, competition_code, team_id, snapshot_date)
            )
        ''')
        
        # Data collection log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER,
                competition_code TEXT,
                data_type TEXT,
                records_collected INTEGER,
                collection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("Database tables initialized")
    
    def collect_historical_matches(self, competition_code: str, seasons: list):
        """
        Collect historical match data for multiple seasons
        
        Args:
            competition_code: Competition code (e.g., 'PL')
            seasons: List of season years
        """
        print(f"\n=== Collecting Historical Matches for {competition_code} ===")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for season in seasons:
            print(f"\nCollecting season {season}...")
            
            try:
                # Get matches for season
                matches_data = self.client.get_matches(competition_code, season=season)
                matches = matches_data.get('matches', [])
                
                print(f"Found {len(matches)} matches")
                
                # Save raw JSON
                raw_file = self.raw_data_path / f"{competition_code}_{season}_matches.json"
                with open(raw_file, 'w') as f:
                    json.dump(matches_data, f, indent=2)
                
                # Insert matches into database
                for match in matches:
                    cursor.execute('''
                        INSERT OR REPLACE INTO matches 
                        (match_id, season, competition_code, competition_name, matchday, status, 
                         utc_date, home_team_id, home_team_name, away_team_id, away_team_name,
                         home_score, away_score, winner, duration, raw_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        match['id'],
                        season,
                        competition_code,
                        match['competition']['name'],
                        match.get('matchday'),
                        match['status'],
                        match['utcDate'],
                        match['homeTeam']['id'],
                        match['homeTeam']['name'],
                        match['awayTeam']['id'],
                        match['awayTeam']['name'],
                        match['score']['fullTime']['home'],
                        match['score']['fullTime']['away'],
                        match['score'].get('winner'),
                        match['score'].get('duration'),
                        json.dumps(match)
                    ))
                
                # Log collection
                cursor.execute('''
                    INSERT INTO collection_log 
                    (season, competition_code, data_type, records_collected, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (season, competition_code, 'matches', len(matches), 'success', f'Collected {len(matches)} matches'))
                
                conn.commit()
                print(f"✓ Season {season} saved to database")
                
                # Rate limiting - wait between requests
                time.sleep(6)  # 10 calls/min = 6 seconds between calls
                
            except Exception as e:
                print(f"✗ Error collecting season {season}: {str(e)}")
                cursor.execute('''
                    INSERT INTO collection_log 
                    (season, competition_code, data_type, records_collected, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (season, competition_code, 'matches', 0, 'error', str(e)))
                conn.commit()
        
        conn.close()
        print(f"\n✓ Historical data collection complete")
    
    def collect_teams(self, competition_code: str):
        """
        Collect team data
        
        Args:
            competition_code: Competition code (e.g., 'PL')
        """
        print(f"\n=== Collecting Teams for {competition_code} ===")
        
        try:
            teams_data = self.client.get_teams(competition_code)
            teams = teams_data.get('teams', [])
            
            print(f"Found {len(teams)} teams")
            
            # Save raw JSON
            raw_file = self.raw_data_path / f"{competition_code}_teams.json"
            with open(raw_file, 'w') as f:
                json.dump(teams_data, f, indent=2)
            
            # Insert teams into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for team in teams:
                cursor.execute('''
                    INSERT OR REPLACE INTO teams 
                    (team_id, name, short_name, tla, crest, founded, venue, raw_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    team['id'],
                    team['name'],
                    team.get('shortName'),
                    team.get('tla'),
                    team.get('crest'),
                    team.get('founded'),
                    team.get('venue'),
                    json.dumps(team)
                ))
            
            conn.commit()
            conn.close()
            
            print(f"✓ Teams saved to database")
            
        except Exception as e:
            print(f"✗ Error collecting teams: {str(e)}")
    
    def collect_standings(self, competition_code: str, seasons: list):
        """
        Collect standings data for multiple seasons
        
        Args:
            competition_code: Competition code (e.g., 'PL')
            seasons: List of season years
        """
        print(f"\n=== Collecting Standings for {competition_code} ===")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for season in seasons:
            print(f"\nCollecting standings for season {season}...")
            
            try:
                standings_data = self.client.get_standings(competition_code, season=season)
                
                if 'standings' in standings_data and standings_data['standings']:
                    table = standings_data['standings'][0]['table']
                    
                    print(f"Found {len(table)} teams in standings")
                    
                    # Save raw JSON
                    raw_file = self.raw_data_path / f"{competition_code}_{season}_standings.json"
                    with open(raw_file, 'w') as f:
                        json.dump(standings_data, f, indent=2)
                    
                    # Insert standings
                    snapshot_date = datetime.now().strftime('%Y-%m-%d')
                    
                    for entry in table:
                        cursor.execute('''
                            INSERT OR REPLACE INTO standings
                            (season, competition_code, team_id, team_name, position, 
                             played_games, won, draw, lost, points, goals_for, goals_against,
                             goal_difference, form, snapshot_date, raw_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            season,
                            competition_code,
                            entry['team']['id'],
                            entry['team']['name'],
                            entry['position'],
                            entry['playedGames'],
                            entry['won'],
                            entry['draw'],
                            entry['lost'],
                            entry['points'],
                            entry['goalsFor'],
                            entry['goalsAgainst'],
                            entry['goalDifference'],
                            entry.get('form'),
                            snapshot_date,
                            json.dumps(entry)
                        ))
                    
                    conn.commit()
                    print(f"✓ Standings for season {season} saved")
                
                # Rate limiting
                time.sleep(6)
                
            except Exception as e:
                print(f"✗ Error collecting standings for season {season}: {str(e)}")
        
        conn.close()
    
    def run_full_collection(self):
        """Run full data collection for ALL leagues"""
        print("\n" + "="*60)
        print("STARTING FULL DATA COLLECTION - ALL LEAGUES")
        print("="*60)
        
        from config import LEAGUES, SEASONS
        
        seasons = SEASONS['all_available']
        
        # Collect data for ALL leagues
        for league_key, league_info in LEAGUES.items():
            competition_code = league_info['id']
            league_name = league_info['name']
            
            print(f"\n{'='*60}")
            print(f"COLLECTING DATA FOR: {league_name}")
            print(f"{'='*60}")
            
            try:
                # Step 1: Collect teams
                self.collect_teams(competition_code)
                time.sleep(6)
                
                # Step 2: Collect historical matches
                self.collect_historical_matches(competition_code, seasons)
                
                # Step 3: Collect standings
                self.collect_standings(competition_code, seasons)
                
                print(f"\n✓ {league_name} collection complete!")
                
            except Exception as e:
                print(f"\n✗ Error collecting {league_name}: {str(e)}")
                continue
        
        print("\n" + "="*60)
        print("ALL LEAGUES DATA COLLECTION COMPLETE!")
        print("="*60)
        
        # Print summary
        self.print_database_summary()
    
    def print_database_summary(self):
        """Print summary of data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n=== DATABASE SUMMARY ===")
        
        # Count matches by league and season
        cursor.execute("""
            SELECT competition_code, competition_name, season, COUNT(*) 
            FROM matches 
            GROUP BY competition_code, competition_name, season 
            ORDER BY competition_code, season
        """)
        
        print("\nMatches by League and Season:")
        current_comp = None
        for comp_code, comp_name, season, count in cursor.fetchall():
            if comp_code != current_comp:
                print(f"\n{comp_name} ({comp_code}):")
                current_comp = comp_code
            print(f"  Season {season}: {count} matches")
        
        # Total matches
        cursor.execute("SELECT COUNT(*) FROM matches")
        match_count = cursor.fetchone()[0]
        print(f"\n{'='*40}")
        print(f"TOTAL MATCHES: {match_count}")
        
        # Count teams
        cursor.execute("SELECT COUNT(*) FROM teams")
        team_count = cursor.fetchone()[0]
        print(f"TOTAL TEAMS: {team_count}")
        
        # Count standings snapshots
        cursor.execute("SELECT COUNT(DISTINCT snapshot_date) FROM standings")
        snapshot_count = cursor.fetchone()[0]
        print(f"STANDINGS SNAPSHOTS: {snapshot_count}")
        print(f"{'='*40}")
        
        conn.close()
    
    def print_database_summary(self):
        """Print summary of data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n=== DATABASE SUMMARY ===")
        
        # Count matches
        cursor.execute("SELECT COUNT(*) FROM matches")
        match_count = cursor.fetchone()[0]
        print(f"Total matches: {match_count}")
        
        # Count by season
        cursor.execute("SELECT season, COUNT(*) FROM matches GROUP BY season ORDER BY season")
        for season, count in cursor.fetchall():
            print(f"  Season {season}: {count} matches")
        
        # Count teams
        cursor.execute("SELECT COUNT(*) FROM teams")
        team_count = cursor.fetchone()[0]
        print(f"\nTotal teams: {team_count}")
        
        # Count standings snapshots
        cursor.execute("SELECT COUNT(DISTINCT snapshot_date) FROM standings")
        snapshot_count = cursor.fetchone()[0]
        print(f"Standings snapshots: {snapshot_count}")
        
        conn.close()


# Main execution
if __name__ == "__main__":
    print("SOCCER PREDICTION DATA COLLECTOR")
    print("="*60)
    
    collector = DataCollector()
    collector.run_full_collection()