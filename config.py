"""
Configuration file for Soccer Prediction System
BETTING-FOCUSED: Real-time predictions with continuous data updates
Current Date: November 2025 (2025-2026 Season)
API: Football-Data.org (FREE tier)
"""

# API Configuration - Football-Data.org
API_CONFIG = {
    'base_url': 'https://api.football-data.org/v4',
    'api_key': 'YOUR_API_KEY_HERE',  # Get a free key at https://www.football-data.org/client/register
    'rate_limit': 10,  # 10 calls per minute on free tier
    'timeout': 30  # seconds
}

# Target Leagues - Football-Data.org uses different IDs
LEAGUES = {
    'premier_league': {
        'id': 'PL',  # Premier League code
        'name': 'Premier League',
        'country': 'England'
    },
    'la_liga': {
        'id': 'PD',  # Primera Division
        'name': 'La Liga',
        'country': 'Spain'
    },
    'bundesliga': {
        'id': 'BL1',
        'name': 'Bundesliga',
        'country': 'Germany'
    },
    'serie_a': {
        'id': 'SA',
        'name': 'Serie A',
        'country': 'Italy'
    },
    'ligue_1': {
        'id': 'FL1',
        'name': 'Ligue 1',
        'country': 'France'
    },
    'champions_league': {
        'id': 'CL',
        'name': 'UEFA Champions League',
        'country': 'Europe'
    }
}

# Seasons Strategy (Updated for November 2025)
# Football-Data.org uses year for season (2024 = 2024-2025 season)
SEASONS = {
    'historical': [2022, 2023, 2024],  # Last 3 COMPLETED seasons for training
    'current': 2025,  # Current ongoing season (2025-2026)
    'all_available': [2022, 2023, 2024, 2025]  # All seasons we can access
}

# Data storage paths
STORAGE = {
    'raw_data_path': 'data/raw',
    'processed_data_path': 'data/processed',
    'database_path': 'data/soccer_predictions.db',  # SQLite database
    'log_path': 'logs'
}

# MVP Settings - Start with Premier League only
MVP_CONFIG = {
    'target_league': 'premier_league',
    'target_league_code': 'PL',
    'historical_seasons': [2022, 2023, 2024],  # 3 completed years for training
    'current_season': 2025,  # What we're predicting on NOW
    'update_frequency': 'daily',  # Update database daily with new results
    'prediction_mode': 'live'  # Focus on upcoming matches for betting
}

# Betting Strategy
BETTING_CONFIG = {
    'min_confidence': 0.60,  # Only bet when model is 60%+ confident
    'target_roi': 0.15,  # Aim for 15% return on investment
    'bankroll_percent': 0.02,  # Risk 2% per bet (Kelly Criterion)
    'track_performance': True  # Log all predictions vs actual results
}

# Update Schedule (for automation later)
UPDATE_SCHEDULE = {
    'historical_pull': 'once',  # Pull historical data once and store in DB
    'daily_update': '06:00',  # Update at 6 AM daily for new match results
    'pre_match_update': '4_hours_before',  # Refresh data 4 hours before kickoff
    'live_odds_check': False  # Football-Data.org doesn't have odds on free tier
}