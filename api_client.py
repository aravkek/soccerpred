"""
API Client for Football-Data.org
Handles all API requests with rate limiting and error handling
"""

import requests
import time
from typing import Dict, Optional
from pathlib import Path
import logging
from datetime import datetime

# Import from config.py in same directory
from config import API_CONFIG, STORAGE

class FootballDataClient:
    """Client for interacting with Football-Data.org API"""
    
    def __init__(self):
        """Initialize the API client"""
        # Load configuration from config.py
        self.base_url = API_CONFIG['base_url']
        self.api_key = API_CONFIG['api_key']
        self.rate_limit = API_CONFIG['rate_limit']
        self.timeout = API_CONFIG['timeout']
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60 / self.rate_limit
        
        # Set up logging
        self._setup_logging()
        
        # Validate API key
        if self.api_key == "YOUR_FOOTBALL_DATA_ORG_API_KEY":
            raise ValueError("Please replace YOUR_FOOTBALL_DATA_ORG_API_KEY with your actual API key in config.py")
        
        self.logger.info("Football-Data.org Client initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path(STORAGE['log_path'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"api_client_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Implement rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the API with error handling
        
        Args:
            endpoint: API endpoint (e.g., '/competitions/PL/matches')
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            'X-Auth-Token': self.api_key
        }
        
        try:
            self.logger.info(f"Making request to {endpoint} with params: {params}")
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            self.logger.info(f"Successfully retrieved data from {endpoint}")
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response: {e.response.text}")
            raise
    
    def get_competitions(self) -> Dict:
        """Get all available competitions"""
        return self._make_request('/competitions')
    
    def get_competition(self, competition_code: str) -> Dict:
        """
        Get details of a specific competition
        
        Args:
            competition_code: Competition code (e.g., 'PL' for Premier League)
            
        Returns:
            Dictionary containing competition data
        """
        return self._make_request(f'/competitions/{competition_code}')
    
    def get_matches(self, competition_code: str, season: Optional[int] = None, 
                   status: Optional[str] = None) -> Dict:
        """
        Get matches for a specific competition
        
        Args:
            competition_code: Competition code (e.g., 'PL')
            season: Season year (e.g., 2024 for 2024-2025 season). If None, gets current season
            status: Match status filter ('SCHEDULED', 'LIVE', 'IN_PLAY', 'PAUSED', 'FINISHED', 'POSTPONED', 'SUSPENDED', 'CANCELLED')
            
        Returns:
            Dictionary containing matches data
        """
        endpoint = f'/competitions/{competition_code}/matches'
        params = {}
        
        if season:
            params['season'] = season
        if status:
            params['status'] = status
            
        return self._make_request(endpoint, params=params)
    
    def get_match(self, match_id: int) -> Dict:
        """
        Get details of a specific match
        
        Args:
            match_id: Match ID
            
        Returns:
            Dictionary containing match data
        """
        return self._make_request(f'/matches/{match_id}')
    
    def get_standings(self, competition_code: str, season: Optional[int] = None) -> Dict:
        """
        Get league standings
        
        Args:
            competition_code: Competition code (e.g., 'PL')
            season: Season year (defaults to current season)
            
        Returns:
            Dictionary containing standings data
        """
        endpoint = f'/competitions/{competition_code}/standings'
        params = {}
        
        if season:
            params['season'] = season
            
        return self._make_request(endpoint, params=params)
    
    def get_teams(self, competition_code: str, season: Optional[int] = None) -> Dict:
        """
        Get teams in a competition
        
        Args:
            competition_code: Competition code (e.g., 'PL')
            season: Season year (defaults to current season)
            
        Returns:
            Dictionary containing teams data
        """
        endpoint = f'/competitions/{competition_code}/teams'
        params = {}
        
        if season:
            params['season'] = season
            
        return self._make_request(endpoint, params=params)
    
    def get_team(self, team_id: int) -> Dict:
        """
        Get details of a specific team
        
        Args:
            team_id: Team ID
            
        Returns:
            Dictionary containing team data
        """
        return self._make_request(f'/teams/{team_id}')
    
    def get_team_matches(self, team_id: int, season: Optional[int] = None, 
                        status: Optional[str] = None) -> Dict:
        """
        Get matches for a specific team
        
        Args:
            team_id: Team ID
            season: Season year
            status: Match status filter
            
        Returns:
            Dictionary containing matches data
        """
        endpoint = f'/teams/{team_id}/matches'
        params = {}
        
        if season:
            params['season'] = season
        if status:
            params['status'] = status
            
        return self._make_request(endpoint, params=params)
    
    def get_scorers(self, competition_code: str, season: Optional[int] = None) -> Dict:
        """
        Get top scorers for a competition
        
        Args:
            competition_code: Competition code (e.g., 'PL')
            season: Season year (defaults to current season)
            
        Returns:
            Dictionary containing top scorers data
        """
        endpoint = f'/competitions/{competition_code}/scorers'
        params = {}
        
        if season:
            params['season'] = season
            
        return self._make_request(endpoint, params=params)


# Test function
if __name__ == "__main__":
    # Test the API client
    client = FootballDataClient()
    
    print("\n=== Testing Football-Data.org Client ===")
    
    # Test: Get Premier League info
    print("\n1. Fetching Premier League competition info...")
    pl_info = client.get_competition('PL')
    print(f"Competition: {pl_info['name']}")
    print(f"Current Season: {pl_info['currentSeason']['startDate']} to {pl_info['currentSeason']['endDate']}")
    
    # Test: Get current season matches
    print("\n2. Fetching Premier League 2025-2026 season matches...")
    matches = client.get_matches('PL')  # Defaults to current season
    print(f"Total matches: {matches.get('resultSet', {}).get('count', 0)}")
    if matches.get('matches'):
        first_match = matches['matches'][0]
        print(f"First match: {first_match['homeTeam']['name']} vs {first_match['awayTeam']['name']}")
        print(f"Status: {first_match['status']}")
        print(f"Date: {first_match['utcDate']}")
    
    # Test: Get standings
    print("\n3. Fetching current Premier League standings...")
    standings = client.get_standings('PL')
    if standings.get('standings'):
        table = standings['standings'][0]['table']
        print(f"\nTop 3 teams:")
        for i, team in enumerate(table[:3], 1):
            print(f"{i}. {team['team']['name']} - {team['points']} points")
    
    print("\n=== All tests passed! ===")