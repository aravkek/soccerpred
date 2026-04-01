#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:57:23 2025

@author: aravkekane
"""

"""
Advanced Betting Module
- Kelly Criterion for optimal bet sizing
- Monte Carlo simulation for risk assessment
- Bankroll management
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from pathlib import Path


class KellyCriterion:
    """
    Kelly Criterion: Optimal bet sizing formula
    
    Formula: f = (bp - q) / b
    Where:
        f = fraction of bankroll to bet
        b = odds received (decimal odds - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    """
    
    def __init__(self, bankroll=1000, max_bet_pct=0.05):
        """
        Args:
            bankroll: Total bankroll in currency
            max_bet_pct: Maximum % of bankroll to risk (safety cap)
        """
        self.bankroll = bankroll
        self.max_bet_pct = max_bet_pct
    
    def calculate_kelly(self, win_prob, decimal_odds):
        """
        Calculate optimal bet size using Kelly Criterion
        
        Args:
            win_prob: Your model's probability (0-1)
            decimal_odds: Bookmaker odds (e.g., 2.0 for even money)
            
        Returns:
            bet_amount, kelly_fraction, recommendation
        """
        # Kelly formula
        b = decimal_odds - 1  # Net odds
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Fractional Kelly (more conservative - use 25% of full Kelly)
        # This reduces variance and risk of ruin
        fractional_kelly = kelly_fraction * 0.25
        
        # Cap at max_bet_pct for safety
        final_fraction = min(fractional_kelly, self.max_bet_pct)
        final_fraction = max(final_fraction, 0)  # No negative bets
        
        bet_amount = self.bankroll * final_fraction
        
        # Recommendation
        if final_fraction < 0.005:  # Less than 0.5%
            recommendation = "SKIP - Edge too small"
        elif final_fraction < 0.01:  # 0.5-1%
            recommendation = "SMALL BET - Minimal edge"
        elif final_fraction < 0.03:  # 1-3%
            recommendation = "MEDIUM BET - Good edge"
        else:  # 3%+
            recommendation = "LARGE BET - Strong edge"
        
        return {
            'bet_amount': round(bet_amount, 2),
            'kelly_fraction': round(final_fraction, 4),
            'kelly_percentage': round(final_fraction * 100, 2),
            'recommendation': recommendation,
            'expected_value': self._calculate_ev(win_prob, decimal_odds)
        }
    
    def _calculate_ev(self, win_prob, decimal_odds):
        """Calculate expected value of bet"""
        win_amount = decimal_odds - 1  # Net profit if win
        lose_amount = -1  # Lose stake
        
        ev = (win_prob * win_amount) + ((1 - win_prob) * lose_amount)
        return round(ev, 4)
    
    def should_bet(self, win_prob, decimal_odds, min_ev=0.05):
        """
        Determine if bet has positive expected value
        
        Args:
            win_prob: Model probability
            decimal_odds: Bookmaker odds
            min_ev: Minimum EV to consider bet (5% = 0.05)
            
        Returns:
            bool, reason
        """
        ev = self._calculate_ev(win_prob, decimal_odds)
        
        if ev < 0:
            return False, f"Negative EV ({ev:.2%})"
        elif ev < min_ev:
            return False, f"EV too low ({ev:.2%} < {min_ev:.2%})"
        else:
            return True, f"Positive EV ({ev:.2%})"
    
    def calculate_portfolio(self, bets):
        """
        Calculate optimal portfolio of multiple bets
        
        Args:
            bets: List of dict with {'win_prob': 0.6, 'odds': 2.0, 'match': 'A vs B'}
            
        Returns:
            Portfolio recommendations
        """
        recommendations = []
        total_allocation = 0
        
        for bet in bets:
            result = self.calculate_kelly(bet['win_prob'], bet['odds'])
            result['match'] = bet.get('match', 'Unknown')
            recommendations.append(result)
            total_allocation += result['kelly_fraction']
        
        # Adjust if total allocation > max
        if total_allocation > self.max_bet_pct * len(bets):
            scale_factor = (self.max_bet_pct * len(bets)) / total_allocation
            for rec in recommendations:
                rec['bet_amount'] *= scale_factor
                rec['kelly_fraction'] *= scale_factor
                rec['kelly_percentage'] *= scale_factor
        
        return recommendations


class MonteCarloSimulator:
    """
    Monte Carlo Simulation for match outcomes
    
    Simulates matches thousands of times to:
    1. Get probability distributions
    2. Assess risk/variance
    3. Calculate confidence intervals
    """
    
    def __init__(self, n_simulations=10000):
        """
        Args:
            n_simulations: Number of times to simulate (10,000 default)
        """
        self.n_simulations = n_simulations
    
    def simulate_match_poisson(self, home_attack, home_defense, 
                                away_attack, away_defense, home_advantage=1.3):
        """
        Simulate match using Poisson distribution
        
        Args:
            home_attack: Home team attacking strength (goals/game)
            home_defense: Home team defensive strength (goals conceded/game)
            away_attack: Away team attacking strength
            away_defense: Away team defensive strength
            home_advantage: Multiplier for home team (typically 1.2-1.4)
            
        Returns:
            Probabilities for Home/Draw/Away
        """
        # Calculate expected goals using attacking/defensive strength
        home_xg = home_attack * away_defense * home_advantage
        away_xg = away_attack * home_defense
        
        # Simulate n_simulations matches
        home_goals = np.random.poisson(home_xg, self.n_simulations)
        away_goals = np.random.poisson(away_xg, self.n_simulations)
        
        # Calculate outcome probabilities
        home_wins = (home_goals > away_goals).sum() / self.n_simulations
        draws = (home_goals == away_goals).sum() / self.n_simulations
        away_wins = (home_goals < away_goals).sum() / self.n_simulations
        
        # Most likely score
        score_counts = {}
        for h, a in zip(home_goals, away_goals):
            score = (h, a)
            score_counts[score] = score_counts.get(score, 0) + 1
        
        most_likely_score = max(score_counts, key=score_counts.get)
        
        # Over/Under 2.5 goals
        total_goals = home_goals + away_goals
        over_2_5 = (total_goals > 2.5).sum() / self.n_simulations
        
        # Both teams to score
        btts = ((home_goals > 0) & (away_goals > 0)).sum() / self.n_simulations
        
        return {
            'home_win_prob': home_wins,
            'draw_prob': draws,
            'away_win_prob': away_wins,
            'expected_home_goals': home_xg,
            'expected_away_goals': away_xg,
            'most_likely_score': most_likely_score,
            'over_2_5_prob': over_2_5,
            'btts_prob': btts,
            'simulations': self.n_simulations
        }
    
    def simulate_betting_outcomes(self, win_prob, odds, stake, n_bets=100):
        """
        Simulate betting outcomes over many bets
        
        Args:
            win_prob: Probability of winning each bet
            odds: Decimal odds
            stake: Amount bet each time
            n_bets: Number of bets to simulate
            
        Returns:
            Profit distribution and statistics
        """
        profits = []
        
        for _ in range(self.n_simulations):
            bankroll = 0
            for _ in range(n_bets):
                # Random outcome based on probability
                if np.random.random() < win_prob:
                    bankroll += stake * (odds - 1)  # Win
                else:
                    bankroll -= stake  # Lose
            profits.append(bankroll)
        
        profits = np.array(profits)
        
        return {
            'mean_profit': profits.mean(),
            'median_profit': np.median(profits),
            'std_profit': profits.std(),
            'min_profit': profits.min(),
            'max_profit': profits.max(),
            'prob_profit': (profits > 0).sum() / self.n_simulations,
            'percentile_5': np.percentile(profits, 5),
            'percentile_95': np.percentile(profits, 95),
            'risk_of_ruin': (profits < -stake * n_bets * 0.5).sum() / self.n_simulations
        }
    
    def calculate_confidence_interval(self, probabilities, confidence=0.95):
        """
        Calculate confidence interval for probability estimate
        
        Args:
            probabilities: Array of simulated probabilities
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            Lower and upper bounds
        """
        alpha = 1 - confidence
        lower = np.percentile(probabilities, alpha/2 * 100)
        upper = np.percentile(probabilities, (1 - alpha/2) * 100)
        
        return lower, upper


class BankrollManager:
    """Manage betting bankroll and track performance"""
    
    def __init__(self, initial_bankroll=1000):
        """
        Args:
            initial_bankroll: Starting bankroll
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bets_history = []
    
    def place_bet(self, stake, outcome, odds, description=""):
        """
        Record a bet
        
        Args:
            stake: Amount bet
            outcome: 'win' or 'loss'
            odds: Decimal odds
            description: Match description
        """
        if outcome == 'win':
            profit = stake * (odds - 1)
        else:
            profit = -stake
        
        self.current_bankroll += profit
        
        self.bets_history.append({
            'stake': stake,
            'outcome': outcome,
            'odds': odds,
            'profit': profit,
            'bankroll_after': self.current_bankroll,
            'description': description,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_stats(self):
        """Get betting statistics"""
        if not self.bets_history:
            return {"message": "No bets placed yet"}
        
        df = pd.DataFrame(self.bets_history)
        
        total_bets = len(df)
        wins = (df['outcome'] == 'win').sum()
        losses = (df['outcome'] == 'loss').sum()
        win_rate = wins / total_bets
        
        total_profit = df['profit'].sum()
        roi = (total_profit / self.initial_bankroll) * 100
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi_percentage': roi,
            'current_bankroll': self.current_bankroll,
            'initial_bankroll': self.initial_bankroll,
            'max_bankroll': df['bankroll_after'].max(),
            'min_bankroll': df['bankroll_after'].min()
        }
    
    def plot_bankroll_history(self, save_path='bankroll_history.png'):
        """Plot bankroll over time"""
        if not self.bets_history:
            print("No bets to plot")
            return
        
        df = pd.DataFrame(self.bets_history)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['bankroll_after'], marker='o')
        plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll ($)')
        plt.title('Bankroll Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("ADVANCED BETTING MODULE - EXAMPLES")
    print("="*70)
    
    # Example 1: Kelly Criterion
    print("\n1. KELLY CRITERION EXAMPLE")
    print("-"*70)
    
    kelly = KellyCriterion(bankroll=1000, max_bet_pct=0.05)
    
    # Your model says 60% home win, bookmaker offers 2.0 odds
    result = kelly.calculate_kelly(win_prob=0.60, decimal_odds=2.0)
    
    print(f"Model probability: 60%")
    print(f"Bookmaker odds: 2.0")
    print(f"Kelly bet size: ${result['bet_amount']} ({result['kelly_percentage']}% of bankroll)")
    print(f"Expected Value: {result['expected_value']:.2%}")
    print(f"Recommendation: {result['recommendation']}")
    
    # Example 2: Monte Carlo Simulation
    print("\n\n2. MONTE CARLO SIMULATION EXAMPLE")
    print("-"*70)
    
    mc = MonteCarloSimulator(n_simulations=10000)
    
    # Liverpool (strong attack, good defense) vs Nottingham Forest (weak)
    result = mc.simulate_match_poisson(
        home_attack=2.1,  # Liverpool scores 2.1 goals/game at home
        home_defense=0.8,  # Liverpool concedes 0.8 goals/game at home
        away_attack=1.1,   # Nottingham scores 1.1 away
        away_defense=1.5,  # Nottingham concedes 1.5 away
        home_advantage=1.3
    )
    
    print(f"Simulated {result['simulations']} matches:")
    print(f"  Home Win: {result['home_win_prob']:.1%}")
    print(f"  Draw: {result['draw_prob']:.1%}")
    print(f"  Away Win: {result['away_win_prob']:.1%}")
    print(f"  Most likely score: {result['most_likely_score'][0]}-{result['most_likely_score'][1]}")
    print(f"  Over 2.5 goals: {result['over_2_5_prob']:.1%}")
    print(f"  Both teams to score: {result['btts_prob']:.1%}")
    
    # Example 3: Betting simulation
    print("\n\n3. BETTING OUTCOME SIMULATION")
    print("-"*70)
    
    betting_sim = mc.simulate_betting_outcomes(
        win_prob=0.60,  # 60% win rate
        odds=2.0,       # 2.0 odds
        stake=20,       # $20 per bet
        n_bets=100      # 100 bets
    )
    
    print(f"After 100 bets at $20 each:")
    print(f"  Mean profit: ${betting_sim['mean_profit']:.2f}")
    print(f"  Median profit: ${betting_sim['median_profit']:.2f}")
    print(f"  5th percentile (worst case): ${betting_sim['percentile_5']:.2f}")
    print(f"  95th percentile (best case): ${betting_sim['percentile_95']:.2f}")
    print(f"  Probability of profit: {betting_sim['prob_profit']:.1%}")
    print(f"  Risk of losing >50% bankroll: {betting_sim['risk_of_ruin']:.1%}")
    
    print("\n" + "="*70)
    print("✅ Ready to use in your predictor!")
    print("="*70)