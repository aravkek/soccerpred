# Soccer Match Prediction Algorithm

A machine learning system that predicts soccer match outcomes across Europe's top leagues using historical match data and advanced ensemble models.

## What it does

- Pulls historical match data from the [Football-Data.org API](https://www.football-data.org/) for 6 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Champions League)
- Engineers features from match results, standings, and form
- Trains ensemble models (Random Forest, XGBoost, LightGBM, Gradient Boosting) per league
- Produces win/draw/loss probability predictions for upcoming fixtures
- Includes an adaptive learning system that retrains on new results over time

## Leagues covered

| Code | League | Country |
|------|--------|---------|
| PL | Premier League | England |
| PD | La Liga | Spain |
| BL1 | Bundesliga | Germany |
| SA | Serie A | Italy |
| FL1 | Ligue 1 | France |
| CL | UEFA Champions League | Europe |

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm requests
   ```
3. Get a free API key at [football-data.org](https://www.football-data.org/client/register)
4. In `config.py`, replace `YOUR_API_KEY_HERE` with your key
5. Collect data:
   ```bash
   python data_collector.py
   ```
6. Train models:
   ```bash
   python league_specific_binary_training_optimized.py
   ```
7. Run predictions:
   ```bash
   python predictor2.py
   ```

## Project structure

```
├── config.py                              # League/API/model configuration
├── api_client.py                          # Football-Data.org API wrapper
├── data_collector.py                      # Pulls and stores match data
├── feature_engineering.py                 # Feature construction from raw data
├── feature_selection.py                   # Selects best features per league
├── model_training.py                      # Base model training pipeline
├── binary_model_training.py               # Binary (home/away) model variant
├── league_specific_binary_training.py     # Per-league model training
├── league_specific_binary_training_optimized.py  # Optimized version
├── adaptive_learning_system.py            # Retraining on new match results
├── advanced_betting.py                    # Kelly Criterion bankroll management
├── predictor.py / predictor2.py           # Prediction runners
└── models/
    └── feature_selection/                 # Selected feature lists per league
```

## Models

Each league gets its own trained ensemble. The binary classification task is predicting whether the home team wins (1) or doesn't (0). Final predictions use a weighted ensemble of all trained models.
