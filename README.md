# LightGBM Tree Visualizer

Interactive web app for visualizing all trees in a LightGBM boosted forest, built with Dash + Plotly.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and macOS with Homebrew.

```bash
# Install OpenMP (required by LightGBM on macOS, one-time)
brew install libomp

# Install Python dependencies
uv sync
```

## Usage

**1. Train the model**
```bash
uv run python train_model.py
```
Trains a LightGBM regression model on `housing.csv` and saves `model.pkl` + `train_data.pkl`.

**2. Start the visualizer**
```bash
uv run python app.py
```
Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

## Features

- **Feature importance chart** — ranked by gain across all trees
- **Tree navigator** — slider to browse all 50 trees in the forest
- **Tree diagram** — interactive graph showing:
  - Split nodes (blue): feature name, threshold, sample count, gain
  - Leaf nodes (green): sample count, predicted value
  - Node size scaled to sample count
- **Tree statistics** — node/leaf counts and total samples per tree

## File Structure

```
lightgbm_viz/
├── housing.csv         # Dataset (5000 rows, housing prices)
├── pyproject.toml      # uv dependencies
├── tree_extractor.py   # Extracts tree structure from LightGBM booster
├── train_model.py      # Trains and saves the model
├── app.py              # Dash web application
├── model.pkl           # Trained model (generated)
└── train_data.pkl      # Train/test split (generated)
```

## Dataset

`housing.csv` — 5000 synthetic US housing records. Target: `Price`. Features:

- `Avg. Area Income`
- `Avg. Area House Age`
- `Avg. Area Number of Rooms`
- `Avg. Area Number of Bedrooms`
- `Area Population`
