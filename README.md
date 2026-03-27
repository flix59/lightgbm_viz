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
cd examples
uv run python ../src/train_model.py
cd ..
```
Trains a LightGBM regression model on `housing.csv` and saves `model.pkl` and `train_data.pkl`.

**2. Create visualization**
```bash
uv run python src/create_vizualization.py --model examples/model.pkl --data examples/train_data.pkl --output examples/forest_viz.html
```
Generates an interactive HTML visualization of the forest. Open `examples/forest_viz.html` in your browser.

**3. Analyze feature influence**
```bash
uv run python src/influence.py
```
Analyzes feature influence across the forest (requires trained model).

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
├── src/
│   ├── tree_extractor.py      # Extracts tree structure from LightGBM booster
│   ├── train_model.py         # Trains and saves the model
│   ├── create_vizualization.py # Creates interactive forest visualization
│   └── influence.py           # Analyzes feature influence
├── examples/
│   ├── housing.csv      # Dataset (5000 rows, housing prices)
│   ├── model.pkl        # Trained model (generated)
│   ├── train_data.pkl   # Train/test split (generated)
│   └── forest_viz.html  # Interactive visualization (generated)
├── frontend/            # Frontend components (if applicable)
├── pyproject.toml       # uv dependencies
└── .pre-commit-config.yaml  # Pre-commit hooks configuration
```

## Dataset

`examples/housing.csv` — 5000 synthetic US housing records. Target: `Price`. Features:

- `Avg. Area Income`
- `Avg. Area House Age`
- `Avg. Area Number of Rooms`
- `Avg. Area Number of Bedrooms`
- `Area Population`
