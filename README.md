# LightGBM Tree Visualizer

Interactive visualization tool for exploring all trees in a LightGBM boosted forest with a modern frontend.

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
uv run python example/train_model.py
```
Trains a LightGBM regression model on `housing.csv` and saves `model.pkl` and `train_data.pkl`.

**2. Generate visualization data**
```bash
uv run python src/create_vizualization.py --model example/model.pkl --data example/train_data.pkl --output example/forest_viz.html
```
Processes the trained model and generates visualization output.

**3. View the visualization**
Navigate to the `frontend/` directory and follow its setup instructions to view the interactive visualization.

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
│   ├── tree_extractor.py       # Extracts tree structure from LightGBM booster
│   ├── create_vizualization.py # Creates interactive forest visualization
│   └── influence.py            # Analyzes feature influence
├── example/
│   ├── train_model.py          # Training script (example usage)
│   ├── housing.csv             # Dataset (5000 rows, housing prices)
│   ├── model.pkl               # Trained model (generated)
│   ├── train_data.pkl          # Train/test split (generated)
│   └── forest_viz.html         # Interactive visualization (generated)
├── frontend/                   # Frontend components
├── pyproject.toml              # uv dependencies
└── .pre-commit-config.yaml     # Pre-commit hooks configuration
```

## Dataset

`example/housing.csv` — 5000 synthetic US housing records. Target: `Price`. Features:

- `Avg. Area Income`
- `Avg. Area House Age`
- `Avg. Area Number of Rooms`
- `Avg. Area Number of Bedrooms`
- `Area Population`
