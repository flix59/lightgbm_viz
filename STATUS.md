# LightGBM Tree Visualizer - Project Status

## 📊 Overview
Building a web-based LightGBM tree visualizer using Dash/Plotly to visualize all trees in a boosted forest, showing split statistics and tree contributions.

## ✅ Completed

### 1. Project Setup
- ✅ Initialized uv project structure
- ✅ Added dependencies: `lightgbm`, `pandas`, `dash`, `plotly`, `scikit-learn`
- ✅ Created `pyproject.toml` with uv

### 2. Core Components Created

#### `tree_extractor.py`
- ✅ `TreeExtractor` class to extract tree structure from LightGBM booster
- ✅ Calculate sample counts at each node (how data flows through splits)
- ✅ Get tree summaries (node counts, split/leaf statistics)
- ✅ Extract feature importance
- ✅ Support for leaf index predictions

#### `train_model.py`
- ✅ Script to train LightGBM on housing.csv dataset
- ✅ Preprocesses data (drops Address column)
- ✅ Trains regression model with limited trees (50) and shallow depth (5) for better visualization
- ✅ Saves model to `model.pkl` and data to `train_data.pkl`
- ✅ Displays performance metrics (RMSE, R²)

#### `app.py`
- ✅ Dash web application for interactive visualization
- ✅ Feature importance bar chart
- ✅ Tree slider to navigate between all trees in forest
- ✅ Tree visualization showing:
  - Split nodes (blue) with feature name, threshold, sample count, gain
  - Leaf nodes (green) with sample count and predicted value
  - Node sizes proportional to sample counts
  - Hierarchical tree layout
- ✅ Tree statistics panel

## ❌ Missing / TODO

### 1. **Fix OpenMP Dependency** (Blocking!)
```bash
brew install libomp
```
LightGBM requires OpenMP for parallel processing. Need to install via Homebrew before training the model.

### 2. **Run Training**
```bash
uv run python train_model.py
```
After fixing OpenMP, train the model on housing.csv data.

### 3. **Launch Visualizer**
```bash
uv run python app.py
```
Start the Dash app and navigate to http://127.0.0.1:8050

## 🚀 Next Steps

1. **Install OpenMP**: `brew install libomp`
2. **Train model**: `uv run python train_model.py`
3. **Run visualizer**: `uv run python app.py`
4. **Test visualization**:
   - Navigate between trees using the slider
   - Verify split counts/proportions are displayed
   - Check feature importance chart
   - Inspect individual tree structures

## 🎯 Features Implemented

- ✅ Multi-tree forest visualization
- ✅ Sample flow statistics (how many samples go through each node)
- ✅ Split gain information
- ✅ Feature importance ranking
- ✅ Interactive tree navigation
- ✅ Node sizing based on sample counts
- ✅ Hierarchical tree layout

## 📁 File Structure

```
lightgbm_viz/
├── housing.csv           # Dataset
├── pyproject.toml        # uv dependencies
├── tree_extractor.py     # Core tree extraction logic
├── train_model.py        # Model training script
├── app.py               # Dash web application
├── model.pkl            # (generated) Trained model
├── train_data.pkl       # (generated) Training/test data
└── STATUS.md            # This file
```

## 💡 Tech Stack

- **Framework**: Dash + Plotly (customizable, production-ready)
- **ML Library**: LightGBM
- **Data**: Pandas, NumPy, scikit-learn
- **Package Manager**: uv
- **Visualization**: Plotly graph objects with hierarchical tree layout

## 🐛 Known Issue

OpenMP library missing on macOS - needs `brew install libomp` before LightGBM can run.
