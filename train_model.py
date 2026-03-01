"""Train a LightGBM model on the housing dataset."""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pickle


def train_housing_model():
    """Train LightGBM model on housing price data."""
    print("Loading data...")
    df = pd.read_csv('housing.csv')

    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Prepare features and target
    # Drop 'Address' column as it's not useful for prediction
    X = df.drop(['Price', 'Address'], axis=1)
    y = df['Price']

    print(f"\nFeature columns: {list(X.columns)}")
    print(f"Target: Price")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Model parameters - keep it small for visualization
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Small trees for better visualization
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': 5  # Limit depth for clarity
    }

    print("\nTraining model...")
    print(f"Parameters: {params}")

    # Train model with early stopping
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=50,  # Limited number of trees for viz
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10)
        ]
    )

    print(f"\nNumber of trees: {booster.num_trees()}")

    # Evaluate
    train_pred = booster.predict(X_train)
    test_pred = booster.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print("\n" + "="*50)
    print("Model Performance:")
    print("="*50)
    print(f"Train RMSE: ${train_rmse:,.2f}")
    print(f"Test RMSE:  ${test_rmse:,.2f}")
    print(f"Train R²:   {train_r2:.4f}")
    print(f"Test R²:    {test_r2:.4f}")
    print("="*50)

    # Feature importance
    print("\nFeature Importance (gain):")
    importance = booster.feature_importance(importance_type='gain')
    for name, imp in zip(booster.feature_name(), importance):
        print(f"  {name:30s}: {imp:,.0f}")

    # Save model and data
    print("\nSaving model and data...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(booster, f)

    # Save train data for the visualizer
    with open('train_data.pkl', 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test), f)

    print("Model saved to 'model.pkl'")
    print("Data saved to 'train_data.pkl'")

    return booster, X_train, y_train


if __name__ == '__main__':
    train_housing_model()
