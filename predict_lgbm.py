"""
LightGBM prediction for THU-BDC2026.
Loads trained model and predicts top-5 stocks on the latest date.

Fixes:
  - Strict feature validation: raises on ANY mismatch
  - Deterministic: same features, same order as training
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))
from utils import engineer_features_158plus39


def main():
    data_path = './data/train.csv'
    model_path = './model/lgbm/best.txt'
    features_path = './model/lgbm/feature_cols.json'
    output_path = './output/result.csv'
    os.makedirs('./output', exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature list not found: {features_path}")

    # Load expected features
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)

    # Load and prepare data
    df = pd.read_csv(data_path, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)

    # Cross-sectional rank features (same as training — computed on full pool)
    for col in ['return_1', 'return_5', 'return_10', 'rsi', 'macd', 'volume_ratio', 'volatility_20']:
        rcol = f'{col}_rank'
        if col in df.columns and rcol in feature_cols:
            df[rcol] = df.groupby('日期')[col].rank(pct=True)

    # ── Strict feature validation ──
    latest = df[df['日期'] == df['日期'].max()].copy()
    missing_features = [c for c in feature_cols if c not in latest.columns]
    if missing_features:
        raise ValueError(
            f"Prediction data is missing {len(missing_features)} features "
            f"that were used during training: {missing_features[:10]}..."
        )

    # Build feature matrix in exact same order as training
    X = latest[feature_cols].values  # use feature_cols order, not DataFrame column order
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict
    model = lgb.Booster(model_file=model_path)
    latest = latest.copy()
    latest['pred'] = model.predict(X)
    top = latest.nlargest(5, 'pred')

    result = pd.DataFrame({
        'stock_id': top['股票代码'].values,
        'weight': [0.2] * 5
    })
    result.to_csv(output_path, index=False)

    print(f"Prediction date: {latest['日期'].max().date()}")
    print(f"Stocks: {result['stock_id'].tolist()}")
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
