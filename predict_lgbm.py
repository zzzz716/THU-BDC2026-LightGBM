"""
LightGBM prediction script for THU-BDC2026.
Loads model and predicts top 5 stocks on the latest date.
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
    output_path = './output/result.csv'
    os.makedirs('./output', exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load feature cols
    with open('./model/lgbm/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Load and prepare data
    df = pd.read_csv(data_path, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)

    # Add cross-sectional rank features
    for col in ['return_1', 'return_5', 'return_10', 'rsi', 'macd', 'volume_ratio', 'volatility_20']:
        rcol = f'{col}_rank'
        if col in df.columns and rcol in feature_cols:
            df[rcol] = df.groupby('日期')[col].rank(pct=True)

    # Load model
    model = lgb.Booster(model_file=model_path)

    # Predict on latest date
    latest = df[df['日期'] == df['日期'].max()].copy()
    avail = [c for c in feature_cols if c in latest.columns]
    X = np.nan_to_num(latest[avail].values, nan=0, posinf=0, neginf=0)

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
