"""
LightGBM training script for THU-BDC2026.
Saves model to model/lgbm/best.txt and scaler info.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings, os, sys, json

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))
from utils import engineer_features_158plus39


def load_and_engineer(csv_path):
    df = pd.read_csv(csv_path, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df = df.dropna(subset=['label'])
    df = df[df['open_t1'] > 1e-4]
    for col in ['return_1', 'return_5', 'return_10', 'rsi', 'macd', 'volume_ratio', 'volatility_20']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('日期')[col].rank(pct=True)
    meta = {'股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
            '振幅', '涨跌额', '换手率', '涨跌幅', 'label', 'open_t1', 'open_t5'}
    feature_cols = [c for c in df.columns if c not in meta]
    return df, feature_cols


def eval_competition(preds, labels, dates, top_k=5):
    unique_dates = np.unique(dates)
    scores = []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < top_k: continue
        p, lb = preds[mask], labels[mask]
        top = np.argsort(p)[::-1][:top_k]
        wr = lb[top].sum() * 0.2
        true_top = np.argsort(lb)[::-1][:top_k]
        mx = lb[true_top].sum() * 0.2
        rd = lb.mean() * top_k * 0.2
        scores.append((wr, mx, rd))
    if not scores: return 0, {}
    s = np.array(scores)
    wr, mx, rd = s.mean(axis=0)
    denom = mx - rd
    comp = (wr - rd) / (denom + 1e-12) if abs(denom) > 1e-6 else 0
    return wr, {'wr': wr, 'max': mx, 'rand': rd, 'comp': comp}


def main():
    data_path = './data/train.csv'
    output_dir = './model/lgbm'
    os.makedirs(output_dir, exist_ok=True)

    df, feature_cols = load_and_engineer(data_path)

    # Validation split
    last_date = df['日期'].max()
    val_start = last_date - pd.DateOffset(months=2)
    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_start].copy()

    X_train = np.nan_to_num(train_df[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_train = train_df['label'].values
    X_val = np.nan_to_num(val_df[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_val = val_df['label'].values
    val_dates = val_df['日期'].values

    params = {
        'objective': 'regression', 'metric': 'mse', 'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 20,
        'lambda_l1': 0.1, 'lambda_l2': 1.0, 'verbose': -1, 'n_jobs': -1,
        'seed': 9999
    }

    # Phase 1: Find best iteration on validation
    print("Finding best iteration...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=8000, valid_sets=[dval],
                      callbacks=[lgb.log_evaluation(2000)])

    best_wr, best_iter = -999, 50
    for n in range(50, 8001, 50):
        preds = model.predict(X_val, num_iteration=n)
        wr, _ = eval_competition(preds, y_val, val_dates)
        if wr > best_wr:
            best_wr, best_iter = wr, n
    _, m = eval_competition(model.predict(X_val, num_iteration=best_iter), y_val, val_dates)
    print(f"Best: iter={best_iter} WR={m['wr']:.6f} Comp={m['comp']:.6f}")

    # Phase 2: Retrain on ALL data
    print(f"Retraining on all data ({best_iter} iterations)...")
    X_all = np.nan_to_num(df[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_all = df['label'].values
    dtrain_all = lgb.Dataset(X_all, label=y_all)
    final_model = lgb.train(params, dtrain_all, num_boost_round=best_iter)

    final_model.save_model(os.path.join(output_dir, 'best.txt'))
    with open(os.path.join(output_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)
    with open(os.path.join(output_dir, 'train_info.json'), 'w') as f:
        json.dump({'best_iter': best_iter, 'val_wr': float(m['wr']), 'val_comp': float(m['comp'])}, f, indent=2)

    print(f"Saved: {output_dir}/best.txt ({best_iter} trees)")


if __name__ == '__main__':
    main()
