"""
LightGBM training for THU-BDC2026 v3.

All known issues fixed:
  1. Data sorted by (日期, 股票代码) before LambdaRank grouping
  2. Holdout is NEVER used for training — final model uses CV avg iterations
  3. Purge gap of 5 trading days between train/val splits
  4. RSQR feature fixed (rolling correlation on full series)
  5. Continuous relevance (cross-sectional rank of returns) instead of binary
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings, os, sys, json

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))
from utils import engineer_features_158plus39

DATA_PATH = './data/train.csv'
MODEL_DIR = './model/lgbm'
SEED = 42
TOP_K = 5
PURGE_DAYS = 5  # label uses T+1 to T+5, so need ≥5 day gap


# ─────────────────────────────────────────────────────
# 1. Data loading & feature engineering
# ─────────────────────────────────────────────────────
def load_and_engineer(csv_path):
    df = pd.read_csv(csv_path, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # Per-stock feature engineering
    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)

    # ── Cross-sectional rank features (BEFORE label filtering) ──
    for col in ['return_1', 'return_5', 'return_10', 'rsi', 'macd',
                'volume_ratio', 'volatility_20']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('日期')[col].rank(pct=True)

    # ── Labels ──
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df = df.dropna(subset=['label'])
    df = df[df['open_t1'] > 1e-4]

    # ── Continuous relevance: cross-sectional rank of returns per day ──
    # Higher return → higher relevance → model learns to rank by return magnitude
    df['relevance'] = df.groupby('日期')['label'].rank(pct=True)
    # Scale to 0-4 for NDCG (integer-ish grades work better with LambdaRank)
    df['relevance'] = (df['relevance'] * 4).clip(0, 4).astype(int)

    # ── CRITICAL: sort by (日期, 股票代码) so LambdaRank groups are correct ──
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    # ── Feature columns ──
    meta = {'股票代码', '日期', '开盘', '收盘', '最高', '最低',
            '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
            'label', 'open_t1', 'open_t5', 'relevance'}
    feature_cols = [c for c in df.columns if c not in meta]
    return df, feature_cols


# ─────────────────────────────────────────────────────
# 2. Evaluation
# ─────────────────────────────────────────────────────
def eval_competition(preds, labels, dates, top_k=5):
    unique_dates = np.unique(dates)
    scores = []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < top_k:
            continue
        p, lb = preds[mask], labels[mask]
        top = np.argsort(p)[::-1][:top_k]
        wr = lb[top].sum() / top_k
        true_top = np.argsort(lb)[::-1][:top_k]
        mx = lb[true_top].sum() / top_k
        rd = lb.mean()
        scores.append((wr, mx, rd))
    if not scores:
        return 0, {}
    s = np.array(scores)
    wr, mx, rd = s.mean(axis=0)
    denom = mx - rd
    comp = (wr - rd) / (denom + 1e-12) if abs(denom) > 1e-6 else 0
    return wr, {'wr': float(wr), 'max': float(mx), 'rand': float(rd),
                'comp': float(comp), 'n': len(scores)}


# ─────────────────────────────────────────────────────
# 3. Time-series splits with PURGE gap
# ─────────────────────────────────────────────────────
def purge_split(df, n_folds=3, holdout_months=1):
    """Time-series splits with PURGE_DAYS gap between train and val.

    Timeline:  [──fold0 train──][gap][──fold0 val──]...[gap][──holdout──]
    """
    dates = sorted(df['日期'].unique())
    n_dates = len(dates)

    # Holdout: last holdout_months
    holdout_start = df['日期'].max() - pd.DateOffset(months=holdout_months)
    ho_dates = [d for d in dates if d >= holdout_start]
    cv_dates = [d for d in dates if d < holdout_start]
    n_cv = len(cv_dates)

    # Each fold: train = [0:split], gap = [split:split+PURGE], val = [split+PURGE:split+fold_size+PURGE]
    fold_size = (n_cv - PURGE_DAYS * n_folds) // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        gap_end = train_end + PURGE_DAYS
        val_end = gap_end + fold_size

        train_dates = set(cv_dates[:train_end])
        val_dates = set(cv_dates[gap_end:min(val_end, n_cv)])

        folds.append((train_dates, val_dates))

    return folds, set(ho_dates), holdout_start


def get_groups(dates):
    _, counts = np.unique(dates, return_counts=True)
    return counts.tolist()


# ─────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────
BASE_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5],
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'n_jobs': 1,
    'force_row_wise': True,
    'seed': SEED,
    'deterministic': True,
}


def train_fold(X_tr, y_tr, tr_dates, X_v, y_v, v_dates, n_rounds=3000):
    dtrain = lgb.Dataset(X_tr, label=y_tr, group=get_groups(tr_dates))
    dval = lgb.Dataset(X_v, label=y_v, group=get_groups(v_dates), reference=dtrain)
    model = lgb.train(
        BASE_PARAMS, dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(500),
        ]
    )
    return model


# ─────────────────────────────────────────────────────
# 5. Save & predict
# ─────────────────────────────────────────────────────
def predict_top5(model, feature_cols, top_k=5):
    df = pd.read_csv(DATA_PATH, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)

    for col in ['return_1', 'return_5', 'return_10', 'rsi', 'macd',
                'volume_ratio', 'volatility_20']:
        rcol = f'{col}_rank'
        if col in df.columns and rcol in feature_cols:
            df[rcol] = df.groupby('日期')[col].rank(pct=True)

    latest = df[df['日期'] == df['日期'].max()].copy()
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = latest[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    latest = latest.copy()
    latest['pred'] = model.predict(X)
    top = latest.nlargest(top_k, 'pred')

    result = pd.DataFrame({
        'stock_id': top['股票代码'].values,
        'weight': [1.0 / top_k] * top_k
    })
    os.makedirs('./output', exist_ok=True)
    result.to_csv('./output/result.csv', index=False)
    print(f"  Prediction date: {latest['日期'].max().date()}")
    print(f"  Stocks: {result['stock_id'].tolist()}")
    return result


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("LightGBM v3 — LambdaRank + Purge CV + Fixed Groups")
    print("=" * 60)

    df, feature_cols = load_and_engineer(DATA_PATH)
    print(f"  Features: {len(feature_cols)}, Rows: {len(df)}")
    print(f"  Date range: {df['日期'].min().date()} ~ {df['日期'].max().date()}")
    print(f"  Data sorted by (日期, 股票代码): {df['日期'].is_monotonic_increasing}")

    # ── Purge CV ──
    folds, ho_dates, holdout_start = purge_split(df, n_folds=3, holdout_months=1)
    holdout_df = df[df['日期'].isin(ho_dates)].copy()
    cv_df = df[~df['日期'].isin(ho_dates)].copy()

    print(f"\n── CV (3 folds, purge={PURGE_DAYS} days) ──")
    cv_results = []
    for i, (train_dates, val_dates) in enumerate(folds):
        f_train = cv_df[cv_df['日期'].isin(train_dates)]
        f_val = cv_df[cv_df['日期'].isin(val_dates)]

        X_tr = np.nan_to_num(f_train[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr = f_train['relevance'].values
        X_v = np.nan_to_num(f_val[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_v_cont = f_val['label'].values
        y_v_rel = f_val['relevance'].values
        val_dt = f_val['日期'].values

        model = train_fold(X_tr, y_tr, f_train['日期'].values,
                           X_v, y_v_rel, val_dt, n_rounds=3000)

        preds = model.predict(X_v)
        wr, m = eval_competition(preds, y_v_cont, val_dt)
        print(f"  Fold {i}: {f_train['日期'].min().date()}~{f_train['日期'].max().date()} "
              f"| {f_val['日期'].min().date()}~{f_val['日期'].max().date()} "
              f"| WR={m['wr']:.6f} Comp={m['comp']:.6f} iter={model.best_iteration}")
        cv_results.append({'fold': i, 'wr': m['wr'], 'comp': m['comp'],
                           'best_iter': model.best_iteration})

    avg_iter = int(np.mean([r['best_iter'] for r in cv_results]))
    avg_wr = np.mean([r['wr'] for r in cv_results])
    avg_comp = np.mean([r['comp'] for r in cv_results])
    print(f"\n  CV avg: WR={avg_wr:.6f} Comp={avg_comp:.6f} avg_iter={avg_iter}")

    # ── Train final model on CV data only (NO holdout) ──
    print(f"\n── Final model (cv data only, {avg_iter} rounds) ──")
    X_cv = np.nan_to_num(cv_df[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_cv = cv_df['relevance'].values
    dtrain_final = lgb.Dataset(X_cv, label=y_cv, group=get_groups(cv_df['日期'].values))
    final_model = lgb.train(BASE_PARAMS, dtrain_final, num_boost_round=avg_iter)

    # ── Evaluate on holdout (unseen, never used for anything) ──
    X_ho = np.nan_to_num(holdout_df[feature_cols].values, nan=0, posinf=0, neginf=0)
    ho_preds = final_model.predict(X_ho)
    ho_wr, ho_m = eval_competition(ho_preds, holdout_df['label'].values,
                                    holdout_df['日期'].values)
    print(f"  Holdout (unseen): WR={ho_m['wr']:.6f} Comp={ho_m['comp']:.6f}")

    # ── Save ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model.save_model(os.path.join(MODEL_DIR, 'best.txt'))
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)
    report = {
        'seed': SEED, 'objective': 'lambdarank', 'n_jobs': 1, 'deterministic': True,
        'purge_days': PURGE_DAYS, 'cv_results': cv_results,
        'cv_avg_wr': float(avg_wr), 'cv_avg_comp': float(avg_comp),
        'holdout_wr': float(ho_m['wr']), 'holdout_comp': float(ho_m['comp']),
        'final_iter': avg_iter,
    }
    with open(os.path.join(MODEL_DIR, 'train_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # ── Predict ──
    print(f"\n── Prediction ──")
    predict_top5(final_model, feature_cols)
    print("\nDone.")
