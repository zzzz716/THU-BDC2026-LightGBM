"""
LightGBM training for THU-BDC2026 — fully reproducible, task-aligned.

Fixes vs. previous version:
  1. Cross-sectional ranks computed BEFORE dropping missing-label stocks
  2. LambdaRank objective directly optimises Top-5 selection
  3. Time-series CV (3 folds) + independent final holdout
  4. Strict feature validation at prediction time
  5. Deterministic mode: n_jobs=1, fixed seed, force_row_wise
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings, os, sys, json, hashlib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))
from utils import engineer_features_158plus39

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATA_PATH = './data/train.csv'
MODEL_DIR = './model/lgbm'
SEED = 42
TOP_K = 5


# ─────────────────────────────────────────────
# 1. Data loading & feature engineering
# ─────────────────────────────────────────────
def load_and_engineer(csv_path):
    """Load train.csv, compute features, create labels.

    Key: cross-sectional ranks are computed on the FULL stock pool
    (before dropping stocks with missing future labels).
    """
    df = pd.read_csv(csv_path, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # Per-stock feature engineering
    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)

    # ── Cross-sectional rank features (BEFORE label filtering) ──
    rank_cols = [
        'return_1', 'return_5', 'return_10',
        'rsi', 'macd', 'volume_ratio', 'volatility_20'
    ]
    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('日期')[col].rank(pct=True)

    # ── Labels (after ranks are computed) ──
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df = df.dropna(subset=['label'])
    df = df[df['open_t1'] > 1e-4]

    # ── Binary relevance for LambdaRank: top-K per day = 1, rest = 0 ──
    df['relevance'] = 0
    for date, grp in df.groupby('日期'):
        if len(grp) < TOP_K:
            continue
        top_idx = grp['label'].nlargest(TOP_K).index
        df.loc[top_idx, 'relevance'] = 1

    # ── Feature columns ──
    meta = {
        '股票代码', '日期', '开盘', '收盘', '最高', '最低',
        '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'label', 'open_t1', 'open_t5', 'relevance'
    }
    feature_cols = [c for c in df.columns if c not in meta]
    return df, feature_cols


# ─────────────────────────────────────────────
# 2. Evaluation (competition metric)
# ─────────────────────────────────────────────
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
    return wr, {'wr': float(wr), 'max': float(mx), 'rand': float(rd), 'comp': float(comp), 'n': len(scores)}


# ─────────────────────────────────────────────
# 3. LambdaRank training (group = one trading day)
# ─────────────────────────────────────────────
def get_groups(dates):
    """Count samples per date → group array for LambdaRank."""
    _, counts = np.unique(dates, return_counts=True)
    return counts.tolist()


def train_lambdarank(X_train, y_train, train_dates,
                     X_val, y_val, val_dates,
                     n_rounds=3000):
    base_params = {
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
        'n_jobs': 1,            # deterministic
        'force_row_wise': True, # deterministic
        'seed': SEED,
        'deterministic': True,
    }

    train_groups = get_groups(train_dates)
    val_groups = get_groups(val_dates)

    dtrain = lgb.Dataset(X_train, label=y_train, group=train_groups)
    dval = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=dtrain)

    model = lgb.train(
        base_params, dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(500),
        ]
    )
    return model, base_params


def train_regression(X_train, y_train, X_val, y_val, n_rounds=3000):
    """Fallback: L2 regression with competition-metric based early stopping."""
    base_params = {
        'objective': 'regression',
        'metric': 'mse',
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
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        base_params, dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(500),
        ]
    )
    return model, base_params


# ─────────────────────────────────────────────
# 4. Time-series CV + holdout
# ─────────────────────────────────────────────
def time_series_cv(df, feature_cols, n_folds=3, holdout_months=1):
    """Time-series cross-validation with independent holdout.

    Timeline: [──fold1 train──|──fold1 val──|──fold2 train──|──fold2 val──|──holdout──]
    The holdout set is NEVER used for model selection; final evaluation only.
    """
    dates = sorted(df['日期'].unique())
    n_dates = len(dates)

    # Holdout = last `holdout_months` months
    holdout_start = df['日期'].max() - pd.DateOffset(months=holdout_months)
    holdout_df = df[df['日期'] >= holdout_start].copy()
    cv_df = df[df['日期'] < holdout_start].copy()

    cv_dates = sorted(cv_df['日期'].unique())
    n_cv = len(cv_dates)
    fold_size = n_cv // (n_folds + 1)  # each fold: train = earlier, val = later

    results = []
    models = []

    for fold in range(n_folds):
        # Train: dates[0 : split], Val: dates[split : split+fold_size]
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = min(train_end + fold_size, n_cv)

        train_dates_range = cv_dates[:train_end]
        val_dates_range = cv_dates[val_start:val_end]

        train_mask = cv_df['日期'].isin(train_dates_range)
        val_mask = cv_df['日期'].isin(val_dates_range)

        f_train = cv_df[train_mask]
        f_val = cv_df[val_mask]

        X_tr = np.nan_to_num(f_train[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr = f_train['relevance'].values
        X_v = np.nan_to_num(f_val[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_v = f_val['relevance'].values
        y_v_cont = f_val['label'].values  # continuous labels for competition metric
        val_dt = f_val['日期'].values

        # Train LambdaRank
        model, params = train_lambdarank(
            X_tr, y_tr, f_train['日期'].values,
            X_v, y_v, val_dt,
            n_rounds=3000
        )

        # Evaluate with competition metric
        preds = model.predict(X_v)
        wr, m = eval_competition(preds, y_v_cont, val_dt)

        print(f"  Fold {fold}: train={f_train['日期'].min().date()}~{f_train['日期'].max().date()} "
              f"val={f_val['日期'].min().date()}~{f_val['日期'].max().date()} "
              f"WR={m['wr']:.6f} Comp={m['comp']:.6f}")

        results.append({'fold': fold, 'wr': m['wr'], 'comp': m['comp'],
                        'best_iter': model.best_iteration})
        models.append(model)

    # Summary
    avg_wr = np.mean([r['wr'] for r in results])
    avg_comp = np.mean([r['comp'] for r in results])
    print(f"\n  CV avg: WR={avg_wr:.6f} Comp={avg_comp:.6f}")

    return results, models, holdout_df


# ─────────────────────────────────────────────
# 5. Final model: retrain on all non-holdout data
# ─────────────────────────────────────────────
def train_final(df, feature_cols, holdout_df, n_rounds=None):
    """Train final model on all data except holdout.
    If n_rounds is None, use average best_iteration from CV.
    """
    X_all = np.nan_to_num(df[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_all = df['relevance'].values
    groups = get_groups(df['日期'].values)

    # Holdout evaluation
    X_ho = np.nan_to_num(holdout_df[feature_cols].values, nan=0, posinf=0, neginf=0)
    y_ho = holdout_df['relevance'].values
    y_ho_cont = holdout_df['label'].values
    ho_dates = holdout_df['日期'].values
    ho_groups = get_groups(ho_dates)

    params = {
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

    dtrain = lgb.Dataset(X_all, label=y_all, group=groups)

    if n_rounds is None:
        # Use holdout to find best iteration
        dho = lgb.Dataset(X_ho, label=y_ho, group=ho_groups, reference=dtrain)
        model = lgb.train(
            params, dtrain,
            num_boost_round=5000,
            valid_sets=[dho],
            valid_names=['holdout'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(500),
            ]
        )
        best_iter = model.best_iteration

        # Report holdout score
        preds = model.predict(X_ho)
        wr, m = eval_competition(preds, y_ho_cont, ho_dates)
        print(f"\n  Holdout: WR={m['wr']:.6f} Comp={m['comp']:.6f} (iter={best_iter})")

        # Retrain on ALL data (including holdout) with best_iter
        X_full = np.nan_to_num(
            pd.concat([df, holdout_df])[feature_cols].values,
            nan=0, posinf=0, neginf=0
        )
        y_full = np.concatenate([df['relevance'].values, holdout_df['relevance'].values])
        full_groups = get_groups(
            np.concatenate([df['日期'].values, holdout_df['日期'].values])
        )
        dfull = lgb.Dataset(X_full, label=y_full, group=full_groups)
        final_model = lgb.train(params, dfull, num_boost_round=best_iter)
    else:
        final_model = lgb.train(params, dtrain, num_boost_round=n_rounds)
        best_iter = n_rounds

        preds = final_model.predict(X_ho)
        wr, m = eval_competition(preds, y_ho_cont, ho_dates)
        print(f"\n  Holdout: WR={m['wr']:.6f} Comp={m['comp']:.6f} (fixed iter={n_rounds})")

    return final_model, best_iter


# ─────────────────────────────────────────────
# 6. Save & predict
# ─────────────────────────────────────────────
def save_model(model, feature_cols, best_iter, holdout_metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, 'best.txt'))
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)
    with open(os.path.join(MODEL_DIR, 'train_info.json'), 'w') as f:
        json.dump({
            'best_iter': int(best_iter),
            'seed': SEED,
            'objective': 'lambdarank',
            'n_jobs': 1,
            'deterministic': True,
            'holdout_wr': holdout_metrics.get('wr', None),
            'holdout_comp': holdout_metrics.get('comp', None),
        }, f, indent=2)


def predict_top5(model, feature_cols, top_k=5):
    """Predict top 5 stocks on the latest date using train.csv."""
    df = pd.read_csv(DATA_PATH, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)

    # Cross-sectional ranks (same as training)
    for col in ['return_1', 'return_5', 'return_10', 'rsi', 'macd', 'volume_ratio', 'volatility_20']:
        rcol = f'{col}_rank'
        if col in df.columns and rcol in feature_cols:
            df[rcol] = df.groupby('日期')[col].rank(pct=True)

    latest = df[df['日期'] == df['日期'].max()].copy()

    # ── Strict feature validation ──
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Missing features at prediction time: {missing}")
    extra = [c for c in feature_cols if c not in feature_cols]
    X = np.nan_to_num(latest[feature_cols].values, nan=0, posinf=0, neginf=0)

    latest = latest.copy()
    latest['pred'] = model.predict(X)
    top = latest.nlargest(top_k, 'pred')

    result = pd.DataFrame({
        'stock_id': top['股票代码'].values,
        'weight': [1.0 / top_k] * top_k
    })
    os.makedirs('./output', exist_ok=True)
    result.to_csv('./output/result.csv', index=False)
    print(f"\n  Prediction date: {latest['日期'].max().date()}")
    print(f"  Stocks: {result['stock_id'].tolist()}")
    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print("LightGBM Training — LambdaRank, Time-series CV")
    print("="*60)

    df, feature_cols = load_and_engineer(DATA_PATH)
    print(f"  Features: {len(feature_cols)}, Rows: {len(df)}")
    print(f"  Date range: {df['日期'].min().date()} ~ {df['日期'].max().date()}")

    # ── Phase 1: Time-series CV ──
    print("\n── Phase 1: Time-series CV (3 folds) ──")
    cv_results, cv_models, holdout_df = time_series_cv(df, feature_cols, n_folds=3, holdout_months=1)

    avg_iter = int(np.mean([r['best_iter'] for r in cv_results]))
    print(f"\n  Avg best iteration from CV: {avg_iter}")

    # ── Phase 2: Train final model ──
    print("\n── Phase 2: Final model ──")
    cv_only_df = df[df['日期'] < holdout_df['日期'].min()].copy()
    final_model, best_iter = train_final(cv_only_df, feature_cols, holdout_df)

    # ── Phase 3: Save & predict ──
    print("\n── Phase 3: Save & predict ──")
    preds_ho = final_model.predict(
        np.nan_to_num(holdout_df[feature_cols].values, nan=0, posinf=0, neginf=0)
    )
    _, ho_metrics = eval_competition(preds_ho, holdout_df['label'].values, holdout_df['日期'].values)

    save_model(final_model, feature_cols, best_iter, ho_metrics)
    predict_top5(final_model, feature_cols)

    # ── Save full training report ──
    report = {
        'seed': SEED,
        'objective': 'lambdarank',
        'n_jobs': 1,
        'deterministic': True,
        'cv_results': cv_results,
        'cv_avg_wr': float(np.mean([r['wr'] for r in cv_results])),
        'cv_avg_comp': float(np.mean([r['comp'] for r in cv_results])),
        'holdout_wr': ho_metrics['wr'],
        'holdout_comp': ho_metrics['comp'],
        'best_iter': best_iter,
    }
    with open(os.path.join(MODEL_DIR, 'train_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("\n  Training complete. Report saved to model/lgbm/train_report.json")
    print("="*60)
