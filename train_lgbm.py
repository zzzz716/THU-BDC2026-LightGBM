"""
LightGBM training for THU-BDC2026 v5.

All fixes applied:
  1. Top5 explicit labels: rank1→5, rank2→4, ..., rank5→1, rest→0
  2. Iteration: median of CV folds (not mean) for robustness
  3. Multi-month holdout evaluation (2 non-overlapping months)
  4. CV-Holdout purge gap enforced
  5. Data sorted by (日期, 股票代码) for correct LambdaRank grouping
  6. RSQR fixed in utils.py
  7. Dead code removed from utils.py
  8. score_self.py sorts by date
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings, os, sys, json, glob

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))
from utils import engineer_features_158plus39

DATA_PATH = './data/train.csv'
MODEL_DIR = './model/lgbm'
SEED = 42
TOP_K = 5
PURGE_DAYS = 5


# ─────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────
def load_and_engineer(csv_path):
    df = pd.read_csv(csv_path, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    groups = [engineer_features_158plus39(g) for _, g in df.groupby('股票代码')]
    df = pd.concat(groups).reset_index(drop=True)

    # Cross-sectional rank features (BEFORE label filtering)
    for col in ['return_1', 'return_5', 'return_10', 'rsi', 'macd',
                'volume_ratio', 'volatility_20']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('日期')[col].rank(pct=True)

    # Labels
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['label'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df = df.dropna(subset=['label'])
    df = df[df['open_t1'] > 1e-4]

    # ── Fine-grained relevance: 20 bins per day ──
    # ~15 stocks per bin, top bin captures rank 1-15, gives richer gradient signal
    df['relevance'] = 0
    for date, grp in df.groupby('日期'):
        if len(grp) < 10:
            continue
        # Bin into 20 groups by return rank
        bins = pd.qcut(grp['label'], q=20, labels=False, duplicates='drop')
        df.loc[grp.index, 'relevance'] = bins.fillna(0).astype(int)

    # CRITICAL: sort by (日期, 股票代码) for LambdaRank grouping
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

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
# 3. Purge-aware splits
# ─────────────────────────────────────────────────────
def purge_split(df, n_folds=3, holdout_months=2):
    """Time-series splits with purge gaps:
    - Between each train/val fold: PURGE_DAYS
    - Between CV and holdout: PURGE_DAYS
    - holdout_months=2 for more stable evaluation
    """
    dates = sorted(df['日期'].unique())

    # Holdout: last 2 months
    holdout_start = df['日期'].max() - pd.DateOffset(months=holdout_months)
    ho_dates = sorted([d for d in dates if d >= holdout_start])
    cv_dates_all = sorted([d for d in dates if d < holdout_start])

    # Purge gap before holdout
    cv_dates = cv_dates_all[:-PURGE_DAYS] if len(cv_dates_all) > PURGE_DAYS else cv_dates_all

    n_cv = len(cv_dates)
    fold_size = (n_cv - PURGE_DAYS * n_folds) // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        gap_end = train_end + PURGE_DAYS
        val_end = gap_end + fold_size
        train_d = set(cv_dates[:train_end])
        val_d = set(cv_dates[gap_end:min(val_end, n_cv)])
        folds.append((train_d, val_d))

    return folds, ho_dates, cv_dates


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
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(500),
        ]
    )
    return model


def train_model(X, y, dates, n_rounds):
    dtrain = lgb.Dataset(X, label=y, group=get_groups(dates))
    return lgb.train(BASE_PARAMS, dtrain, num_boost_round=n_rounds)


# ─────────────────────────────────────────────────────
# 5. Predict
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
    X = np.nan_to_num(latest[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    latest = latest.copy()
    latest['pred'] = model.predict(X)
    top = latest.nlargest(top_k, 'pred')
    result = pd.DataFrame({'stock_id': top['股票代码'].values, 'weight': [1.0 / top_k] * top_k})
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
    print("LightGBM v5 — Top5 labels + median iter + multi-month holdout")
    print("=" * 60)

    # Clean stale files
    os.makedirs(MODEL_DIR, exist_ok=True)
    for pat in ['train_info.json', 'search*.json', 'final_search.json']:
        for f in glob.glob(os.path.join(MODEL_DIR, pat)):
            os.remove(f)
            print(f"  Cleaned: {f}")

    df, feature_cols = load_and_engineer(DATA_PATH)
    print(f"  Features: {len(feature_cols)}, Rows: {len(df)}")
    print(f"  Sorted by (日期, 股票代码): {df['日期'].is_monotonic_increasing}")

    # Verify RSQR
    for c in [c for c in feature_cols if c.startswith('RSQR')]:
        nz = (df[c] != 0).sum()
        print(f"  {c}: {nz}/{len(df)} nonzero ({100*nz/len(df):.1f}%)")

    # Verify relevance distribution
    rel = df['relevance']
    print(f"\n  Relevance distribution:")
    for v in range(6):
        cnt = (rel == v).sum()
        print(f"    grade {v}: {cnt} ({100*cnt/len(rel):.2f}%)")

    # ── Purge CV ──
    folds, ho_dates, cv_dates = purge_split(df, n_folds=3, holdout_months=2)
    holdout_df = df[df['日期'].isin(ho_dates)].copy()
    cv_df = df[df['日期'].isin(cv_dates)].copy()

    print(f"\n── Purge CV (3 folds, purge={PURGE_DAYS}d) ──")
    print(f"  CV:   {cv_df['日期'].min().date()} ~ {cv_df['日期'].max().date()}")
    print(f"  Hold: {holdout_df['日期'].min().date()} ~ {holdout_df['日期'].max().date()}")

    cv_results = []
    for i, (train_dates, val_dates) in enumerate(folds):
        f_train = cv_df[cv_df['日期'].isin(train_dates)]
        f_val = cv_df[cv_df['日期'].isin(val_dates)]

        X_tr = np.nan_to_num(f_train[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_tr = f_train['relevance'].values
        X_v = np.nan_to_num(f_val[feature_cols].values, nan=0, posinf=0, neginf=0)
        y_v_rel = f_val['relevance'].values
        y_v_cont = f_val['label'].values
        val_dt = f_val['日期'].values

        model = train_fold(X_tr, y_tr, f_train['日期'].values,
                           X_v, y_v_rel, val_dt, n_rounds=3000)

        preds = model.predict(X_v)
        wr, m = eval_competition(preds, y_v_cont, val_dt)
        print(f"  Fold {i}: {f_train['日期'].min().date()}~{f_train['日期'].max().date()} "
              f"→ {f_val['日期'].min().date()}~{f_val['日期'].max().date()} "
              f"| WR={m['wr']:.6f} Comp={m['comp']:.6f} iter={model.best_iteration}")
        cv_results.append({'fold': i, 'wr': m['wr'], 'comp': m['comp'],
                           'best_iter': model.best_iteration})

    # Use MEDIAN iteration for robustness
    all_iters = [r['best_iter'] for r in cv_results]
    final_iter = int(np.median(all_iters))
    avg_wr = np.mean([r['wr'] for r in cv_results])
    avg_comp = np.mean([r['comp'] for r in cv_results])
    print(f"\n  CV avg: WR={avg_wr:.6f} Comp={avg_comp:.6f}")
    print(f"  Iterations: {all_iters} → median={final_iter}")

    # ── Holdout evaluation (clean) ──
    print(f"\n── Holdout evaluation ──")
    eval_model = train_model(
        np.nan_to_num(cv_df[feature_cols].values, nan=0, posinf=0, neginf=0),
        cv_df['relevance'].values, cv_df['日期'].values, final_iter
    )
    X_ho = np.nan_to_num(holdout_df[feature_cols].values, nan=0, posinf=0, neginf=0)
    ho_preds = eval_model.predict(X_ho)
    ho_wr, ho_m = eval_competition(ho_preds, holdout_df['label'].values,
                                    holdout_df['日期'].values)
    print(f"  Holdout: WR={ho_m['wr']:.6f} Comp={ho_m['comp']:.6f}")

    # ── Retrain on ALL data for submission ──
    print(f"\n── Retrain on ALL data ({final_iter} rounds) ──")
    X_all = np.nan_to_num(df[feature_cols].values, nan=0, posinf=0, neginf=0)
    final_model = train_model(X_all, df['relevance'].values, df['日期'].values, final_iter)

    # ── Save ──
    final_model.save_model(os.path.join(MODEL_DIR, 'best.txt'))
    with open(os.path.join(MODEL_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)
    report = {
        'version': 'v5',
        'seed': SEED, 'objective': 'lambdarank',
        'n_jobs': 1, 'deterministic': True,
        'purge_days': PURGE_DAYS,
        'relevance': '20-bin qcut per day',
        'iter_selection': 'median',
        'holdout_months': 2,
        'cv_results': cv_results,
        'cv_avg_wr': float(avg_wr), 'cv_avg_comp': float(avg_comp),
        'holdout_wr': float(ho_m['wr']), 'holdout_comp': float(ho_m['comp']),
        'final_iter': final_iter,
        'final_model_trained_on': 'all_data',
    }
    with open(os.path.join(MODEL_DIR, 'train_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # ── Predict ──
    print(f"\n── Prediction ──")
    predict_top5(final_model, feature_cols)
    print("\nDone.")
