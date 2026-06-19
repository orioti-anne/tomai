"""
Yield Model V5
- 타겟: 스냅샷일(100/125/150/180/220/260)까지의 누적 수확량(kg/평)
- cult_duration = 시나리오 일수 (v4는 전체 재배기간)
- 환경·생육 피처: 스냅샷일 기준으로 슬라이스
- 저장: v5_yield_pipeline.joblib
"""
import os, sys, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/macmini/tomAI')

from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

MODEL_DIR = '/Users/macmini/tomAI/smartfarm/ml/models'
SNAPSHOT_DAYS = [100, 125, 150, 180, 220, 260]


def load_data():
    from smartfarm import create_app, db
    app = create_app()
    with app.app_context():

        base = pd.read_sql(text("""
            SELECT c.cult_id, c.survey_year, c.planting_date, c.planting_area,
                   c.planting_density, c.crop_cycle, c.house_type, c.house_form,
                   c.item_variety,
                   f.region_l1, f.region_l2, f.total_area,
                   ps.cult_end_date,
                   (ps.cult_end_date - c.planting_date)::int AS actual_duration
            FROM cultivations c
            JOIN farms f ON c.farm_id = f.farm_id
            JOIN prod_summary ps ON c.cult_id = ps.cult_id
            WHERE ps.cult_total_quantity > 0
              AND c.planting_date IS NOT NULL
              AND ps.cult_end_date IS NOT NULL
              AND EXISTS (SELECT 1 FROM env_summary e WHERE e.cult_id = c.cult_id)
        """), db.engine)

        cult_ids = tuple(base['cult_id'].tolist())

        products = pd.read_sql(text(f"""
            SELECT cult_id, production_date, total_quantity
            FROM products
            WHERE cult_id IN {cult_ids} AND total_quantity > 0
        """), db.engine)

        env = pd.read_sql(text(f"""
            SELECT cult_id, measure_date, daily_in_temp, daily_in_humidity,
                   daily_in_co2, daily_acc_solar, acc_temp, acc_solar,
                   daily_rain_detection
            FROM env_summary WHERE cult_id IN {cult_ids}
        """), db.engine)

        grow = pd.read_sql(text(f"""
            SELECT cult_id, inspect_date, plant_height, growth_length,
                   leaf_count, cluster_num, fruits_per_cluster, branch_width,
                   flowers_per_cluster, blooming_per_cluster
            FROM grow_summary WHERE cult_id IN {cult_ids}
        """), db.engine)

    base['planting_date']  = pd.to_datetime(base['planting_date'])
    base['cult_end_date']  = pd.to_datetime(base['cult_end_date'])
    products['production_date'] = pd.to_datetime(products['production_date'])
    env['measure_date']    = pd.to_datetime(env['measure_date'])
    grow['inspect_date']   = pd.to_datetime(grow['inspect_date'])

    print(f"[로드] base={len(base)}, products={len(products):,}, env={len(env):,}, grow={len(grow):,}")
    return base, products, env, grow


def env_features_slice(cult_id, planting_date, snapshot_date, env_all):
    df = env_all[
        (env_all['cult_id'] == cult_id) &
        (env_all['measure_date'] <= snapshot_date)
    ].copy()
    if len(df) < 5:
        return {}

    df = df.sort_values('measure_date')
    total_days = (df['measure_date'].max() - df['measure_date'].min()).days + 1

    def stats(col):
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) == 0:
            return {f'{col}_mean': np.nan, f'{col}_std': np.nan,
                    f'{col}_min': np.nan,  f'{col}_max': np.nan}
        return {f'{col}_mean': s.mean(), f'{col}_std': s.std(),
                f'{col}_min': s.min(),   f'{col}_max': s.max()}

    feats = {'env_total_days': total_days}
    for col in ['daily_in_temp', 'daily_in_humidity', 'daily_in_co2', 'daily_acc_solar']:
        feats.update(stats(col))

    temp = pd.to_numeric(df['daily_in_temp'], errors='coerce')
    feats['heat_days'] = int((temp > 30).sum())
    feats['cold_days'] = int((temp < 10).sum())
    feats['rain_days'] = int(pd.to_numeric(df['daily_rain_detection'], errors='coerce').fillna(0).sum())

    last = df.dropna(subset=['acc_temp']).iloc[-1] if df['acc_temp'].notna().any() else None
    feats['acc_temp_final']  = float(last['acc_temp'])  if last is not None else np.nan
    feats['acc_solar_final'] = float(last['acc_solar']) if last is not None else np.nan

    if total_days > 30:
        late = df[df['measure_date'] >= df['measure_date'].max() - pd.Timedelta(days=30)]
        t_late = pd.to_numeric(late['daily_in_temp'], errors='coerce').dropna()
        feats['late30_temp_mean'] = t_late.mean() if len(t_late) > 0 else np.nan
    else:
        feats['late30_temp_mean'] = np.nan

    return feats


def grow_features_slice(cult_id, snapshot_date, grow_all):
    df = grow_all[
        (grow_all['cult_id'] == cult_id) &
        (grow_all['inspect_date'] <= snapshot_date)
    ].copy()
    if len(df) == 0:
        return {}

    num_cols = ['plant_height', 'growth_length', 'leaf_count', 'cluster_num',
                'fruits_per_cluster', 'branch_width', 'flowers_per_cluster', 'blooming_per_cluster']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    daily = df.groupby('inspect_date')[num_cols].mean().reset_index().sort_values('inspect_date')
    n_dates = len(daily)
    feats = {'grow_n_dates': n_dates}
    feats['grow_span_days'] = (daily['inspect_date'].iloc[-1] - daily['inspect_date'].iloc[0]).days if n_dates >= 2 else 0

    for col in num_cols:
        s = daily[col].dropna()
        if len(s) == 0:
            feats.update({f'{col}_mean': np.nan, f'{col}_std': np.nan,
                          f'{col}_final': np.nan, f'{col}_slope': np.nan})
            continue
        feats[f'{col}_mean']  = s.mean()
        feats[f'{col}_std']   = s.std()
        feats[f'{col}_final'] = float(daily[col].dropna().iloc[-1])
        valid = daily[['inspect_date', col]].dropna()
        if len(valid) >= 3:
            x = (valid['inspect_date'] - valid['inspect_date'].min()).dt.days.values
            feats[f'{col}_slope'] = np.polyfit(x, valid[col].values, 1)[0]
        else:
            feats[f'{col}_slope'] = np.nan

    return feats


def _season(m):
    return {12:'winter',1:'winter',2:'winter',
            3:'spring',4:'spring',5:'spring',
            6:'summer',7:'summer',8:'summer',
            9:'fall',10:'fall',11:'fall'}.get(m)


def build_features(base, products, env_all, grow_all):
    print("[V5] 스냅샷별 피처 추출 중...")
    rows = []

    prod_by_cult = {cid: grp for cid, grp in products.groupby('cult_id')}

    for _, row in base.iterrows():
        cid           = row['cult_id']
        planting_date = row['planting_date']
        actual_dur    = row['actual_duration']
        area          = float(row['planting_area']) if pd.notna(row['planting_area']) and float(row['planting_area']) > 0 else None
        if area is None:
            continue

        prod_df = prod_by_cult.get(cid, pd.DataFrame())

        for snap_day in SNAPSHOT_DAYS:
            if snap_day > actual_dur:
                continue

            snapshot_date = planting_date + pd.Timedelta(days=snap_day)

            # 누적 수확량 / 면적
            if prod_df.empty:
                continue
            cum_qty = prod_df[prod_df['production_date'] <= snapshot_date]['total_quantity'].sum()
            if cum_qty <= 0:
                continue
            yield_per_area = cum_qty / area

            ef = env_features_slice(cid, planting_date, snapshot_date, env_all)
            gf = grow_features_slice(cid, snapshot_date, grow_all)

            pm = planting_date.month
            total_area = float(row['total_area']) if pd.notna(row['total_area']) and float(row['total_area']) > 0 else None

            feat = {
                'cult_id':          cid,
                'snapshot_day':     snap_day,
                'yield_per_area':   yield_per_area,
                'planting_area':    area,
                'planting_density': float(row['planting_density']) if pd.notna(row['planting_density']) else np.nan,
                'crop_cycle':       float(row['crop_cycle'])       if pd.notna(row['crop_cycle'])       else np.nan,
                'survey_year':      float(row['survey_year'])      if pd.notna(row['survey_year'])      else np.nan,
                'area_ratio':       area / total_area if total_area else np.nan,
                'planting_month':   float(pm),
                'cult_duration':    float(snap_day),   # ← 핵심 변경: 시나리오 일수
                'house_type':       row['house_type'],
                'house_form':       row['house_form'],
                'region_l1':        row['region_l1'],
                'planting_season':  _season(pm),
            }
            feat.update(ef)
            feat.update(gf)
            rows.append(feat)

    df = pd.DataFrame(rows)
    print(f"[V5] 피처 완성: {len(df)}건 (cult {df['cult_id'].nunique()}개 × 스냅샷 {len(SNAPSHOT_DAYS)}개)")
    print(f"스냅샷별 샘플 수:")
    print(df.groupby('snapshot_day').size().to_string())
    return df


def compare_models(X_all, y, num_feats, cat_feats):
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
        ]), num_feats),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_feats),
    ])

    candidates = {
        'XGBoost': XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0),
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            max_features='sqrt', random_state=42),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=42),
        'Ridge': Ridge(alpha=10.0),
    }
    if HAS_LGB:
        candidates['LightGBM'] = LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=5,
            reg_alpha=0.1, random_state=42, verbose=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n[V5] 5-Fold KFold 비교")
    for name, model in candidates.items():
        pipe = Pipeline([('pre', preprocessor), ('model', model)])
        r2_scores  = cross_val_score(pipe, X_all, y, cv=kf, scoring='r2')
        mae_scores = cross_val_score(pipe, X_all, y, cv=kf, scoring='neg_mean_absolute_error')
        r2  = r2_scores.mean()
        mae = -mae_scores.mean()
        results[name] = {'r2': r2, 'mae': mae, 'model': model, 'preprocessor': preprocessor}
        print(f"  {name:20s}  R²={r2:.4f}  MAE={mae:.2f} kg/평")

    best_name = max(results, key=lambda k: results[k]['r2'])
    print(f"\n최적 모델: {best_name}  (R²={results[best_name]['r2']:.4f})")
    return best_name, results, preprocessor


def train_best(best_name, results, preprocessor, X_all, y, num_feats, cat_feats, feat_df):
    best_model = results[best_name]['model']
    pipe = Pipeline([('pre', preprocessor), ('model', best_model)])
    pipe.fit(X_all, y)

    pred = pipe.predict(X_all)
    r2  = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)

    # 스냅샷별 성능
    print("\n[V5] 스냅샷별 in-sample 성능:")
    snap_col = feat_df['snapshot_day'].values
    for snap in SNAPSHOT_DAYS:
        mask = snap_col == snap
        if mask.sum() == 0:
            continue
        r2s  = r2_score(y[mask], pred[mask])
        maes = mean_absolute_error(y[mask], pred[mask])
        print(f"  {snap:3d}일  R²={r2s:.4f}  MAE={maes:.2f} kg/평  (n={mask.sum()})")

    # 피처 중요도
    try:
        ohe_names = (pipe.named_steps['pre']
                     .named_transformers_['cat']
                     .named_steps['ohe']
                     .get_feature_names_out(cat_feats))
        all_feat_names = num_feats + list(ohe_names)
        importances = pipe.named_steps['model'].feature_importances_
        imp_df = pd.DataFrame({'feature': all_feat_names, 'importance': importances})
        imp_df = imp_df.sort_values('importance', ascending=False)
        print("\n피처 중요도 TOP 15:")
        print(imp_df.head(15).to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, 6))
        imp_df.head(15).sort_values('importance').plot.barh(x='feature', y='importance', ax=ax)
        ax.set_title(f'{best_name} 피처 중요도 TOP 15\nR²={r2:.4f}, MAE={mae:.2f} kg/평')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'v5_yield_importance.png'), dpi=120)
        plt.close()
    except Exception:
        pass

    # 시각화: 스냅샷별 예측 vs 실제
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, snap in zip(axes.flatten(), SNAPSHOT_DAYS):
        mask = snap_col == snap
        if mask.sum() == 0:
            continue
        ax.scatter(y[mask], pred[mask], alpha=0.6, s=30)
        lim = [0, max(y[mask].max(), pred[mask].max()) + 5]
        ax.plot(lim, lim, 'r--', lw=1)
        r2s  = r2_score(y[mask], pred[mask])
        maes = mean_absolute_error(y[mask], pred[mask])
        ax.set_title(f'+{snap}일  R²={r2s:.3f}  MAE={maes:.1f} kg/평 (n={mask.sum()})')
        ax.set_xlabel('실제'); ax.set_ylabel('예측')
        ax.grid(True, alpha=0.2)

    plt.suptitle('V5 수확량 모델 — 스냅샷별 예측 성능', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'v5_yield_report.png'), dpi=120)
    plt.close()

    bundle = {
        'model':        pipe,
        'features':     num_feats + cat_feats,
        'num_features': num_feats,
        'cat_features': cat_feats,
        'best_algorithm': best_name,
        'snapshot_days': SNAPSHOT_DAYS,
        'loocv_results': {n: {'r2': results[n]['r2'], 'mae': results[n]['mae']} for n in results},
        'metrics': {
            'r2_insample':   r2,
            'mae_insample':  mae,
            'r2_loocv':      results[best_name]['r2'],
            'mae_loocv':     results[best_name]['mae'],
        },
    }
    out = os.path.join(MODEL_DIR, 'v5_yield_pipeline.joblib')
    joblib.dump(bundle, out)

    print(f"\n{'='*55}")
    print(f"[V5] 최적 알고리즘 : {best_name}")
    print(f"[V5] LOOCV  R²    : {results[best_name]['r2']:.4f}")
    print(f"[V5] LOOCV  MAE   : {results[best_name]['mae']:.2f} kg/평")
    print(f"[V5] In-sample R² : {r2:.4f}")
    print(f"[V5] 저장           : {out}")
    print(f"{'='*55}")


if __name__ == '__main__':
    base, products, env_all, grow_all = load_data()
    feat_df = build_features(base, products, env_all, grow_all)

    NUM_FEATURES = [c for c in feat_df.columns
                    if c not in ('cult_id', 'snapshot_day', 'yield_per_area',
                                 'house_type', 'house_form', 'region_l1', 'planting_season')]
    CAT_FEATURES = ['house_type', 'house_form', 'region_l1', 'planting_season']

    all_null = [c for c in NUM_FEATURES if feat_df[c].notna().sum() == 0]
    if all_null:
        print(f"[제거] 전체 null 컬럼: {all_null}")
    NUM_FEATURES = [c for c in NUM_FEATURES if c not in all_null]
    CAT_FEATURES = [c for c in CAT_FEATURES if feat_df[c].notna().sum() > 0]

    # IQR×3 이상치 제거
    y_raw = feat_df['yield_per_area']
    q1, q3 = y_raw.quantile(0.25), y_raw.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    lower = max(0, q1 - 3 * iqr)
    mask = y_raw.between(lower, upper)
    removed = (~mask).sum()
    feat_df = feat_df[mask].reset_index(drop=True)
    print(f"\n이상치 제거: {removed}건 (범위 [{lower:.1f}, {upper:.1f}] kg/평)")

    X_all = feat_df[NUM_FEATURES + CAT_FEATURES]
    y     = feat_df['yield_per_area']

    print(f"수치 피처: {len(NUM_FEATURES)}개 / 카테고리: {len(CAT_FEATURES)}개 / 샘플: {len(y)}건")
    print(f"타겟 yield_per_area: 평균={y.mean():.2f}, std={y.std():.2f}, 범위=[{y.min():.1f}, {y.max():.1f}]")

    best_name, results, preprocessor = compare_models(X_all, y, NUM_FEATURES, CAT_FEATURES)
    train_best(best_name, results, preprocessor, X_all, y, NUM_FEATURES, CAT_FEATURES, feat_df)
