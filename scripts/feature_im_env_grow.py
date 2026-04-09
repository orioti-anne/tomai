import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import text
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# [1] 경로 및 앱 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def calculate_vpd(temp, humid):
    """온도와 습도로 포차(VPD, kPa) 계산"""
    es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (humid / 100)
    return es - ea


def save_best_growth_results(model, features, score):
    """학습된 모델과 중요도 그래프를 models 폴더에 저장"""
    model_dir = os.path.join(project_root, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. 모델 저장 (.joblib)
    model_path = os.path.join(model_dir, 'env_growth_model.joblib')
    model_data = {
        'model': model,
        'features': features,
        'r2_score': score
    }
    joblib.dump(model_data, model_path)
    print(f"\n[모델 저장 완료]: {model_path}")

    # 2. 중요도 그래프 저장 (.png)
    plt.figure(figsize=(10, 8))
    pd.Series(model.feature_importances_, index=features).sort_values(ascending=True).plot(kind='barh', color='navy')
    plt.title(f'최종 생육 모델 피처별 중요도 (R2: {score:.4f})')
    plt.tight_layout()

    img_save_path = os.path.join(model_dir, 'env_growth_importance.png')
    plt.savefig(img_save_path)
    print(f"[이미지 저장 완료]: {img_save_path}")

    return img_save_path


def run_optimized_diet_prediction():
    app = create_app()
    with app.app_context():
        print("1. 핵심 환경 지표 추출 중...")
        env_query = text("""
                         SELECT CULT_ID,
                                MEASURE_DATE,
                                AVG(IN_TEMP)                              as AVG_TEMP,
                                MAX(IN_TEMP)                              as MAX_TEMP,
                                MIN(IN_TEMP)                              as MIN_TEMP,
                                AVG(IN_HUMIDITY)                          as AVG_HUMID,
                                AVG(IN_CO2)                               as AVG_CO2,
                                SUM(OUT_SOLAR_RAD)                        as DAILY_SOLAR,
                                COUNT(CASE WHEN IN_TEMP >= 30 THEN 1 END) as HIGH_TEMP_HOURS
                         FROM ENV_CLEANED
                         GROUP BY CULT_ID, MEASURE_DATE
                         """)
        df_env_daily = pd.read_sql(env_query, db.engine)
        df_env_daily.columns = [col.upper() for col in df_env_daily.columns]
        df_env_daily['MEASURE_DATE'] = pd.to_datetime(df_env_daily['MEASURE_DATE'])

        print("2. 생육 데이터 정제 및 RGR 계산 중...")
        grow_query = text(
            "SELECT CULT_ID, INSPECT_DATE, PLANT_HEIGHT, LEAF_COUNT, GROWTH_DAYS as DAP FROM GROW_SUMMARY")
        df_grow = pd.read_sql(grow_query, db.engine)
        df_grow.columns = [col.upper() for col in df_grow.columns]
        df_grow['INSPECT_DATE'] = pd.to_datetime(df_grow['INSPECT_DATE'])

        df_grow = df_grow[df_grow['PLANT_HEIGHT'] > 0].sort_values(['CULT_ID', 'INSPECT_DATE'])
        df_grow['PREV_DATE'] = df_grow.groupby('CULT_ID')['INSPECT_DATE'].shift(1)
        df_grow['PREV_HEIGHT'] = df_grow.groupby('CULT_ID')['PLANT_HEIGHT'].shift(1)
        df_grow['DAYS_DIFF'] = (df_grow['INSPECT_DATE'] - df_grow['PREV_DATE']).dt.days

        df_grow = df_grow[(df_grow['DAYS_DIFF'] > 0) & (df_grow['PREV_HEIGHT'] > 0)].copy()
        df_grow['RGR_HEIGHT'] = (np.log(df_grow['PLANT_HEIGHT']) - np.log(df_grow['PREV_HEIGHT'])) / df_grow[
            'DAYS_DIFF']
        df_grow = df_grow[df_grow['RGR_HEIGHT'].between(0, 0.4)]

        print("3. 핵심 피처 선별 및 복합 지표 생성 중...")

        def get_diet_features(row):
            mask = (df_env_daily['CULT_ID'] == row['CULT_ID']) & \
                   (df_env_daily['MEASURE_DATE'] > row['PREV_DATE']) & \
                   (df_env_daily['MEASURE_DATE'] <= row['INSPECT_DATE'])
            p_data = df_env_daily.loc[mask]
            if p_data.empty: return pd.Series([None] * 6)

            avg_temp = p_data['AVG_TEMP'].mean()
            avg_humid = p_data['AVG_HUMID'].mean()
            vpd = calculate_vpd(avg_temp, avg_humid)
            solar_acc = p_data['DAILY_SOLAR'].sum()

            return pd.Series([
                (p_data['AVG_TEMP'] - 10).clip(lower=0).sum(),
                vpd, solar_acc, vpd * solar_acc,
                p_data['HIGH_TEMP_HOURS'].sum(), p_data['AVG_CO2'].mean()
            ])

        diet_cols = ['PERIOD_GDD', 'PERIOD_VPD', 'PERIOD_SOLAR_ACC', 'VPD_SOLAR_INTERACT', 'HIGH_TEMP_SUM',
                     'PERIOD_CO2_AVG']
        df_grow[diet_cols] = df_grow.apply(get_diet_features, axis=1)
        df_ml = df_grow.dropna(subset=diet_cols).copy()

        features = diet_cols + ['PREV_HEIGHT', 'LEAF_COUNT', 'DAP']
        X = df_ml[features]
        y = df_ml['RGR_HEIGHT']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.9,
                             random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        score = r2_score(y_test, model.predict(X_test))
        print("-" * 40)
        print(f"R2 Score: {score:.4f}")
        print(f"MAE: {mean_absolute_error(y_test, model.predict(X_test)):.6f}")

        save_best_growth_results(model, features, score)
        plt.show()


if __name__ == "__main__":
    run_optimized_diet_prediction()