import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# [1] 경로 및 앱 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

# 한글 폰트 설정 (Mac 환경 반영)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def train_final_price_model():
    app = create_app()
    with app.app_context():
        # --- [1. 데이터 로드 및 1kg 단가 전처리] ---
        print("🔍 1. 데이터 통합 로드 (가격 + 기상)...")
        query = """
                SELECT P.*, W.AVG_TEMP, W.SUNSHINE, W.RAIN, W.HUMID
                FROM KAMIS_TOMATO_PRICE P
                         JOIN WEATHER_INDEX W ON P.PRICE_DATE = W.W_DATE
                ORDER BY P.PRICE_DATE ASC
                """
        df = pd.read_sql(query, db.engine)
        df.columns = [col.upper() for col in df.columns]
        df['PRICE_DATE'] = pd.to_datetime(df['PRICE_DATE'])

        print("🔍 2. 고도화 피처 생성 및 단위 보정...")
        groups = df.groupby(['MARKET_NAME', 'GRADE', 'UNIT_KG'])

        # 시계열 및 파생 피처 생성
        df['MONTH'] = df['PRICE_DATE'].dt.month
        df['WEEK'] = df['PRICE_DATE'].dt.isocalendar().week.astype(int)

        # 1kg 단가 기준 이동평균 및 변동폭
        df['PRICE_MA_3D'] = groups['PRICE_PER_KG'].transform(lambda x: x.rolling(window=3).mean())
        df['PRICE_DIFF'] = groups['PRICE_PER_KG'].diff()

        # 타겟 설정: 7일 뒤 1kg 단가
        df['TARGET_PER_KG'] = groups['PRICE_PER_KG'].shift(-7)
        df['PREV_PER_KG_1D'] = groups['PRICE_PER_KG'].shift(1)

        # 기상 시차 변수 (7일 전 기온)
        df['TEMP_LAG7'] = df['AVG_TEMP'].shift(7)

        # 학습용 데이터셋 정제
        df_ml = df.dropna(subset=['TARGET_PER_KG', 'PREV_PER_KG_1D', 'PRICE_MA_3D', 'TEMP_LAG7']).copy()

        features = [
            'GRADE_SCORE', 'UNIT_KG', 'MONTH', 'WEEK',
            'PREV_PER_KG_1D', 'PRICE_MA_3D', 'PRICE_DIFF',
            'AVG_TEMP', 'SUNSHINE', 'TEMP_LAG7'
        ]
        X = df_ml[features]
        y = df_ml['TARGET_PER_KG']

        # --- [2. 모델 학습] ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"🚀 3. 모델 학습 시작 (표준 단위: 1kg, 데이터 수: {len(df_ml)})")

        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # --- [3. 성능 검증] ---
        y_pred_per_kg = model.predict(X_test)
        mae_per_kg = mean_absolute_error(y_test, y_pred_per_kg)
        r2 = r2_score(y_test, y_pred_per_kg)

        print("\n" + "=" * 60)
        print(f"  [최종 고도화 검증 결과]")
        print(f"  - R2 Score: {r2:.4f}")
        print(f"  - 1kg당 평균 오차(MAE): {mae_per_kg:.2f}원")
        print("-" * 60)
        print(f"   실전 체감 오차 (상자 규격별):")
        print(f"     5kg 상자 예측 시: ±{mae_per_kg * 5:.0f}원 내외")
        print(f"     10kg 상자 예측 시: ±{mae_per_kg * 10:.0f}원 내외")
        print("=" * 60)

        # --- [4. 모델 및 중요도 이미지 저장] ---
        # 기존 컨벤션에 맞춘 경로 및 파일명 설정
        model_dir = os.path.join(project_root, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # 1) 모델 저장 (.joblib 확장자 통일)
        model_save_path = os.path.join(model_dir, 'tomato_price_model.joblib')
        joblib.dump(model, model_save_path)
        print(f"모델 엔진 저장 완료: {model_save_path}")

        # 2) 시각화 및 이미지 저장 (_importance.png 규칙 통일)
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # 피처 중요도 차트
        pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh', ax=ax[0], color='skyblue')
        ax[0].set_title('피처 중요도 (1kg 단가 기준)')

        # 예측 산점도 차트
        ax[1].scatter(y_test, y_pred_per_kg, alpha=0.3, color='purple')
        ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax[1].set_title(f'예측 정확도 (R2: {r2:.4f})')

        plt.tight_layout()

        plot_save_path = os.path.join(model_dir, 'tomato_price_importance.png')
        plt.savefig(plot_save_path)
        print(f"분석 그래프 저장 완료: {plot_save_path}")

        plt.show()


if __name__ == "__main__":
    train_final_price_model()