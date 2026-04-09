import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import text
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def save_pure_growth_model(model, features, score):
    """학습된 순수 생육 모델과 피처 리스트를 저장"""
    model_dir = os.path.join(project_root, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'prod_growth_model.joblib')
    model_data = {
        'model': model,
        'features': features,
        'r2_score': score
    }
    joblib.dump(model_data, model_path)
    print(f"\n[모델 저장 완료]: {model_path}")
    return model_dir


def run_pure_growth_analysis():
    app = create_app()
    with app.app_context():
        print("1. 수확량 지표를 제외한 순수 생육 데이터 로드 중...")
        query = text("""
                     SELECT G.CULT_ID,
                            G.GROWTH_DAYS    as DAP,
                            G.PLANT_HEIGHT,
                            G.LEAF_COUNT,
                            G.GROWTH_LENGTH,
                            G.LEAF_LENGTH,
                            G.LEAF_WIDTH,
                            G.BRANCH_WIDTH,
                            G.CLUSTER_HEIGHT,
                            G.CLUSTER_NUM,
                            G.FLOWERS_PER_CLUSTER,
                            G.BLOOMING_PER_CLUSTER,
                            G.FRUITS_PER_CLUSTER,
                            P.TOTAL_QUANTITY as QTY
                     FROM GROW_SUMMARY G
                              JOIN PRODUCTS P ON G.CULT_ID = P.CULT_ID
                         AND G.GROWTH_DAYS =
                             (P.PRODUCTION_DATE - (SELECT PLANTING_DATE FROM CULTIVATIONS WHERE CULT_ID = P.CULT_ID))
                     WHERE G.GROWTH_DAYS BETWEEN 0 AND 350
                     """)
        df = pd.read_sql(query, db.engine)
        df.columns = [col.upper() for col in df.columns]
        df = df.sort_values(['CULT_ID', 'DAP'])

        # 타겟 설정: 7일 뒤 수확량
        df['TARGET_QTY'] = df.groupby('CULT_ID')['QTY'].shift(-7)

        # 수확량 관련 피처 배제
        pure_growth_features = [
            'DAP', 'PLANT_HEIGHT', 'LEAF_COUNT', 'GROWTH_LENGTH',
            'LEAF_LENGTH', 'LEAF_WIDTH', 'BRANCH_WIDTH',
            'CLUSTER_HEIGHT', 'CLUSTER_NUM', 'FLOWERS_PER_CLUSTER',
            'BLOOMING_PER_CLUSTER', 'FRUITS_PER_CLUSTER'
        ]

        df_ml = df.dropna(subset=['TARGET_QTY'] + pure_growth_features).copy()
        X = df_ml[pure_growth_features]
        y = df_ml['TARGET_QTY']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 학습
        print("2. 순수 생육 기반 모델 학습 시작...")
        model = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=8, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # 결과 분석
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        print("-" * 40)
        print(f" R2 Score: {score:.4f}")
        print(f" MAE: {mean_absolute_error(y_test, y_pred):.2f} kg")

        # 모델 및 중요도 이미지 저장
        model_dir = save_pure_growth_model(model, pure_growth_features, score)

        plt.figure(figsize=(12, 8))
        importances = pd.Series(model.feature_importances_, index=pure_growth_features).sort_values(ascending=True)
        importances.plot(kind='barh', color='salmon')
        plt.title('순수 생육 지표 기반 수확량 예측 중요도')
        plt.xlabel('Importance Score')

        img_save_path = os.path.join(model_dir, 'prod_growth_importance.png')
        plt.tight_layout()
        plt.savefig(img_save_path)
        print(f"[이미지 저장 완료]: {img_save_path}")
        plt.show()


if __name__ == "__main__":
    run_pure_growth_analysis()