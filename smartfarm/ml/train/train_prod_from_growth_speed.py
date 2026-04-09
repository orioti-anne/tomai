import sys
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import text
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))

# 예: /smartfarm/ml/train/train_prod_from_growth_speed.py
# project_root => /smartfarm
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def get_model_dir():
    """모델 저장 디렉터리 반환"""
    model_dir = os.path.join(project_root, 'ml', 'models')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_pure_growth_model(
    model,
    features,
    score,
    mae,
    train_rows,
    test_rows,
    importances,
    target_desc='7행 뒤 수확량 (TARGET_QTY = shift(-7))'
):
    """학습된 순수 생육 모델과 메타정보를 joblib로 저장"""
    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, 'prod_growth_model.joblib')

    model_data = {
        'model': model,
        'features': features,
        'r2_score': float(score),
        'mae': float(mae),
        'train_rows': int(train_rows),
        'test_rows': int(test_rows),
        'feature_importances': {k: float(v) for k, v in importances.items()},
        'target_desc': target_desc,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    joblib.dump(model_data, model_path)
    print(f"\n[모델 저장 완료]: {model_path}")
    return model_dir


def save_feature_importance_chart(model, features, score, mae, train_rows, test_rows, model_dir):
    """피처 중요도 차트 저장 + 성능 요약 표시"""
    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(15, 9))
    bars = ax.barh(importances.index, importances.values)

    ax.set_title('순수 생육 지표 기반 수확량 예측 중요도', fontsize=14)
    ax.set_xlabel('Importance Score')

    # 막대 끝에 중요도 수치 표시
    max_val = importances.max() if len(importances) > 0 else 0
    offset = max_val * 0.01 if max_val > 0 else 0.001

    for bar, value in zip(bars, importances.values):
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.4f}',
            va='center',
            fontsize=9
        )

    # 요약 박스
    summary_text = (
        f'R2 Score: {score:.4f}\n'
        f'MAE: {mae:.2f} kg\n'
        f'Train Rows: {train_rows:,}\n'
        f'Test Rows: {test_rows:,}\n'
        f'Feature Count: {len(features)}'
    )

    fig.text(
        0.74,
        0.18,
        summary_text,
        fontsize=10,
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='gray'
        )
    )

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    img_save_path = os.path.join(model_dir, 'prod_growth_importance.png')
    plt.savefig(img_save_path, dpi=150, bbox_inches='tight')
    print(f"[이미지 저장 완료]: {img_save_path}")

    plt.show()

    return importances.to_dict()


def run_pure_growth_analysis():
    # scheduler 옵션 지원하면 끄고, 아니면 기존 방식으로 fallback
    try:
        app = create_app(enable_scheduler=False)
    except TypeError:
        app = create_app()

    with app.app_context():
        print("1. 수확량 지표를 제외한 순수 생육 데이터 로드 중...")

        query = text("""
            SELECT
                G.CULT_ID,
                G.GROWTH_DAYS AS DAP,
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
                P.TOTAL_QUANTITY AS QTY
            FROM GROW_SUMMARY G
            JOIN PRODUCTS P
              ON G.CULT_ID = P.CULT_ID
             AND G.GROWTH_DAYS = (
                 P.PRODUCTION_DATE - (
                     SELECT PLANTING_DATE
                     FROM CULTIVATIONS
                     WHERE CULT_ID = P.CULT_ID
                 )
             )
            WHERE G.GROWTH_DAYS BETWEEN 0 AND 350
        """)

        df = pd.read_sql(query, db.engine)
        df.columns = [col.upper() for col in df.columns]
        df = df.sort_values(['CULT_ID', 'DAP']).reset_index(drop=True)

        print(f" - 로드 완료: {len(df)} rows")

        # 타겟 설정: 7행 뒤 수확량
        # 주의: 실제 7일 뒤가 아니라 7행 뒤 기준
        df['TARGET_QTY'] = df.groupby('CULT_ID')['QTY'].shift(-7)

        pure_growth_features = [
            'DAP',
            'PLANT_HEIGHT',
            'LEAF_COUNT',
            'GROWTH_LENGTH',
            'LEAF_LENGTH',
            'LEAF_WIDTH',
            'BRANCH_WIDTH',
            'CLUSTER_HEIGHT',
            'CLUSTER_NUM',
            'FLOWERS_PER_CLUSTER',
            'BLOOMING_PER_CLUSTER',
            'FRUITS_PER_CLUSTER'
        ]

        df_ml = df.dropna(subset=['TARGET_QTY'] + pure_growth_features).copy()
        print(f" - 학습 대상 row 수: {len(df_ml)}")

        X = df_ml[pure_growth_features]
        y = df_ml['TARGET_QTY']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("2. 순수 생육 기반 모델 학습 시작...")

        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=8,
            random_state=42
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        train_rows = len(X_train)
        test_rows = len(X_test)

        print("-" * 40)
        print(f"R2 Score: {score:.4f}")
        print(f"MAE: {mae:.2f} kg")

        model_dir = get_model_dir()

        importances = save_feature_importance_chart(
            model=model,
            features=pure_growth_features,
            score=score,
            mae=mae,
            train_rows=train_rows,
            test_rows=test_rows,
            model_dir=model_dir
        )

        save_pure_growth_model(
            model=model,
            features=pure_growth_features,
            score=score,
            mae=mae,
            train_rows=train_rows,
            test_rows=test_rows,
            importances=importances,
            target_desc='7행 뒤 수확량 (TARGET_QTY = shift(-7))'
        )


if __name__ == "__main__":
    run_pure_growth_analysis()