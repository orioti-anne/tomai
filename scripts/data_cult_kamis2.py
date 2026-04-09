import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# 프로젝트 루트 등록
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app


def load_data():
    app = create_app()

    with app.app_context():
        query = """
        SELECT
            c.CULT_ID,
            c.FARM_ID,
            c.ITEM,
            c.ITEM_VARIETY,
            c.CROP_CYCLE,
            c.PLANTING_DATE,
            c.PLANTING_AREA,
            c.PLANTING_DENSITY,
            c.HOUSE_TYPE,
            c.HOUSE_FORM,
            c.SURVEY_YEAR,

            f.REGION_L1,
            f.REGION_L2,
            f.TOTAL_AREA AS FARM_TOTAL_AREA,
            f.FIRST_SURVEY_YEAR,

            p.PRODUCTION_DATE,
            p.TOTAL_QUANTITY

        FROM CULTIVATIONS c
        JOIN FARMS f
            ON c.FARM_ID = f.FARM_ID
        JOIN PRODUCTS p
            ON c.CULT_ID = p.CULT_ID
        WHERE p.TOTAL_QUANTITY IS NOT NULL
        """

        df = pd.read_sql(query, db.engine)

    df.columns = [c.upper() for c in df.columns]

    # 날짜 처리
    df["PLANTING_DATE"] = pd.to_datetime(df["PLANTING_DATE"])
    df["PRODUCTION_DATE"] = pd.to_datetime(df["PRODUCTION_DATE"])

    # 파생변수
    df["DAYS_FROM_PLANTING"] = (
        df["PRODUCTION_DATE"] - df["PLANTING_DATE"]
    ).dt.days

    df["PLANTING_MONTH"] = df["PLANTING_DATE"].dt.month

    df["PLANTING_SEASON"] = df["PLANTING_MONTH"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    })

    df = df[
        (df["PLANTING_AREA"].notnull()) &
        (df["PLANTING_AREA"] > 0) &
        (df["TOTAL_QUANTITY"].notnull()) &
        (df["TOTAL_QUANTITY"] > 0)
    ].copy()

    # 면적당 생산량
    df["YIELD_PER_M2"] = (
        df["TOTAL_QUANTITY"] / df["PLANTING_AREA"]
    )

    # 이상치 제거
    q1 = df["YIELD_PER_M2"].quantile(0.01)
    q99 = df["YIELD_PER_M2"].quantile(0.99)

    df = df[
        (df["YIELD_PER_M2"] >= q1) &
        (df["YIELD_PER_M2"] <= q99)
    ].copy()

    return df


def preprocess(df):
    numeric_features = [
        "PLANTING_AREA",
        "PLANTING_DENSITY",
        "CROP_CYCLE",
        "DAYS_FROM_PLANTING",
        "FARM_TOTAL_AREA",
        "FIRST_SURVEY_YEAR",
        "SURVEY_YEAR",
        "PLANTING_MONTH",
    ]

    categorical_features = [
        "ITEM",
        "ITEM_VARIETY",
        "HOUSE_TYPE",
        "HOUSE_FORM",
        "REGION_L1",
        "REGION_L2",
        "PLANTING_SEASON",
    ]

    X = df[numeric_features + categorical_features]

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ])

    return X, preprocessor


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print(f"\n[{name}]")
    print(f"R2  : {r2:.4f}")
    print(f"MAE : {mae:.2f}")

    return {
        "model": name,
        "r2": r2,
        "mae": mae,
        "pred": pred
    }


def main():
    print("🔍 데이터 로딩...")
    df = load_data()

    print(f"학습 데이터 수: {len(df)}")

    X, preprocessor = preprocess(df)

    y = df["TOTAL_QUANTITY"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # RandomForest
    # ---------------------------------------------------
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # ---------------------------------------------------
    # XGBoost
    # ---------------------------------------------------
    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # ---------------------------------------------------
    # Log-Target XGBoost
    # 작은 값 과대예측 완화용
    # ---------------------------------------------------
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    xgb_log_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\n🚀 모델 비교 시작")

    rf_result = evaluate_model(
        "RandomForest",
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    xgb_result = evaluate_model(
        "XGBoost",
        xgb_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Log-target XGBoost
    xgb_log_model.fit(X_train, y_train_log)

    pred_log = xgb_log_model.predict(X_test)
    pred_log_back = np.expm1(pred_log)

    log_r2 = r2_score(y_test, pred_log_back)
    log_mae = mean_absolute_error(y_test, pred_log_back)

    print("\n[XGBoost + log1p(TOTAL_QUANTITY)]")
    print(f"R2  : {log_r2:.4f}")
    print(f"MAE : {log_mae:.2f}")

    # 결과 테이블
    result_df = pd.DataFrame([
        {
            "MODEL": "RandomForest",
            "R2": round(rf_result["r2"], 4),
            "MAE": round(rf_result["mae"], 2)
        },
        {
            "MODEL": "XGBoost",
            "R2": round(xgb_result["r2"], 4),
            "MAE": round(xgb_result["mae"], 2)
        },
        {
            "MODEL": "XGBoost_LogTarget",
            "R2": round(log_r2, 4),
            "MAE": round(log_mae, 2)
        }
    ])

    print("\n" + "=" * 60)
    print(result_df)
    print("=" * 60)

    # 샘플 비교
    sample_compare = pd.DataFrame({
        "actual": y_test.iloc[:10].values,
        "rf_pred": np.round(rf_result["pred"][:10], 1),
        "xgb_pred": np.round(xgb_result["pred"][:10], 1),
        "xgb_log_pred": np.round(pred_log_back[:10], 1)
    })

    print("\n샘플 10건 비교")
    print(sample_compare)

    # 가장 좋은 모델 저장
    scores = [
        ("RandomForest", rf_result["mae"], rf_model),
        ("XGBoost", xgb_result["mae"], xgb_model),
        ("XGBoost_LogTarget", log_mae, xgb_log_model)
    ]

    best_name, best_mae, best_model = sorted(scores, key=lambda x: x[1])[0]

    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, f"yield_best_{best_name}.joblib")
    joblib.dump(best_model, save_path)

    print(f"\n🏆 Best Model: {best_name}")
    print(f"저장 경로: {save_path}")


if __name__ == "__main__":
    main()