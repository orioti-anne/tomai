import os
import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app


def train_yield_model():
    app = create_app()

    with app.app_context():
        print("🔍 생산량 예측용 데이터 조회 중...")

        query = """
        SELECT
            c.CULT_ID,
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
            p.TOTAL_QUANTITY,
            p.TOTAL_SALES

        FROM CULTIVATIONS c
        JOIN FARMS f
            ON c.FARM_ID = f.FARM_ID
        JOIN PRODUCTS p
            ON c.CULT_ID = p.CULT_ID

        WHERE p.TOTAL_QUANTITY IS NOT NULL
        """

        df = pd.read_sql(query, db.engine)
        df.columns = [col.upper() for col in df.columns]

        print(f"조회 건수: {len(df)}")

        # 날짜 변환
        df["PLANTING_DATE"] = pd.to_datetime(df["PLANTING_DATE"])
        df["PRODUCTION_DATE"] = pd.to_datetime(df["PRODUCTION_DATE"])

        # 파생 변수
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

        # 면적당 생산량도 같이 만들어두면 분석에 유용
        df["YIELD_PER_M2"] = (
            df["TOTAL_QUANTITY"] / df["PLANTING_AREA"]
        )

        # 이상치 제거
        df = df[
            (df["PLANTING_AREA"].notnull()) &
            (df["PLANTING_AREA"] > 0) &
            (df["TOTAL_QUANTITY"] > 0)
        ].copy()

        # 너무 비정상적인 값 제거
        q1 = df["YIELD_PER_M2"].quantile(0.01)
        q99 = df["YIELD_PER_M2"].quantile(0.99)

        df = df[
            (df["YIELD_PER_M2"] >= q1) &
            (df["YIELD_PER_M2"] <= q99)
        ]

        print(f"이상치 제거 후 학습 건수: {len(df)}")

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

        # 총 생산량 예측
        y = df["TOTAL_QUANTITY"]

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

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("🚀 생산량 예측 모델 학습 시작...")

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        print("\n" + "=" * 60)
        print("생산량 예측 모델 결과")
        print(f"R2 Score : {r2:.4f}")
        print(f"MAE      : {mae:.2f}")
        print("=" * 60)

        # 모델 저장
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "yield_model.joblib")
        joblib.dump(model, model_path)

        print(f"모델 저장 완료: {model_path}")


if __name__ == "__main__":
    train_yield_model()