import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import text

# [1] 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from smartfarm import db, create_app


def process_growth_final(df, cult_df):
    if df.empty: return df
    df = df.copy()

    # 0. 컬럼명 정규화
    df.columns = [c.upper().strip() for c in df.columns]
    cult_df.columns = [c.upper().strip() for c in cult_df.columns]

    # [수정] 원본 GROWTH_ID 보존
    if 'GROWTH_ID' in df.columns:
        df = df.rename(columns={'GROWTH_ID': 'GROW_ID'})

    df['INSPECT_DATE'] = pd.to_datetime(df['INSPECT_DATE'])
    cult_df['PLANTING_DATE'] = pd.to_datetime(cult_df['PLANTING_DATE'])

    # [1] 생육일수 계산
    df = df.merge(cult_df[['CULT_ID', 'PLANTING_DATE']], on='CULT_ID', how='left')
    df['GROWTH_DAYS'] = (df['INSPECT_DATE'] - df['PLANTING_DATE']).dt.days

    # [2] 품질 마킹 (0:원본, 1:보간, 2:날짜오류)
    df['ORIGIN_TYPE'] = 0
    correction_mask = df['FLOWERS_PER_CLUSTER'].isna() | (df['FLOWERS_PER_CLUSTER'] == 0)
    df.loc[correction_mask, 'ORIGIN_TYPE'] = 1

    date_error_mask = df['GROWTH_DAYS'] < 0
    if date_error_mask.any():
        print(f"🚩 [Quality Check] 날짜 오류 {date_error_mask.sum()}건 발견 -> TYPE 2 지정")
        df.loc[date_error_mask, 'ORIGIN_TYPE'] = 2

    # [3] 시계열 보간
    df = df.sort_values(['CULT_ID', 'PLANT_NUM', 'CLUSTER_NUM', 'GROWTH_DAYS'])
    group_keys = ['CULT_ID', 'PLANT_NUM', 'CLUSTER_NUM']
    target_cols = ['FLOWERS_PER_CLUSTER', 'BLOOMING_PER_CLUSTER', 'FRUITS_PER_CLUSTER', 'GROWTH_LENGTH', 'PLANT_HEIGHT']

    for col in target_cols:
        if col in df.columns:
            df[col] = df.groupby(group_keys)[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )

    # [4] 결측치 보충
    valid_data = df[df['FLOWERS_PER_CLUSTER'] > 0]
    farm_reference = valid_data.groupby('CULT_ID')['FLOWERS_PER_CLUSTER'].median().to_dict()
    unfilled_mask = df['FLOWERS_PER_CLUSTER'].isna() | (df['FLOWERS_PER_CLUSTER'] == 0)
    df.loc[unfilled_mask, 'FLOWERS_PER_CLUSTER'] = df.loc[unfilled_mask, 'CULT_ID'].map(farm_reference).fillna(6.0)

    # [5] 역산 로직
    if 'BLOOMING_GROUP' in df.columns:
        calc_mask = df['BLOOMING_PER_CLUSTER'].isna() & df['BLOOMING_GROUP'].notna()
        if calc_mask.any():
            decimal = df['BLOOMING_GROUP'] - np.floor(df['BLOOMING_GROUP'])
            df.loc[calc_mask, 'BLOOMING_PER_CLUSTER'] = (decimal * df['FLOWERS_PER_CLUSTER']).round(1)
            f_mask = df['FRUITS_PER_CLUSTER'].isna()
            df.loc[f_mask, 'FRUITS_PER_CLUSTER'] = (df.loc[f_mask, 'BLOOMING_PER_CLUSTER'] * 0.85).round(1)

    # [6] GROWSU_ID 시퀀스 넘버 생성
    df = df.reset_index(drop=True)
    df['GROWSU_ID'] = df.index + 1

    # [7] 최종 컬럼 정리
    final_cols = [
        'GROWSU_ID', 'GROW_ID', 'CULT_ID', 'INSPECT_DATE', 'GROWTH_DAYS', 'PLANT_NUM', 'BRANCH_NUM',
        'PLANT_HEIGHT', 'GROWTH_LENGTH', 'LEAF_COUNT', 'LEAF_LENGTH', 'LEAF_WIDTH',
        'BRANCH_WIDTH', 'CLUSTER_HEIGHT', 'CLUSTER_NUM', 'FLOWERS_PER_CLUSTER',
        'BLOOMING_PER_CLUSTER', 'FRUITS_PER_CLUSTER', 'ORIGIN_TYPE'
    ]

    return df[[c for c in final_cols if c in df.columns]]


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        print("📂 오라클 DB 로드 시작...")
        raw_growth = pd.read_sql(text("SELECT * FROM GROWTH"), db.engine)
        raw_cult = pd.read_sql(text("SELECT CULT_ID, PLANTING_DATE FROM CULTIVATIONS"), db.engine)

        if not raw_growth.empty:
            # 1. 전처리 실행
            final_df = process_growth_final(raw_growth, raw_cult)

            # 2. 백업용 CSV 저장
            data_dir = os.path.join(project_root, 'data')
            if not os.path.exists(data_dir): os.makedirs(data_dir)
            final_df.to_csv(os.path.join(data_dir, 'grow_summary_backup.csv'), index=False, encoding='utf-8-sig')

            # 3. 오라클 DB 적재 실행
            print(f"🚀 GROW_SUMMARY 테이블에 {len(final_df):,}행 적재를 시작합니다...")
            try:
                # 기존 데이터 삭제 (테이블 구조는 유지)
                with db.engine.begin() as conn:
                    conn.execute(text("TRUNCATE TABLE GROW_SUMMARY"))

                # 데이터 삽입 (컬럼 순서가 맞아야 함)
                final_df.to_sql(
                    name='grow_summary',
                    con=db.engine,
                    if_exists='append',
                    index=False,
                    chunksize=1000  # 대량 데이터 처리를 위해 분할 삽입
                )
                print("✨ DB 적재가 성공적으로 완료되었습니다!")

            except Exception as e:
                print(f"❌ DB 적재 중 오류 발생: {e}")

        else:
            print("❌ 원본 데이터를 가져오지 못했습니다.")