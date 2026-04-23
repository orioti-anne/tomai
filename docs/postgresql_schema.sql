-- TomAI PostgreSQL Schema
-- Oracle에서 변환됨

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- USERS
CREATE TABLE users (
    user_id     SERIAL PRIMARY KEY,
    username    VARCHAR(50)  NOT NULL UNIQUE,
    password    VARCHAR(255) NOT NULL,
    name        VARCHAR(100) NOT NULL,
    phone       VARCHAR(20)  NOT NULL UNIQUE,
    email       VARCHAR(150),
    address     VARCHAR(500),
    created_at  TIMESTAMP DEFAULT NOW()
);

INSERT INTO users (user_id, username, password, name, phone)
VALUES (0, 'ghost_user', 'system_protected_no_login', '탈퇴회원', '000-0000-0000');

-- FARMS
CREATE TABLE farms (
    farm_id           SERIAL PRIMARY KEY,
    user_id           INTEGER NOT NULL REFERENCES users(user_id),
    farm_name         VARCHAR(100) NOT NULL,
    farm_num          INTEGER,
    region_l1         VARCHAR(50),
    region_l2         VARCHAR(50),
    total_area        NUMERIC,
    first_survey_year INTEGER,
    created_at        TIMESTAMP DEFAULT NOW(),
    is_active         CHAR(1) DEFAULT 'Y'
);

-- CULTIVATIONS
CREATE TABLE cultivations (
    cult_id                SERIAL PRIMARY KEY,
    farm_id                INTEGER NOT NULL REFERENCES farms(farm_id),
    item                   VARCHAR(100) NOT NULL,
    item_variety           VARCHAR(100),
    crop_cycle             INTEGER NOT NULL,
    planting_date          DATE NOT NULL,
    planting_area          NUMERIC NOT NULL,
    planting_density       NUMERIC,
    house_type             VARCHAR(50) NOT NULL CHECK (house_type IN ('비닐', '유리', '기타')),
    house_form             VARCHAR(50) CHECK (house_form IN ('연동', '단동', '광폭', '연동양액', '기타')),
    status                 VARCHAR(20) NOT NULL DEFAULT 'active',
    survey_year            INTEGER,
    created_at             TIMESTAMP DEFAULT NOW(),
    cult_name              VARCHAR(200),
    virtual_sensor_enabled CHAR(1) DEFAULT 'N'
);

-- QUESTION
CREATE TABLE question (
    id           SERIAL PRIMARY KEY,
    subject      VARCHAR(200) NOT NULL,
    content      TEXT NOT NULL,
    created_date TIMESTAMP NOT NULL,
    user_id      INTEGER NOT NULL REFERENCES users(user_id)
);

-- ANSWER
CREATE TABLE answer (
    id           SERIAL PRIMARY KEY,
    question_id  INTEGER REFERENCES question(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,
    created_date TIMESTAMP NOT NULL,
    user_id      INTEGER NOT NULL REFERENCES users(user_id)
);

-- ENVIRONMENT
CREATE TABLE environment (
    env_id              SERIAL PRIMARY KEY,
    cult_id             INTEGER NOT NULL REFERENCES cultivations(cult_id),
    measure_time        TIMESTAMP NOT NULL,
    out_temp            NUMERIC(5,2),
    out_wind_direction  NUMERIC(5,2),
    out_wind_speed      NUMERIC(5,2),
    out_solar_rad       NUMERIC(10,2),
    out_acc_solar_rad   NUMERIC(15,2),
    rain_detection      SMALLINT,
    in_temp             NUMERIC(5,2),
    in_humidity         NUMERIC(5,2),
    in_co2              NUMERIC(10,2),
    soil_temp           NUMERIC(5,2),
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ENV_CLEANED
CREATE TABLE env_cleaned (
    envcl_id                SERIAL PRIMARY KEY,
    env_id                  INTEGER NOT NULL REFERENCES environment(env_id),
    cult_id                 INTEGER NOT NULL REFERENCES cultivations(cult_id),
    measure_time            TIMESTAMP NOT NULL,
    out_temp                NUMERIC(5,2),
    out_wind_direction      NUMERIC(5,2),
    out_wind_speed          NUMERIC(5,2),
    out_solar_rad           NUMERIC(10,2),
    out_acc_solar_rad       NUMERIC(10,2),
    rain_detection          SMALLINT,
    in_temp                 NUMERIC(5,2),
    in_humidity             NUMERIC(5,2),
    in_co2                  NUMERIC(10,2),
    soil_temp               NUMERIC(5,2),
    out_acc_solar_rad_status SMALLINT DEFAULT 0,
    in_temp_status          SMALLINT DEFAULT 0,
    in_humidity_status      SMALLINT DEFAULT 0,
    in_co2_status           SMALLINT DEFAULT 0,
    measure_date            DATE,
    measure_hour            SMALLINT,
    created_at              TIMESTAMP DEFAULT NOW()
);

-- ENV_SUMMARY
CREATE TABLE env_summary (
    envsu_id            SERIAL PRIMARY KEY,
    cult_id             INTEGER NOT NULL REFERENCES cultivations(cult_id),
    measure_date        DATE NOT NULL,
    daily_out_temp      NUMERIC(10,2),
    daily_acc_solar     NUMERIC(10,2),
    daily_rain_detection SMALLINT,
    daily_in_temp       NUMERIC(10,2),
    daily_in_humidity   NUMERIC(10,2),
    daily_in_co2        NUMERIC(10,2),
    daily_soil_temp     NUMERIC(10,2),
    acc_temp            NUMERIC(10,2),
    acc_solar           NUMERIC(10,2),
    created_at          TIMESTAMP DEFAULT NOW()
);

-- GROWTH
CREATE TABLE growth (
    growth_id            SERIAL PRIMARY KEY,
    cult_id              INTEGER NOT NULL REFERENCES cultivations(cult_id),
    inspect_date         DATE NOT NULL,
    plant_num            INTEGER,
    branch_num           INTEGER,
    plant_height         NUMERIC,
    growth_length        NUMERIC,
    leaf_count           INTEGER,
    leaf_length          NUMERIC,
    leaf_width           NUMERIC,
    branch_width         NUMERIC,
    cluster_height       NUMERIC,
    cluster_num          INTEGER,
    flowers_per_cluster  INTEGER,
    blooming_per_cluster INTEGER,
    fruits_per_cluster   INTEGER,
    blooming_group       INTEGER,
    fruiting_group       INTEGER,
    remarks              VARCHAR(1000),
    created_at           TIMESTAMP DEFAULT NOW()
);

-- GROW_SUMMARY
CREATE TABLE grow_summary (
    growsu_id            SERIAL PRIMARY KEY,
    growth_id            INTEGER REFERENCES growth(growth_id),
    cult_id              INTEGER NOT NULL REFERENCES cultivations(cult_id),
    inspect_date         DATE NOT NULL,
    growth_days          INTEGER,
    plant_num            INTEGER,
    branch_num           INTEGER,
    plant_height         NUMERIC(10,2),
    growth_length        NUMERIC(10,2),
    leaf_count           INTEGER,
    leaf_length          NUMERIC(10,2),
    leaf_width           NUMERIC(10,2),
    branch_width         NUMERIC(10,2),
    cluster_height       NUMERIC(10,2),
    cluster_num          INTEGER,
    flowers_per_cluster  NUMERIC(10,2),
    blooming_per_cluster NUMERIC(10,2),
    fruits_per_cluster   NUMERIC(10,2),
    created_at           TIMESTAMP DEFAULT NOW(),
    origin_type          INTEGER DEFAULT 0
);

-- KAMIS_TOMATO_PRICE
CREATE TABLE kamis_tomato_price (
    price_id    SERIAL PRIMARY KEY,
    price_date  DATE NOT NULL,
    market_name VARCHAR(50),
    item_name   VARCHAR(50),
    trade_unit  VARCHAR(50),
    grade       VARCHAR(20),
    avg_price   NUMERIC(15,2),
    unit_kg     NUMERIC(10,2),
    price_per_kg NUMERIC(15,2),
    grade_score SMALLINT,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- PREDICTION_RESULTS
CREATE TABLE prediction_results (
    prediction_id          SERIAL PRIMARY KEY,
    user_id                INTEGER NOT NULL REFERENCES users(user_id),
    farm_id                INTEGER NOT NULL REFERENCES farms(farm_id),
    cult_id                INTEGER NOT NULL REFERENCES cultivations(cult_id),
    prediction_date        TIMESTAMP DEFAULT NOW() NOT NULL,
    item                   VARCHAR(100),
    item_variety           VARCHAR(100),
    crop_cycle             INTEGER,
    planting_date          DATE,
    expected_harvest_date  DATE,
    planting_area          NUMERIC,
    planting_density       NUMERIC,
    house_type             VARCHAR(50),
    house_form             VARCHAR(50),
    avg_days_to_peak_harvest NUMERIC,
    avg_yield_per_m2       NUMERIC,
    expected_quantity      NUMERIC,
    expected_price_per_kg  NUMERIC,
    expected_sales         NUMERIC,
    sample_count           INTEGER,
    latest_market_price    NUMERIC,
    market_name            VARCHAR(100),
    created_at             TIMESTAMP DEFAULT NOW(),
    updated_at             TIMESTAMP DEFAULT NOW(),
    prediction_source      VARCHAR(20),
    prediction_confidence  VARCHAR(20),
    prediction_message     TEXT,
    price_day_95           NUMERIC,
    price_day_105          NUMERIC,
    price_day_115          NUMERIC
);

-- PRODUCTS
CREATE TABLE products (
    product_id      SERIAL PRIMARY KEY,
    cult_id         INTEGER NOT NULL REFERENCES cultivations(cult_id),
    production_date DATE,
    total_quantity  NUMERIC,
    total_sales     NUMERIC,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- PROD_SUMMARY
CREATE TABLE prod_summary (
    prodsu_id          SERIAL PRIMARY KEY,
    cult_id            INTEGER NOT NULL REFERENCES cultivations(cult_id),
    cult_end_date      DATE,
    cult_total_quantity NUMERIC,
    cult_total_sales   NUMERIC,
    yield_per_area     NUMERIC(10,2),
    unit_price         NUMERIC(10,2),
    origin_type        INTEGER DEFAULT 0,
    created_at         TIMESTAMP DEFAULT NOW()
);

-- WEATHER_INDEX
CREATE TABLE weather_index (
    w_date     DATE PRIMARY KEY,
    avg_temp   NUMERIC(5,2),
    sunshine   NUMERIC(5,2),
    rain       NUMERIC(7,2),
    humid      NUMERIC(5,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ALEMBIC_VERSION
CREATE TABLE alembic_version (
    version_num VARCHAR(32) PRIMARY KEY
);

-- 시퀀스 초기화 (데이터 마이그레이션 후 실행)
-- SELECT setval('users_user_id_seq', (SELECT MAX(user_id) FROM users));
-- SELECT setval('farms_farm_id_seq', (SELECT MAX(farm_id) FROM farms));
-- SELECT setval('cultivations_cult_id_seq', (SELECT MAX(cult_id) FROM cultivations));
-- SELECT setval('environment_env_id_seq', (SELECT MAX(env_id) FROM environment));
-- SELECT setval('env_cleaned_envcl_id_seq', (SELECT MAX(envcl_id) FROM env_cleaned));
-- SELECT setval('env_summary_envsu_id_seq', (SELECT MAX(envsu_id) FROM env_summary));
-- SELECT setval('growth_growth_id_seq', (SELECT MAX(growth_id) FROM growth));
-- SELECT setval('prediction_results_prediction_id_seq', (SELECT MAX(prediction_id) FROM prediction_results));
