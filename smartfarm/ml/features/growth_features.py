import pandas as pd


DEFAULT_PRODUCTION_FEATURES = [
    "DAP",
    "PLANT_HEIGHT",
    "LEAF_COUNT",
    "GROWTH_LENGTH",
    "LEAF_LENGTH",
    "LEAF_WIDTH",
    "BRANCH_WIDTH",
    "CLUSTER_HEIGHT",
    "CLUSTER_NUM",
    "FLOWERS_PER_CLUSTER",
    "BLOOMING_PER_CLUSTER",
    "FRUITS_PER_CLUSTER",
]


def build_production_features(growth_data: dict, dap: int) -> dict:
    return {
        "DAP": dap,
        "PLANT_HEIGHT": growth_data.get("plant_height"),
        "LEAF_COUNT": growth_data.get("leaf_count"),
        "GROWTH_LENGTH": growth_data.get("growth_length"),
        "LEAF_LENGTH": growth_data.get("leaf_length"),
        "LEAF_WIDTH": growth_data.get("leaf_width"),
        "BRANCH_WIDTH": growth_data.get("branch_width"),
        "CLUSTER_HEIGHT": growth_data.get("cluster_height"),
        "CLUSTER_NUM": growth_data.get("cluster_num"),
        "FLOWERS_PER_CLUSTER": growth_data.get("flowers_per_cluster"),
        "BLOOMING_PER_CLUSTER": growth_data.get("blooming_per_cluster"),
        "FRUITS_PER_CLUSTER": growth_data.get("fruits_per_cluster"),
    }


def fill_missing_production_features(feature_dict: dict) -> dict:
    defaults = {
        "PLANT_HEIGHT": 100.0,
        "LEAF_COUNT": 10.0,
        "GROWTH_LENGTH": 15.0,
        "LEAF_LENGTH": 12.0,
        "LEAF_WIDTH": 8.0,
        "BRANCH_WIDTH": 8.0,
        "CLUSTER_HEIGHT": 20.0,
        "CLUSTER_NUM": 3.0,
        "FLOWERS_PER_CLUSTER": 5.0,
        "BLOOMING_PER_CLUSTER": 3.0,
        "FRUITS_PER_CLUSTER": 2.0,
    }

    result = feature_dict.copy()
    for key, default_value in defaults.items():
        if result.get(key) is None:
            result[key] = default_value

    return result


def to_dataframe(feature_dict: dict, feature_order: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame([feature_dict])

    if feature_order:
        for col in feature_order:
            if col not in df.columns:
                df[col] = None
        df = df[feature_order]

    return df