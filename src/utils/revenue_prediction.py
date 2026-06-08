"""Tien ich train va du doan Revenue cho campaign marketing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - phụ thuộc môi trường cài đặt
    XGBRegressor = None

RANDOM_STATE = 42
TARGET = "Revenue"

FEATURE_PRE_CAMPAIGN = [
    "Brand",
    "Campaign_Type",
    "Target_Audience",
    "Duration",
    "Channel_Used",
    "Acquisition_Cost",
    "Language",
    "Customer_Segment",
    "Month",
    "Quarter",
]

FEATURE_AFTER_FUNNEL = FEATURE_PRE_CAMPAIGN + [
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "CTR",
    "Lead_Rate",
    "Conversion_Rate",
    "Cost_per_Conversion",
]

CAC_COT_CAM = [
    "Revenue",
    "ROI",
    "Revenue_per_Conversion",
    "Revenue_per_Click",
]

CAC_FEATURE_FUNNEL = [
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "CTR",
    "Lead_Rate",
    "Conversion_Rate",
    "Cost_per_Conversion",
]

SCENARIO_CONFIG = {
    "pre_campaign": {
        "ten_hien_thi": "Pre-campaign",
        "mo_ta": "Dự đoán Revenue trước khi campaign chạy.",
        "features": FEATURE_PRE_CAMPAIGN,
        "late_stage": False,
    },
    "after_funnel": {
        "ten_hien_thi": "After-funnel",
        "mo_ta": "Late-stage prediction khi đã có funnel metrics.",
        "features": FEATURE_AFTER_FUNNEL,
        "late_stage": True,
    },
}

ALIAS_COT = {
    "Conversion_Rate": ["CVR"],
    "Cost_per_Conversion": ["CPA"],
}


@dataclass
class KetQuaScenario:
    """Kết quả train/evaluate cho một scenario."""

    scenario: str
    best_model_name: str
    best_pipeline: Pipeline
    metrics: dict[str, dict[str, float]]
    y_test: np.ndarray
    y_pred_best: np.ndarray
    x_test: pd.DataFrame
    split_info: dict[str, Any]
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    feature_importance: list[dict[str, float | str]]


def tai_du_lieu_revenue(duong_dan_csv: str | Path) -> pd.DataFrame:
    """Tải dữ liệu feature và bổ sung alias nội bộ nếu dataset dùng tên KPI khác."""
    duong_dan_csv = Path(duong_dan_csv)
    if not duong_dan_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu ML: {duong_dan_csv}")

    du_lieu = pd.read_csv(duong_dan_csv)
    if "Date" in du_lieu.columns:
        du_lieu["Date"] = pd.to_datetime(du_lieu["Date"], errors="coerce")

    for cot_chuan, cac_alias in ALIAS_COT.items():
        if cot_chuan in du_lieu.columns:
            continue
        for cot_alias in cac_alias:
            if cot_alias in du_lieu.columns:
                du_lieu[cot_chuan] = du_lieu[cot_alias]
                break

    return du_lieu


def kiem_tra_cot_bat_buoc(du_lieu: pd.DataFrame, feature_columns: list[str]) -> None:
    """Báo lỗi rõ ràng nếu thiếu target hoặc feature cần dùng."""
    cac_cot_thieu = sorted(set([TARGET] + feature_columns) - set(du_lieu.columns))
    if cac_cot_thieu:
        raise ValueError(
            "Dữ liệu thiếu cột cho Revenue Prediction: " + ", ".join(cac_cot_thieu)
        )


def kiem_tra_data_leakage(scenario: str, feature_columns: list[str]) -> None:
    """Chặn feature leakage trước khi train model."""
    feature_set = set(feature_columns)
    cac_cot_cam_trong_feature = sorted(feature_set.intersection(CAC_COT_CAM))
    cac_cot_revenue_phat_sinh = sorted(
        cot for cot in feature_columns if "revenue" in cot.lower()
    )
    if cac_cot_cam_trong_feature or cac_cot_revenue_phat_sinh:
        cac_cot_loi = sorted(set(cac_cot_cam_trong_feature + cac_cot_revenue_phat_sinh))
        raise ValueError(
            "Phát hiện data leakage từ Revenue: " + ", ".join(cac_cot_loi)
        )

    if scenario == "pre_campaign":
        cac_feature_funnel = sorted(feature_set.intersection(CAC_FEATURE_FUNNEL))
        if cac_feature_funnel:
            raise ValueError(
                "Pre-campaign không được dùng funnel metrics: "
                + ", ".join(cac_feature_funnel)
            )


def tach_feature_theo_kieu(
    du_lieu: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Tách feature phân loại và feature số."""
    numeric_features = [
        cot for cot in feature_columns if is_numeric_dtype(du_lieu[cot])
    ]
    categorical_features = [cot for cot in feature_columns if cot not in numeric_features]
    return categorical_features, numeric_features


def tao_one_hot_encoder() -> OneHotEncoder:
    """Tạo OneHotEncoder tương thích nhiều phiên bản sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - chỉ dùng cho sklearn cũ
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def tao_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Tạo ColumnTransformer cho categorical và numeric features."""
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", tao_one_hot_encoder()),
        ]
    )
    numeric_pipeline = Pipeline(steps=numeric_steps)

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )


def tao_model_candidates(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Tạo danh sách model cần so sánh; XGBoost là optional."""
    models: dict[str, Any] = {
        "dummy": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=180,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def tach_train_test(
    du_lieu: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Tách train/test theo thời gian nếu có Date, nếu không fallback random split."""
    du_lieu_hop_le = du_lieu.dropna(subset=[TARGET]).copy()
    du_lieu_hop_le = du_lieu_hop_le[du_lieu_hop_le[TARGET] >= 0].copy()

    if "Date" in du_lieu_hop_le.columns and not du_lieu_hop_le["Date"].isna().all():
        du_lieu_hop_le = du_lieu_hop_le.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        vi_tri_cat = max(1, min(len(du_lieu_hop_le) - 1, int(len(du_lieu_hop_le) * (1 - test_size))))
        ngay_cat = du_lieu_hop_le.iloc[vi_tri_cat]["Date"]
        train = du_lieu_hop_le[du_lieu_hop_le["Date"] < ngay_cat].copy()
        test = du_lieu_hop_le[du_lieu_hop_le["Date"] >= ngay_cat].copy()
        if train.empty or test.empty:
            raise ValueError("Không thể tạo temporal split có train/test khác ngày.")
        train_date_max = train["Date"].max()
        test_date_min = test["Date"].min()
        split_info = {
            "strategy": "temporal_strict_date_boundary",
            "requested_test_size": test_size,
            "actual_test_size": float(len(test) / len(du_lieu_hop_le)),
            "cutoff_date": str(ngay_cat.date()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_min": str(train["Date"].min().date()),
            "train_date_max": str(train_date_max.date()),
            "test_date_min": str(test_date_min.date()),
            "test_date_max": str(test["Date"].max().date()),
            "temporal_boundary_valid": bool(train_date_max < test_date_min),
        }
    else:
        train, test = train_test_split(
            du_lieu_hop_le,
            test_size=test_size,
            random_state=random_state,
        )
        split_info = {
            "strategy": "random_fallback",
            "fallback_reason": "Dataset không có cột Date hợp lệ để temporal split.",
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }

    return (
        train[feature_columns].copy(),
        test[feature_columns].copy(),
        train[TARGET].copy(),
        test[TARGET].copy(),
        split_info,
    )


def tinh_mape_an_toan(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính MAPE, bỏ qua dòng target bằng 0 để tránh chia cho 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def tinh_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Tính metrics hồi quy trên thang Revenue gốc."""
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    y_true = np.asarray(y_true, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": tinh_mape_an_toan(y_true, y_pred),
    }


def du_doan_revenue(model: Pipeline, du_lieu_feature: pd.DataFrame) -> np.ndarray:
    """Dự đoán Revenue và đổi ngược từ log1p về thang gốc."""
    y_log = model.predict(du_lieu_feature)
    return np.maximum(np.expm1(y_log), 0)


def lay_ten_feature_sau_encode(model: Pipeline) -> list[str]:
    """Lấy tên feature sau ColumnTransformer nếu sklearn hỗ trợ."""
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        return []
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def lay_feature_importance(model: Pipeline, top_n: int = 25) -> list[dict[str, float | str]]:
    """Lấy feature importance hoặc coefficient nếu model hỗ trợ."""
    regressor = model.named_steps.get("model")
    feature_names = lay_ten_feature_sau_encode(model)
    if not feature_names or regressor is None:
        return []

    if hasattr(regressor, "feature_importances_"):
        values = np.asarray(regressor.feature_importances_, dtype=float)
    elif hasattr(regressor, "coef_"):
        values = np.abs(np.asarray(regressor.coef_, dtype=float)).ravel()
    else:
        return []

    if len(values) != len(feature_names):
        return []
    thu_tu = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": str(feature_names[i]), "importance": float(values[i])}
        for i in thu_tu
    ]


def train_scenario(
    du_lieu: pd.DataFrame,
    scenario: str,
    random_state: int = RANDOM_STATE,
) -> KetQuaScenario:
    """Train toàn bộ model cho một scenario và chọn model tốt nhất theo RMSE."""
    cau_hinh = SCENARIO_CONFIG[scenario]
    feature_columns = list(cau_hinh["features"])
    kiem_tra_data_leakage(scenario, feature_columns)
    kiem_tra_cot_bat_buoc(du_lieu, feature_columns)
    categorical_features, numeric_features = tach_feature_theo_kieu(du_lieu, feature_columns)
    x_train, x_test, y_train, y_test, split_info = tach_train_test(
        du_lieu,
        feature_columns,
        random_state=random_state,
    )
    y_train_log = np.log1p(y_train.to_numpy(dtype=float))
    y_test_goc = y_test.to_numpy(dtype=float)

    ket_qua_metrics: dict[str, dict[str, float]] = {}
    du_doan_theo_model: dict[str, np.ndarray] = {}
    pipeline_theo_model: dict[str, Pipeline] = {}

    for ten_model, model in tao_model_candidates(random_state).items():
        scale_numeric = ten_model == "ridge"
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    tao_preprocessor(
                        categorical_features,
                        numeric_features,
                        scale_numeric=scale_numeric,
                    ),
                ),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train_log)
        y_pred = du_doan_revenue(pipeline, x_test)
        ket_qua_metrics[ten_model] = tinh_metrics(y_test_goc, y_pred)
        du_doan_theo_model[ten_model] = y_pred
        pipeline_theo_model[ten_model] = pipeline

    best_model_name = min(
        ket_qua_metrics,
        key=lambda ten: ket_qua_metrics[ten]["rmse"],
    )
    best_pipeline = pipeline_theo_model[best_model_name]
    return KetQuaScenario(
        scenario=scenario,
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        metrics=ket_qua_metrics,
        y_test=y_test_goc,
        y_pred_best=du_doan_theo_model[best_model_name],
        x_test=x_test,
        split_info=split_info,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        feature_importance=lay_feature_importance(best_pipeline),
    )


def luu_model(model: Pipeline, duong_dan: str | Path) -> None:
    """Lưu model bằng joblib."""
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, duong_dan)


def tai_model(duong_dan: str | Path) -> Pipeline:
    """Tải model revenue prediction."""
    return joblib.load(duong_dan)


def ghi_json(du_lieu: dict[str, Any], duong_dan: str | Path) -> None:
    """Ghi JSON có định dạng."""
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text(json.dumps(du_lieu, indent=2, ensure_ascii=False), encoding="utf-8")
