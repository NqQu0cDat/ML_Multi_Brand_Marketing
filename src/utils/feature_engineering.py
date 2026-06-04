"""Tiện ích tạo feature marketing từ dữ liệu chiến dịch đã làm sạch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


CAC_COT_CAN_CO = [
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "Date",
]

CAC_FEATURE_TY_LE = [
    "CTR",
    "Lead_Rate",
    "Conversion_Rate",
    "Revenue_per_Conversion",
    "Cost_per_Conversion",
    "Revenue_per_Click",
]

CAC_FEATURE_NGAY = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Quarter",
]


def chia_an_toan(tu_so: pd.Series, mau_so: pd.Series) -> pd.Series:
    """Chia hai cột số và trả về 0 khi mẫu số bằng 0."""
    ket_qua = tu_so.div(mau_so.where(mau_so != 0))
    return ket_qua.fillna(0)


def tai_du_lieu_sach(duong_dan_csv: str | Path) -> pd.DataFrame:
    """Tải dữ liệu đã làm sạch từ file CSV."""
    duong_dan_csv = Path(duong_dan_csv)
    if not duong_dan_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu sạch: {duong_dan_csv}")

    du_lieu = pd.read_csv(duong_dan_csv)
    cac_cot_thieu = sorted(set(CAC_COT_CAN_CO) - set(du_lieu.columns))
    if cac_cot_thieu:
        raise ValueError(f"Dữ liệu thiếu các cột cần thiết: {cac_cot_thieu}")

    return du_lieu


def tao_feature_marketing(
    du_lieu_sach: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tạo feature marketing và trả về dữ liệu feature kèm báo cáo."""
    du_lieu_feature = du_lieu_sach.copy()
    so_cot_truoc = len(du_lieu_feature.columns)

    ngay_chien_dich = pd.to_datetime(
        du_lieu_feature["Date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    du_lieu_feature["CTR"] = chia_an_toan(
        du_lieu_feature["Clicks"],
        du_lieu_feature["Impressions"],
    )
    du_lieu_feature["Lead_Rate"] = chia_an_toan(
        du_lieu_feature["Leads"],
        du_lieu_feature["Clicks"],
    )
    du_lieu_feature["Conversion_Rate"] = chia_an_toan(
        du_lieu_feature["Conversions"],
        du_lieu_feature["Leads"],
    )
    du_lieu_feature["Revenue_per_Conversion"] = chia_an_toan(
        du_lieu_feature["Revenue"],
        du_lieu_feature["Conversions"],
    )
    du_lieu_feature["Cost_per_Conversion"] = chia_an_toan(
        du_lieu_feature["Acquisition_Cost"],
        du_lieu_feature["Conversions"],
    )
    du_lieu_feature["Revenue_per_Click"] = chia_an_toan(
        du_lieu_feature["Revenue"],
        du_lieu_feature["Clicks"],
    )

    du_lieu_feature["Year"] = ngay_chien_dich.dt.year
    du_lieu_feature["Month"] = ngay_chien_dich.dt.month
    du_lieu_feature["Day"] = ngay_chien_dich.dt.day
    du_lieu_feature["DayOfWeek"] = ngay_chien_dich.dt.dayofweek
    du_lieu_feature["Quarter"] = ngay_chien_dich.dt.quarter

    bao_cao = {
        "rows": int(len(du_lieu_feature)),
        "columns_before": int(so_cot_truoc),
        "columns_after": int(len(du_lieu_feature.columns)),
        "features_added": CAC_FEATURE_TY_LE + CAC_FEATURE_NGAY,
        "missing_after": du_lieu_feature.isna().sum().astype(int).to_dict(),
        "date_parse_failures": int(ngay_chien_dich.isna().sum()),
        "zero_denominator_counts": {
            "impressions_zero": int((du_lieu_feature["Impressions"] == 0).sum()),
            "clicks_zero": int((du_lieu_feature["Clicks"] == 0).sum()),
            "leads_zero": int((du_lieu_feature["Leads"] == 0).sum()),
            "conversions_zero": int((du_lieu_feature["Conversions"] == 0).sum()),
        },
        "feature_summary": {
            cot: {
                "min": float(du_lieu_feature[cot].min()),
                "mean": float(du_lieu_feature[cot].mean()),
                "max": float(du_lieu_feature[cot].max()),
            }
            for cot in CAC_FEATURE_TY_LE
        },
    }

    return du_lieu_feature, bao_cao


def luu_ket_qua_feature(
    du_lieu_feature: pd.DataFrame,
    bao_cao: dict[str, Any],
    duong_dan_csv: str | Path,
    duong_dan_bao_cao: str | Path,
) -> None:
    """Lưu dữ liệu feature và báo cáo feature engineering."""
    duong_dan_csv = Path(duong_dan_csv)
    duong_dan_bao_cao = Path(duong_dan_bao_cao)

    duong_dan_csv.parent.mkdir(parents=True, exist_ok=True)
    duong_dan_bao_cao.parent.mkdir(parents=True, exist_ok=True)

    du_lieu_feature.to_csv(duong_dan_csv, index=False)
    duong_dan_bao_cao.write_text(json.dumps(bao_cao, indent=2), encoding="utf-8")
