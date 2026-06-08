"""Tiện ích làm sạch dữ liệu chiến dịch marketing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


CAC_COT_BAT_BUOC = [
    "Campaign_ID",
    "Campaign_Type",
    "Target_Audience",
    "Duration",
    "Channel_Used",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Language",
    "Engagement_Score",
    "Customer_Segment",
    "Date",
]

CAC_COT_SO = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score",
]

CAC_COT_PHAN_LOAI = [
    "Campaign_ID",
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment",
    "Brand",
]

CAC_COT_KHONG_AM = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "Engagement_Score",
]

KHOA_DINH_DANH_CAP_DONG = [
    "Brand",
    "Campaign_ID",
    "Date",
    "Channel_Used",
    "Customer_Segment",
]


def suy_ra_ten_brand(duong_dan_tep: str | Path) -> str:
    """Suy ra tên brand hiển thị từ tên file CSV thô."""
    ten_tep = Path(duong_dan_tep).stem
    ten_brand = ten_tep.replace("_campaign_data", "").replace("_", " ").strip()
    return ten_brand.title()


def tai_du_lieu_chien_dich_tho(thu_muc_raw: str | Path) -> pd.DataFrame:
    """Tải toàn bộ CSV thô và thêm cột Brand."""
    duong_dan_raw = Path(thu_muc_raw)
    cac_tep_csv = sorted(duong_dan_raw.glob("*.csv"))

    if not cac_tep_csv:
        raise FileNotFoundError(f"Không tìm thấy file CSV trong {duong_dan_raw}")

    cac_bang_du_lieu = []
    for tep_csv in cac_tep_csv:
        bang_du_lieu = pd.read_csv(tep_csv)
        cac_cot_thieu = sorted(set(CAC_COT_BAT_BUOC) - set(bang_du_lieu.columns))
        if cac_cot_thieu:
            raise ValueError(f"{tep_csv} thiếu các cột: {cac_cot_thieu}")

        bang_du_lieu = bang_du_lieu[CAC_COT_BAT_BUOC].copy()
        bang_du_lieu["Brand"] = suy_ra_ten_brand(tep_csv)
        cac_bang_du_lieu.append(bang_du_lieu)

    return pd.concat(cac_bang_du_lieu, ignore_index=True)


def chuan_hoa_kenh_su_dung(gia_tri: Any) -> str:
    """Chuẩn hóa tổ hợp kênh để các biến thể khác thứ tự về cùng một nhóm."""
    if pd.isna(gia_tri):
        return gia_tri

    cac_kenh = [phan.strip() for phan in str(gia_tri).split(",") if phan.strip()]
    return ", ".join(sorted(dict.fromkeys(cac_kenh)))


def lam_sach_du_lieu_chien_dich(
    bang_du_lieu: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Làm sạch dữ liệu chiến dịch và trả về dữ liệu sạch kèm báo cáo chất lượng."""
    bao_cao: dict[str, Any] = {
        "rows_before": int(len(bang_du_lieu)),
        "columns_before": int(len(bang_du_lieu.columns)),
        "missing_before": bang_du_lieu.isna().sum().astype(int).to_dict(),
        "duplicate_rows_before": int(bang_du_lieu.duplicated().sum()),
        "duplicate_campaign_ids_before": int(bang_du_lieu["Campaign_ID"].duplicated().sum()),
    }

    du_lieu_sach = bang_du_lieu.copy()
    du_lieu_sach.columns = [cot.strip() for cot in du_lieu_sach.columns]

    for cot in CAC_COT_PHAN_LOAI:
        if cot in du_lieu_sach.columns:
            du_lieu_sach[cot] = du_lieu_sach[cot].astype("string").str.strip()

    du_lieu_sach["Channel_Used"] = du_lieu_sach["Channel_Used"].map(
        chuan_hoa_kenh_su_dung
    )

    for cot in CAC_COT_SO:
        du_lieu_sach[cot] = pd.to_numeric(du_lieu_sach[cot], errors="coerce")

    du_lieu_sach["Date"] = pd.to_datetime(
        du_lieu_sach["Date"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    du_lieu_sach = du_lieu_sach.drop_duplicates()
    duplicate_row_keys_before = int(
        du_lieu_sach.duplicated(
            subset=KHOA_DINH_DANH_CAP_DONG,
            keep=False,
        ).sum()
    )
    du_lieu_sach = du_lieu_sach.drop_duplicates(
        subset=KHOA_DINH_DANH_CAP_DONG,
        keep="first",
    )

    dong_loi = (
        du_lieu_sach[CAC_COT_BAT_BUOC + ["Brand"]].isna().any(axis=1)
        | (du_lieu_sach[CAC_COT_KHONG_AM] < 0).any(axis=1)
        | (du_lieu_sach["Clicks"] > du_lieu_sach["Impressions"])
        | (du_lieu_sach["Leads"] > du_lieu_sach["Clicks"])
        | (du_lieu_sach["Conversions"] > du_lieu_sach["Leads"])
    )

    cac_dong_loi = du_lieu_sach.loc[dong_loi].copy()
    du_lieu_sach = du_lieu_sach.loc[~dong_loi].copy()

    du_lieu_sach["Date"] = du_lieu_sach["Date"].dt.strftime("%Y-%m-%d")
    du_lieu_sach = du_lieu_sach.sort_values(
        ["Brand", "Date", "Campaign_ID"]
    ).reset_index(drop=True)

    bao_cao.update(
        {
            "rows_after": int(len(du_lieu_sach)),
            "columns_after": int(len(du_lieu_sach.columns)),
            "rows_removed": int(bao_cao["rows_before"] - len(du_lieu_sach)),
            "invalid_rows_removed": int(len(cac_dong_loi)),
            "missing_after": du_lieu_sach.isna().sum().astype(int).to_dict(),
            "row_identity_key": KHOA_DINH_DANH_CAP_DONG,
            "duplicate_row_keys_before": duplicate_row_keys_before,
            "duplicate_row_keys_after": int(
                du_lieu_sach.duplicated(
                    subset=KHOA_DINH_DANH_CAP_DONG,
                    keep=False,
                ).sum()
            ),
            "duplicate_campaign_ids_after": int(
                du_lieu_sach["Campaign_ID"].duplicated().sum()
            ),
            "raw_channel_categories": int(
                bang_du_lieu["Channel_Used"].nunique(dropna=True)
            ),
            "clean_channel_categories": int(
                du_lieu_sach["Channel_Used"].nunique(dropna=True)
            ),
            "brands": sorted(du_lieu_sach["Brand"].dropna().unique().tolist()),
            "date_min": str(du_lieu_sach["Date"].min()),
            "date_max": str(du_lieu_sach["Date"].max()),
            "negative_roi_rows_kept": int((du_lieu_sach["ROI"] < 0).sum()),
            "funnel_violations_after": {
                "clicks_gt_impressions": int(
                    (du_lieu_sach["Clicks"] > du_lieu_sach["Impressions"]).sum()
                ),
                "leads_gt_clicks": int(
                    (du_lieu_sach["Leads"] > du_lieu_sach["Clicks"]).sum()
                ),
                "conversions_gt_leads": int(
                    (du_lieu_sach["Conversions"] > du_lieu_sach["Leads"]).sum()
                ),
            },
        }
    )

    return du_lieu_sach, bao_cao


def luu_ket_qua_lam_sach(
    du_lieu_sach: pd.DataFrame,
    bao_cao: dict[str, Any],
    duong_dan_csv: str | Path,
    duong_dan_bao_cao: str | Path,
) -> None:
    """Lưu dữ liệu sạch và báo cáo làm sạch."""
    duong_dan_csv = Path(duong_dan_csv)
    duong_dan_bao_cao = Path(duong_dan_bao_cao)

    duong_dan_csv.parent.mkdir(parents=True, exist_ok=True)
    duong_dan_bao_cao.parent.mkdir(parents=True, exist_ok=True)

    du_lieu_sach.to_csv(duong_dan_csv, index=False)
    duong_dan_bao_cao.write_text(json.dumps(bao_cao, indent=2), encoding="utf-8")
