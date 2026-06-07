"""Chạy EDA cho dữ liệu chiến dịch marketing đã feature engineering."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CAC_COT_TUONG_QUAN = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score",
    "CTR",
    "Lead_Rate",
    "Conversion_Rate",
    "Revenue_per_Conversion",
    "Cost_per_Conversion",
    "Revenue_per_Click",
]


def phan_tich_tham_so() -> argparse.Namespace:
    bo_phan_tich = argparse.ArgumentParser(description="Chạy EDA cho campaign data.")
    bo_phan_tich.add_argument(
        "--duong-dan-input",
        "--input-csv",
        dest="duong_dan_input",
        default="data/processed/marketing_campaigns_features.csv",
        help="Đường dẫn file CSV đã tạo feature.",
    )
    bo_phan_tich.add_argument(
        "--thu-muc-bieu-do",
        "--plots-dir",
        dest="thu_muc_bieu_do",
        default="results/plots/eda",
        help="Thư mục lưu biểu đồ EDA.",
    )
    bo_phan_tich.add_argument(
        "--duong-dan-bao-cao",
        "--summary-json",
        dest="duong_dan_bao_cao",
        default="results/metrics/eda_summary.json",
        help="Đường dẫn lưu summary EDA dạng JSON.",
    )
    bo_phan_tich.add_argument(
        "--duong-dan-notebook",
        "--notebook",
        dest="duong_dan_notebook",
        default="notebooks/01_eda.ipynb",
        help="Đường dẫn lưu notebook EDA.",
    )
    return bo_phan_tich.parse_args()


def tai_du_lieu_feature(duong_dan_input: str | Path) -> pd.DataFrame:
    """Tải dữ liệu feature và parse ngày để phục vụ EDA."""
    duong_dan_input = Path(duong_dan_input)
    if not duong_dan_input.exists():
        raise FileNotFoundError(f"Không tìm thấy file feature: {duong_dan_input}")

    du_lieu = pd.read_csv(duong_dan_input)
    du_lieu["Date"] = pd.to_datetime(du_lieu["Date"], errors="coerce")
    return du_lieu


def luu_bieu_do(duong_dan_bieu_do: Path) -> None:
    """Lưu biểu đồ hiện tại với layout gọn."""
    plt.tight_layout()
    plt.savefig(duong_dan_bieu_do, dpi=160, bbox_inches="tight")
    plt.close()


def ve_histogram(
    du_lieu: pd.DataFrame,
    cot: str,
    tieu_de: str,
    duong_dan_bieu_do: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(du_lieu[cot], bins=50, kde=True, color="#2563eb")
    plt.title(tieu_de)
    plt.xlabel(cot)
    plt.ylabel("Campaign count")
    luu_bieu_do(duong_dan_bieu_do)


def ve_bar_tong(
    du_lieu: pd.DataFrame,
    cot_nhom: str,
    cot_gia_tri: str,
    tieu_de: str,
    duong_dan_bieu_do: Path,
) -> pd.DataFrame:
    bang_tong = (
        du_lieu.groupby(cot_nhom, as_index=False)[cot_gia_tri]
        .sum()
        .sort_values(cot_gia_tri, ascending=False)
    )
    chieu_cao = max(6, min(18, 0.35 * len(bang_tong) + 2))
    plt.figure(figsize=(12, chieu_cao))
    sns.barplot(data=bang_tong, x=cot_gia_tri, y=cot_nhom, color="#16a34a")
    plt.title(tieu_de)
    plt.xlabel(cot_gia_tri)
    plt.ylabel(cot_nhom)
    luu_bieu_do(duong_dan_bieu_do)
    return bang_tong


def ve_bar_trung_binh(
    du_lieu: pd.DataFrame,
    cot_nhom: str,
    cot_gia_tri: str,
    tieu_de: str,
    duong_dan_bieu_do: Path,
) -> pd.DataFrame:
    bang_trung_binh = (
        du_lieu.groupby(cot_nhom, as_index=False)[cot_gia_tri]
        .mean()
        .sort_values(cot_gia_tri, ascending=False)
    )
    chieu_cao = max(6, min(18, 0.35 * len(bang_trung_binh) + 2))
    plt.figure(figsize=(12, chieu_cao))
    sns.barplot(data=bang_trung_binh, x=cot_gia_tri, y=cot_nhom, color="#f97316")
    plt.title(tieu_de)
    plt.xlabel(f"Average {cot_gia_tri}")
    plt.ylabel(cot_nhom)
    luu_bieu_do(duong_dan_bieu_do)
    return bang_trung_binh


def ve_duong_theo_thang(
    bang_theo_thang: pd.DataFrame,
    cot_gia_tri: str,
    tieu_de: str,
    duong_dan_bieu_do: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=bang_theo_thang, x="YearMonth", y=cot_gia_tri, marker="o")
    plt.title(tieu_de)
    plt.xlabel("Month")
    plt.ylabel(cot_gia_tri)
    plt.xticks(rotation=45, ha="right")
    luu_bieu_do(duong_dan_bieu_do)


def tao_notebook_eda(duong_dan_notebook: str | Path) -> None:
    """Tạo notebook EDA để mở và chạy lại trong Jupyter."""
    duong_dan_notebook = Path(duong_dan_notebook)
    duong_dan_notebook.parent.mkdir(parents=True, exist_ok=True)

    cac_bieu_do = [
        ("1. Revenue distribution", "../results/plots/eda/01_revenue_distribution.png"),
        ("2. ROI distribution", "../results/plots/eda/02_roi_distribution.png"),
        ("3. Revenue by Brand", "../results/plots/eda/03_revenue_by_brand.png"),
        ("4. Average ROI by Brand", "../results/plots/eda/04_average_roi_by_brand.png"),
        ("5. Revenue by Channel Used", "../results/plots/eda/05_revenue_by_channel_used.png"),
        (
            "6. Average ROI by Channel Used",
            "../results/plots/eda/06_average_roi_by_channel_used.png",
        ),
        (
            "7. Revenue by Campaign Type",
            "../results/plots/eda/07_revenue_by_campaign_type.png",
        ),
        (
            "8. Revenue by Customer Segment",
            "../results/plots/eda/08_revenue_by_customer_segment.png",
        ),
        (
            "9. Revenue trend by Month",
            "../results/plots/eda/09_revenue_trend_by_month.png",
        ),
        (
            "10. Campaign count by Month",
            "../results/plots/eda/10_campaign_count_by_month.png",
        ),
        ("11. Correlation heatmap", "../results/plots/eda/11_correlation_heatmap.png"),
        (
            "12. Acquisition Cost vs Revenue scatter",
            "../results/plots/eda/12_acquisition_cost_vs_revenue_scatter.png",
        ),
        ("13. CTR by Channel Used", "../results/plots/eda/13_ctr_by_channel_used.png"),
        ("14. CVR by Channel Used", "../results/plots/eda/14_cvr_by_channel_used.png"),
        ("15. CPA by Campaign Type", "../results/plots/eda/15_cpa_by_campaign_type.png"),
        (
            "16. Revenue per Click by Brand",
            "../results/plots/eda/16_revenue_per_click_by_brand.png",
        ),
        (
            "17. Heatmap Channel Used x Campaign Type theo Revenue",
            "../results/plots/eda/17_heatmap_channel_campaign_type_revenue.png",
        ),
        ("18. Boxplot ROI by Brand", "../results/plots/eda/18_boxplot_roi_by_brand.png"),
    ]

    noi_dung_code = """import pandas as pd

du_lieu = pd.read_csv('../data/processed/marketing_campaigns_features.csv')
du_lieu.shape
"""
    cac_cell = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EDA - Multi Brand Marketing\n",
                "\n",
                "Notebook này tổng hợp 18 biểu đồ EDA đã tạo từ `marketing_campaigns_features.csv`.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": noi_dung_code.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["du_lieu.head()\n"],
        },
    ]
    for tieu_de, duong_dan_bieu_do in cac_bieu_do:
        cac_cell.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## {tieu_de}\n",
                    f"![{tieu_de}]({duong_dan_bieu_do})\n",
                ],
            }
        )

    noi_dung_notebook = {
        "cells": cac_cell,
        "metadata": {
            "kernelspec": {
                "display_name": "Multi-Brand Marketing (venv)",
                "language": "python",
                "name": "multi_brand_marketing",
            },
            "language_info": {"name": "python", "version": "3.12", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    duong_dan_notebook.write_text(
        json.dumps(noi_dung_notebook, indent=2),
        encoding="utf-8",
    )


def chay_eda() -> None:
    tham_so = phan_tich_tham_so()
    du_lieu = tai_du_lieu_feature(tham_so.duong_dan_input)

    thu_muc_bieu_do = Path(tham_so.thu_muc_bieu_do)
    thu_muc_bieu_do.mkdir(parents=True, exist_ok=True)
    duong_dan_bao_cao = Path(tham_so.duong_dan_bao_cao)
    duong_dan_bao_cao.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    cac_bieu_do: dict[str, str] = {}

    def them_bieu_do(ten: str) -> Path:
        duong_dan = thu_muc_bieu_do / ten
        cac_bieu_do[ten.replace(".png", "")] = str(duong_dan)
        return duong_dan

    ve_histogram(
        du_lieu,
        "Revenue",
        "Revenue distribution",
        them_bieu_do("01_revenue_distribution.png"),
    )
    ve_histogram(
        du_lieu,
        "ROI",
        "ROI distribution",
        them_bieu_do("02_roi_distribution.png"),
    )

    revenue_theo_brand = ve_bar_tong(
        du_lieu,
        "Brand",
        "Revenue",
        "Revenue by Brand",
        them_bieu_do("03_revenue_by_brand.png"),
    )
    roi_theo_brand = ve_bar_trung_binh(
        du_lieu,
        "Brand",
        "ROI",
        "Average ROI by Brand",
        them_bieu_do("04_average_roi_by_brand.png"),
    )
    revenue_theo_kenh = ve_bar_tong(
        du_lieu,
        "Channel_Used",
        "Revenue",
        "Revenue by Channel Used",
        them_bieu_do("05_revenue_by_channel_used.png"),
    )
    roi_theo_kenh = ve_bar_trung_binh(
        du_lieu,
        "Channel_Used",
        "ROI",
        "Average ROI by Channel Used",
        them_bieu_do("06_average_roi_by_channel_used.png"),
    )
    revenue_theo_loai = ve_bar_tong(
        du_lieu,
        "Campaign_Type",
        "Revenue",
        "Revenue by Campaign Type",
        them_bieu_do("07_revenue_by_campaign_type.png"),
    )
    revenue_theo_phan_khuc = ve_bar_tong(
        du_lieu,
        "Customer_Segment",
        "Revenue",
        "Revenue by Customer Segment",
        them_bieu_do("08_revenue_by_customer_segment.png"),
    )

    du_lieu["YearMonth"] = du_lieu["Date"].dt.to_period("M").astype(str)
    revenue_theo_thang = (
        du_lieu.groupby("YearMonth", as_index=False)["Revenue"].sum().sort_values("YearMonth")
    )
    so_campaign_theo_thang = (
        du_lieu.groupby("YearMonth", as_index=False)["Campaign_ID"]
        .count()
        .rename(columns={"Campaign_ID": "Campaign_Count"})
        .sort_values("YearMonth")
    )
    ve_duong_theo_thang(
        revenue_theo_thang,
        "Revenue",
        "Revenue trend by Month",
        them_bieu_do("09_revenue_trend_by_month.png"),
    )
    ve_duong_theo_thang(
        so_campaign_theo_thang,
        "Campaign_Count",
        "Campaign count by Month",
        them_bieu_do("10_campaign_count_by_month.png"),
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        du_lieu[CAC_COT_TUONG_QUAN].corr(),
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.4,
    )
    plt.title("Correlation heatmap")
    luu_bieu_do(them_bieu_do("11_correlation_heatmap.png"))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=du_lieu.sample(n=min(20000, len(du_lieu)), random_state=42),
        x="Acquisition_Cost",
        y="Revenue",
        hue="Brand",
        alpha=0.35,
        linewidth=0,
    )
    plt.title("Acquisition Cost vs Revenue")
    luu_bieu_do(them_bieu_do("12_acquisition_cost_vs_revenue_scatter.png"))

    ctr_theo_kenh = ve_bar_trung_binh(
        du_lieu,
        "Channel_Used",
        "CTR",
        "CTR by Channel Used",
        them_bieu_do("13_ctr_by_channel_used.png"),
    )
    cvr_theo_kenh = ve_bar_trung_binh(
        du_lieu,
        "Channel_Used",
        "Conversion_Rate",
        "CVR by Channel Used",
        them_bieu_do("14_cvr_by_channel_used.png"),
    )
    cpa_theo_loai = ve_bar_trung_binh(
        du_lieu,
        "Campaign_Type",
        "Cost_per_Conversion",
        "CPA by Campaign Type",
        them_bieu_do("15_cpa_by_campaign_type.png"),
    )
    rpc_theo_brand = ve_bar_trung_binh(
        du_lieu,
        "Brand",
        "Revenue_per_Click",
        "Revenue per Click by Brand",
        them_bieu_do("16_revenue_per_click_by_brand.png"),
    )

    bang_nhiet = du_lieu.pivot_table(
        index="Channel_Used",
        columns="Campaign_Type",
        values="Revenue",
        aggfunc="sum",
    ).fillna(0)
    plt.figure(figsize=(12, 14))
    sns.heatmap(bang_nhiet, cmap="YlGnBu", linewidths=0.3)
    plt.title("Revenue heatmap: Channel Used x Campaign Type")
    plt.xlabel("Campaign_Type")
    plt.ylabel("Channel_Used")
    luu_bieu_do(them_bieu_do("17_heatmap_channel_campaign_type_revenue.png"))

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=du_lieu, x="Brand", y="ROI", color="#60a5fa")
    plt.title("Boxplot ROI by Brand")
    plt.xlabel("Brand")
    plt.ylabel("ROI")
    luu_bieu_do(them_bieu_do("18_boxplot_roi_by_brand.png"))

    bao_cao: dict[str, Any] = {
        "rows": int(len(du_lieu)),
        "columns": int(len(du_lieu.columns)),
        "plot_count": len(cac_bieu_do),
        "plots": cac_bieu_do,
        "top_revenue_brand": revenue_theo_brand.head(1).to_dict(orient="records"),
        "top_average_roi_brand": roi_theo_brand.head(1).to_dict(orient="records"),
        "top_revenue_channel": revenue_theo_kenh.head(1).to_dict(orient="records"),
        "top_average_roi_channel": roi_theo_kenh.head(1).to_dict(orient="records"),
        "top_revenue_campaign_type": revenue_theo_loai.head(1).to_dict(orient="records"),
        "top_revenue_customer_segment": revenue_theo_phan_khuc.head(1).to_dict(
            orient="records"
        ),
        "top_ctr_channel": ctr_theo_kenh.head(1).to_dict(orient="records"),
        "top_cvr_channel": cvr_theo_kenh.head(1).to_dict(orient="records"),
        "lowest_cpa_campaign_type": cpa_theo_loai.tail(1).to_dict(orient="records"),
        "top_revenue_per_click_brand": rpc_theo_brand.head(1).to_dict(orient="records"),
    }
    duong_dan_bao_cao.write_text(json.dumps(bao_cao, indent=2), encoding="utf-8")
    tao_notebook_eda(tham_so.duong_dan_notebook)

    print("Hoàn tất EDA")
    print(f"Số biểu đồ đã tạo: {len(cac_bieu_do)}")
    print(f"Đã lưu biểu đồ: {thu_muc_bieu_do}")
    print(f"Đã lưu báo cáo EDA: {duong_dan_bao_cao}")
    print(f"Đã lưu notebook EDA: {Path(tham_so.duong_dan_notebook)}")


if __name__ == "__main__":
    chay_eda()
