"""Bổ sung EDA business, notebook và báo cáo marketing đa thương hiệu."""

from __future__ import annotations

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


DUONG_DAN_DU_LIEU = Path("data/processed/marketing_campaigns_features.csv")
THU_MUC_BIEU_DO = Path("results/plots/business")
DUONG_DAN_TONG_HOP_EDA = Path("results/metrics/eda_summary.json")
DUONG_DAN_TONG_HOP = Path("results/metrics/business_summary.json")
DUONG_DAN_NOTEBOOK = Path("notebooks/02_business_analysis.ipynb")
DUONG_DAN_BAO_CAO = Path("reports/marketing_campaign_analysis_report.md")


def doi_thanh_ban_ghi(bang_du_lieu: pd.DataFrame) -> list[dict[str, Any]]:
    """Đổi DataFrame thành danh sách bản ghi có kiểu dữ liệu JSON hợp lệ."""
    return json.loads(bang_du_lieu.to_json(orient="records"))


def luu_bieu_do(ten_tep: str) -> str:
    """Lưu biểu đồ hiện tại và trả về đường dẫn tương đối."""
    THU_MUC_BIEU_DO.mkdir(parents=True, exist_ok=True)
    duong_dan = THU_MUC_BIEU_DO / ten_tep
    plt.tight_layout()
    plt.savefig(duong_dan, dpi=160, bbox_inches="tight")
    plt.close()
    return str(duong_dan)


def ve_bar_ngang(
    bang_du_lieu: pd.DataFrame,
    cot_nhom: str,
    cot_gia_tri: str,
    tieu_de: str,
    ten_tep: str,
    mau: str = "#2563eb",
) -> str:
    """Vẽ biểu đồ thanh ngang cho KPI theo nhóm."""
    chieu_cao = max(6, min(18, len(bang_du_lieu) * 0.35 + 2))
    plt.figure(figsize=(12, chieu_cao))
    sns.barplot(data=bang_du_lieu, x=cot_gia_tri, y=cot_nhom, color=mau)
    plt.title(tieu_de)
    plt.xlabel(cot_gia_tri)
    plt.ylabel(cot_nhom)
    return luu_bieu_do(ten_tep)


def ve_duong_thang(
    bang_du_lieu: pd.DataFrame,
    cot_gia_tri: str,
    tieu_de: str,
    ten_tep: str,
) -> str:
    """Vẽ xu hướng KPI theo tháng."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=bang_du_lieu, x="YearMonth", y=cot_gia_tri, marker="o")
    plt.title(tieu_de)
    plt.xlabel("Month")
    plt.ylabel(cot_gia_tri)
    plt.xticks(rotation=45, ha="right")
    return luu_bieu_do(ten_tep)


def tao_cac_bang_phan_tich(du_lieu: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tạo các bảng KPI theo brand, channel, campaign type, segment và tháng."""
    cac_bang: dict[str, pd.DataFrame] = {}
    for ten, cot_nhom in [
        ("brand", "Brand"),
        ("channel", "Channel_Used"),
        ("campaign_type", "Campaign_Type"),
        ("segment", "Customer_Segment"),
    ]:
        cac_bang[ten] = (
            du_lieu.groupby(cot_nhom, as_index=False)
            .agg(
                Total_Revenue=("Revenue", "sum"),
                Average_ROI=("ROI", "mean"),
                Total_Acquisition_Cost=("Acquisition_Cost", "sum"),
                Average_CTR=("CTR", "mean"),
                Average_CVR=("Conversion_Rate", "mean"),
                Average_CPA=("Cost_per_Conversion", "mean"),
                Revenue_per_Click=("Revenue_per_Click", "mean"),
                Total_Campaigns=("Campaign_ID", "count"),
            )
            .sort_values("Total_Revenue", ascending=False)
        )

    cac_bang["month"] = (
        du_lieu.groupby("YearMonth", as_index=False)
        .agg(
            Total_Revenue=("Revenue", "sum"),
            Average_ROI=("ROI", "mean"),
            Total_Acquisition_Cost=("Acquisition_Cost", "sum"),
            Total_Campaigns=("Campaign_ID", "count"),
        )
        .sort_values("YearMonth")
    )
    return cac_bang


def tao_bieu_do_bo_sung(
    du_lieu: pd.DataFrame,
    cac_bang: dict[str, pd.DataFrame],
) -> dict[str, str]:
    """Tạo các biểu đồ business còn thiếu so với EDA ban đầu."""
    cac_bieu_do: dict[str, str] = {}

    cac_bieu_do["19_acquisition_cost_by_brand"] = ve_bar_ngang(
        cac_bang["brand"].sort_values("Total_Acquisition_Cost", ascending=False),
        "Brand",
        "Total_Acquisition_Cost",
        "Acquisition Cost by Brand",
        "19_acquisition_cost_by_brand.png",
        "#dc2626",
    )
    cac_bieu_do["20_average_cpa_by_brand"] = ve_bar_ngang(
        cac_bang["brand"].sort_values("Average_CPA", ascending=True),
        "Brand",
        "Average_CPA",
        "Average CPA by Brand",
        "20_average_cpa_by_brand.png",
        "#f97316",
    )

    bang_ctr_cvr = cac_bang["brand"].melt(
        id_vars="Brand",
        value_vars=["Average_CTR", "Average_CVR"],
        var_name="KPI",
        value_name="Value",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bang_ctr_cvr, x="Brand", y="Value", hue="KPI")
    plt.title("CTR and CVR by Brand")
    cac_bieu_do["21_ctr_cvr_by_brand"] = luu_bieu_do("21_ctr_cvr_by_brand.png")

    cac_bieu_do["22_average_roi_by_campaign_type"] = ve_bar_ngang(
        cac_bang["campaign_type"].sort_values("Average_ROI", ascending=False),
        "Campaign_Type",
        "Average_ROI",
        "Average ROI by Campaign Type",
        "22_average_roi_by_campaign_type.png",
        "#16a34a",
    )
    cac_bieu_do["23_campaign_count_by_campaign_type"] = ve_bar_ngang(
        cac_bang["campaign_type"].sort_values("Total_Campaigns", ascending=False),
        "Campaign_Type",
        "Total_Campaigns",
        "Campaign Count by Campaign Type",
        "23_campaign_count_by_campaign_type.png",
        "#7c3aed",
    )
    cac_bieu_do["24_roi_by_customer_segment"] = ve_bar_ngang(
        cac_bang["segment"].sort_values("Average_ROI", ascending=False),
        "Customer_Segment",
        "Average_ROI",
        "Average ROI by Customer Segment",
        "24_roi_by_customer_segment.png",
        "#16a34a",
    )
    cac_bieu_do["25_cvr_by_customer_segment"] = ve_bar_ngang(
        cac_bang["segment"].sort_values("Average_CVR", ascending=False),
        "Customer_Segment",
        "Average_CVR",
        "Average CVR by Customer Segment",
        "25_cvr_by_customer_segment.png",
        "#2563eb",
    )

    bang_nhiet_brand_segment = du_lieu.pivot_table(
        index="Brand",
        columns="Customer_Segment",
        values="Revenue",
        aggfunc="sum",
    )
    plt.figure(figsize=(12, 5))
    sns.heatmap(bang_nhiet_brand_segment, cmap="YlGnBu", annot=True, fmt=".3g")
    plt.title("Revenue Heatmap: Brand x Customer Segment")
    cac_bieu_do["26_heatmap_brand_customer_segment_revenue"] = luu_bieu_do(
        "26_heatmap_brand_customer_segment_revenue.png"
    )

    cac_bieu_do["27_roi_trend_by_month"] = ve_duong_thang(
        cac_bang["month"],
        "Average_ROI",
        "Average ROI Trend by Month",
        "27_roi_trend_by_month.png",
    )
    cac_bieu_do["28_acquisition_cost_trend_by_month"] = ve_duong_thang(
        cac_bang["month"],
        "Total_Acquisition_Cost",
        "Acquisition Cost Trend by Month",
        "28_acquisition_cost_trend_by_month.png",
    )

    bang_nhiet_brand_channel = du_lieu.pivot_table(
        index="Channel_Used",
        columns="Brand",
        values="Revenue",
        aggfunc="sum",
    ).fillna(0)
    plt.figure(figsize=(10, 14))
    sns.heatmap(bang_nhiet_brand_channel, cmap="YlOrRd", linewidths=0.3)
    plt.title("Revenue Heatmap: Brand x Channel")
    cac_bieu_do["29_heatmap_brand_channel_revenue"] = luu_bieu_do(
        "29_heatmap_brand_channel_revenue.png"
    )
    cac_bieu_do["30_cpa_by_channel_used"] = ve_bar_ngang(
        cac_bang["channel"].sort_values("Average_CPA", ascending=True),
        "Channel_Used",
        "Average_CPA",
        "Average CPA by Channel Used",
        "30_cpa_by_channel_used.png",
        "#f97316",
    )
    return cac_bieu_do


def tao_tong_hop(
    du_lieu: pd.DataFrame,
    cac_bang: dict[str, pd.DataFrame],
    cac_bieu_do_bo_sung: dict[str, str],
) -> dict[str, Any]:
    """Tạo summary JSON gồm KPI và bảng xếp hạng business."""
    kpi = {
        "total_revenue": int(du_lieu["Revenue"].sum()),
        "average_roi": float(du_lieu["ROI"].mean()),
        "total_acquisition_cost": float(du_lieu["Acquisition_Cost"].sum()),
        "average_ctr": float(du_lieu["CTR"].mean()),
        "average_cvr": float(du_lieu["Conversion_Rate"].mean()),
        "average_cpa": float(du_lieu["Cost_per_Conversion"].mean()),
        "total_campaigns": int(len(du_lieu)),
        "revenue_per_click": float(du_lieu["Revenue_per_Click"].mean()),
    }
    duong_dan_cu = {}
    if DUONG_DAN_TONG_HOP_EDA.exists():
        duong_dan_cu = json.loads(
            DUONG_DAN_TONG_HOP_EDA.read_text(encoding="utf-8")
        ).get(
            "plots", {}
        )

    return {
        "project_name": "Multi-Brand Marketing Campaign Performance Analysis",
        "analysis_type": "Data Analysis / EDA / Business Analytics",
        "rows": int(len(du_lieu)),
        "columns": int(len(du_lieu.columns)),
        "date_min": str(du_lieu["Date"].min().date()),
        "date_max": str(du_lieu["Date"].max().date()),
        "kpi_summary": kpi,
        "brand_performance": doi_thanh_ban_ghi(cac_bang["brand"]),
        "channel_performance": doi_thanh_ban_ghi(cac_bang["channel"]),
        "campaign_type_performance": doi_thanh_ban_ghi(cac_bang["campaign_type"]),
        "customer_segment_performance": doi_thanh_ban_ghi(cac_bang["segment"]),
        "monthly_performance": doi_thanh_ban_ghi(cac_bang["month"]),
        "plots": {**duong_dan_cu, **cac_bieu_do_bo_sung},
        "caveats": [
            "Dữ liệu tháng 06/2025 chỉ đến ngày 24/06/2025 nên không so sánh trực tiếp tổng tháng với các tháng đầy đủ.",
            "Chênh lệch KPI giữa các brand và nhiều nhóm khá nhỏ; cần kiểm định thêm trước khi thay đổi ngân sách lớn.",
            "Phân tích hiện tại mô tả quan hệ, không chứng minh quan hệ nhân quả.",
        ],
    }


def dinh_dang_tien(gia_tri: float | int) -> str:
    """Định dạng số tiền với dấu phân cách hàng nghìn."""
    return f"{gia_tri:,.0f}"


def tao_bao_cao_markdown(tong_hop: dict[str, Any]) -> None:
    """Tạo báo cáo business analysis bằng Markdown."""
    DUONG_DAN_BAO_CAO.parent.mkdir(parents=True, exist_ok=True)
    kpi = tong_hop["kpi_summary"]
    brand = tong_hop["brand_performance"]
    channel = tong_hop["channel_performance"]
    campaign = tong_hop["campaign_type_performance"]
    segment = tong_hop["customer_segment_performance"]
    thang = tong_hop["monthly_performance"]

    brand_revenue_top = max(brand, key=lambda dong: dong["Total_Revenue"])
    brand_roi_top = max(brand, key=lambda dong: dong["Average_ROI"])
    brand_cpa_top = min(brand, key=lambda dong: dong["Average_CPA"])
    channel_revenue_top = max(channel, key=lambda dong: dong["Total_Revenue"])
    channel_roi_top = max(channel, key=lambda dong: dong["Average_ROI"])
    channel_cpa_top = min(channel, key=lambda dong: dong["Average_CPA"])
    campaign_revenue_top = max(campaign, key=lambda dong: dong["Total_Revenue"])
    campaign_roi_top = max(campaign, key=lambda dong: dong["Average_ROI"])
    campaign_cpa_top = min(campaign, key=lambda dong: dong["Average_CPA"])
    segment_revenue_top = max(segment, key=lambda dong: dong["Total_Revenue"])
    segment_roi_top = max(segment, key=lambda dong: dong["Average_ROI"])
    segment_cvr_top = max(segment, key=lambda dong: dong["Average_CVR"])
    thang_revenue_top = max(thang, key=lambda dong: dong["Total_Revenue"])
    thang_roi_top = max(thang, key=lambda dong: dong["Average_ROI"])

    noi_dung = f"""# Multi-Brand Marketing Campaign Performance Analysis

## 1. Project Overview

Phân tích hiệu quả chiến dịch marketing của Nykaa, Purplle và Tira nhằm đánh giá hiệu suất theo brand, channel, campaign type, customer segment và thời gian. Đây là bài toán Data Analysis / EDA / Business Analytics; chưa sử dụng Machine Learning.

## 2. Dataset Overview

- Số chiến dịch: **{kpi['total_campaigns']:,}**
- Số brand: **3**
- Khoảng thời gian: **{tong_hop['date_min']} đến {tong_hop['date_max']}**
- Dữ liệu sau cleaning và feature engineering không có missing values.
- Lưu ý: tháng 06/2025 chỉ có dữ liệu đến ngày 24/06/2025.

## 3. Key KPI Summary

| KPI | Giá trị |
|---|---:|
| Total Revenue | {dinh_dang_tien(kpi['total_revenue'])} |
| Average ROI | {kpi['average_roi']:.3f} |
| Total Acquisition Cost | {dinh_dang_tien(kpi['total_acquisition_cost'])} |
| Average CTR | {kpi['average_ctr']:.2%} |
| Average CVR | {kpi['average_cvr']:.2%} |
| Average CPA | {kpi['average_cpa']:.3f} |
| Total Campaigns | {kpi['total_campaigns']:,} |
| Revenue per Click | {kpi['revenue_per_click']:.3f} |

## 4. Brand Performance Analysis

- **{brand_revenue_top['Brand']}** có tổng Revenue cao nhất: **{dinh_dang_tien(brand_revenue_top['Total_Revenue'])}**.
- **{brand_roi_top['Brand']}** có Average ROI cao nhất: **{brand_roi_top['Average_ROI']:.3f}**.
- **{brand_cpa_top['Brand']}** có Average CPA thấp nhất: **{brand_cpa_top['Average_CPA']:.3f}**.
- Khoảng cách hiệu suất giữa ba brand tương đối nhỏ; Nykaa dẫn đầu nhưng chưa vượt trội tuyệt đối.

![Revenue by Brand](../results/plots/eda/03_revenue_by_brand.png)

![CTR and CVR by Brand](../results/plots/business/21_ctr_cvr_by_brand.png)

## 5. Channel Performance Analysis

- **{channel_revenue_top['Channel_Used']}** tạo tổng Revenue cao nhất: **{dinh_dang_tien(channel_revenue_top['Total_Revenue'])}**.
- **{channel_roi_top['Channel_Used']}** có Average ROI cao nhất: **{channel_roi_top['Average_ROI']:.3f}**.
- **{channel_cpa_top['Channel_Used']}** có Average CPA thấp nhất: **{channel_cpa_top['Average_CPA']:.3f}**.
- Nên đánh giá đồng thời ROI, CPA và quy mô campaign; channel có ROI cao nhưng volume thấp chưa chắc phù hợp để scale ngay.

![Revenue by Channel](../results/plots/eda/05_revenue_by_channel_used.png)

![CPA by Channel](../results/plots/business/30_cpa_by_channel_used.png)

## 6. Campaign Type Analysis

- **{campaign_revenue_top['Campaign_Type']}** tạo Revenue cao nhất: **{dinh_dang_tien(campaign_revenue_top['Total_Revenue'])}**.
- **{campaign_roi_top['Campaign_Type']}** có Average ROI cao nhất: **{campaign_roi_top['Average_ROI']:.3f}**.
- **{campaign_cpa_top['Campaign_Type']}** có Average CPA thấp nhất: **{campaign_cpa_top['Average_CPA']:.3f}**.
- Paid Ads đang có tổ hợp Revenue, ROI và CPA tốt, phù hợp để thử nghiệm tăng ngân sách có kiểm soát.

![Revenue by Campaign Type](../results/plots/eda/07_revenue_by_campaign_type.png)

![Average ROI by Campaign Type](../results/plots/business/22_average_roi_by_campaign_type.png)

## 7. Customer Segment Analysis

- **{segment_revenue_top['Customer_Segment']}** tạo Revenue cao nhất: **{dinh_dang_tien(segment_revenue_top['Total_Revenue'])}**.
- **{segment_roi_top['Customer_Segment']}** có Average ROI cao nhất: **{segment_roi_top['Average_ROI']:.3f}**.
- **{segment_cvr_top['Customer_Segment']}** có Average CVR cao nhất: **{segment_cvr_top['Average_CVR']:.2%}**.

![Revenue by Customer Segment](../results/plots/eda/08_revenue_by_customer_segment.png)

![Brand x Customer Segment](../results/plots/business/26_heatmap_brand_customer_segment_revenue.png)

## 8. Monthly Trend Analysis

- Tháng có Revenue cao nhất là **{thang_revenue_top['YearMonth']}** với **{dinh_dang_tien(thang_revenue_top['Total_Revenue'])}**.
- Tháng có Average ROI cao nhất là **{thang_roi_top['YearMonth']}** với ROI **{thang_roi_top['Average_ROI']:.3f}**.
- Revenue và campaign count nhìn chung ổn định giữa các tháng đầy đủ.
- Không nên kết luận Revenue giảm mạnh trong 06/2025 vì tháng này chưa đủ dữ liệu.

![Revenue Trend](../results/plots/eda/09_revenue_trend_by_month.png)

![ROI Trend](../results/plots/business/27_roi_trend_by_month.png)

## 9. Key Insights

1. Nykaa dẫn đầu cả Total Revenue và Average ROI, nhưng mức chênh lệch so với Purplle và Tira khá nhỏ.
2. Email tạo Revenue tổng cao nhất, trong khi tổ hợp channel có thể đạt ROI hoặc CPA tốt hơn ở quy mô campaign thấp hơn.
3. Paid Ads là campaign type đáng chú ý nhất nhờ Revenue và ROI cao, đồng thời CPA thấp nhất.
4. College Students tạo Revenue cao nhất; Working Women có ROI tốt; Youth có CVR tốt nhất.
5. Revenue theo tháng tương đối ổn định. Dữ liệu tháng 06/2025 là dữ liệu tháng chưa hoàn chỉnh.
6. Acquisition Cost cao không tự động đảm bảo Revenue cao; cần tối ưu theo ROI và CPA thay vì chỉ tăng ngân sách.

## 10. Business Recommendations

1. Tăng ngân sách thử nghiệm cho Nykaa và Paid Ads theo từng đợt nhỏ, theo dõi ROI và CPA trước khi scale lớn.
2. Duy trì Email để bảo vệ quy mô Revenue; đồng thời thử nghiệm các channel combination có ROI cao và CPA thấp.
3. Ưu tiên College Students cho mục tiêu Revenue, Working Women cho mục tiêu ROI và Youth cho mục tiêu conversion.
4. Thiết lập dashboard theo tháng cho Revenue, ROI, CPA, CTR và CVR; không đánh giá tháng chưa hoàn chỉnh như tháng đầy đủ.
5. Rà soát các campaign có Acquisition Cost cao nhưng Revenue thấp để điều chỉnh targeting, creative hoặc ngân sách.

## Caveats

- Phân tích hiện tại là mô tả, không chứng minh quan hệ nhân quả.
- Các nhóm có số lượng campaign khác nhau; cần xem xét volume trước khi scale.
- Chênh lệch KPI nhỏ cần được kiểm định thêm bằng A/B testing hoặc statistical testing.
"""
    DUONG_DAN_BAO_CAO.write_text(noi_dung, encoding="utf-8")


def tao_notebook(tong_hop: dict[str, Any]) -> None:
    """Tạo notebook theo cấu trúc đề tài đã thống nhất."""
    DUONG_DAN_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    kpi = tong_hop["kpi_summary"]
    cac_phan = [
        ("9. Brand Analysis", ["03_revenue_by_brand.png", "04_average_roi_by_brand.png", "19_acquisition_cost_by_brand.png", "20_average_cpa_by_brand.png", "21_ctr_cvr_by_brand.png"]),
        ("10. Channel Analysis", ["05_revenue_by_channel_used.png", "06_average_roi_by_channel_used.png", "13_ctr_by_channel_used.png", "14_cvr_by_channel_used.png", "30_cpa_by_channel_used.png"]),
        ("11. Campaign Type Analysis", ["07_revenue_by_campaign_type.png", "15_cpa_by_campaign_type.png", "22_average_roi_by_campaign_type.png", "23_campaign_count_by_campaign_type.png"]),
        ("12. Customer Segment Analysis", ["08_revenue_by_customer_segment.png", "24_roi_by_customer_segment.png", "25_cvr_by_customer_segment.png", "26_heatmap_brand_customer_segment_revenue.png"]),
        ("13. Time Trend Analysis", ["09_revenue_trend_by_month.png", "10_campaign_count_by_month.png", "27_roi_trend_by_month.png", "28_acquisition_cost_trend_by_month.png"]),
        ("14. Correlation Analysis", ["11_correlation_heatmap.png", "12_acquisition_cost_vs_revenue_scatter.png"]),
        ("15. Insights and Recommendations", ["17_heatmap_channel_campaign_type_revenue.png", "18_boxplot_roi_by_brand.png", "29_heatmap_brand_channel_revenue.png"]),
    ]

    cac_cell: list[dict[str, Any]] = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Multi-Brand Marketing Campaign Performance Analysis\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Import libraries\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import pandas as pd\n", "import matplotlib.pyplot as plt\n", "import seaborn as sns\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Load dataset\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["du_lieu = pd.read_csv('../data/processed/marketing_campaigns_features.csv')\n", "du_lieu['Date'] = pd.to_datetime(du_lieu['Date'])\n", "du_lieu.head()\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Dataset overview\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["du_lieu.shape\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Check missing values\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["du_lieu.isna().sum()\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Check duplicated rows\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["du_lieu.duplicated().sum()\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 6. Check data types\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["du_lieu.dtypes\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 7. Summary statistics\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["du_lieu.describe().T\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 8. KPI overview\n", f"- Total Revenue: **{dinh_dang_tien(kpi['total_revenue'])}**\n", f"- Average ROI: **{kpi['average_roi']:.3f}**\n", f"- Average CTR: **{kpi['average_ctr']:.2%}**\n", f"- Average CVR: **{kpi['average_cvr']:.2%}**\n", f"- Average CPA: **{kpi['average_cpa']:.3f}**\n"]},
    ]
    for tieu_de, cac_tep in cac_phan:
        cac_cell.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {tieu_de}\n"]})
        for ten_tep in cac_tep:
            cac_cell.append({"cell_type": "markdown", "metadata": {}, "source": [f"![{ten_tep}](../results/plots/{'eda' if int(ten_tep[:2]) <= 18 else 'business'}/{ten_tep})\n"]})

    notebook = {
        "cells": cac_cell,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DUONG_DAN_NOTEBOOK.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def chay_phan_tich_business() -> None:
    """Chạy toàn bộ bước bổ sung business analysis và tạo deliverables."""
    sns.set_theme(style="whitegrid")
    du_lieu = pd.read_csv(DUONG_DAN_DU_LIEU)
    du_lieu["Date"] = pd.to_datetime(du_lieu["Date"])
    du_lieu["YearMonth"] = du_lieu["Date"].dt.to_period("M").astype(str)

    cac_bang = tao_cac_bang_phan_tich(du_lieu)
    cac_bieu_do_bo_sung = tao_bieu_do_bo_sung(du_lieu, cac_bang)
    tong_hop = tao_tong_hop(du_lieu, cac_bang, cac_bieu_do_bo_sung)

    DUONG_DAN_TONG_HOP.parent.mkdir(parents=True, exist_ok=True)
    DUONG_DAN_TONG_HOP.write_text(json.dumps(tong_hop, indent=2), encoding="utf-8")
    tao_bao_cao_markdown(tong_hop)
    tao_notebook(tong_hop)

    print("Hoàn tất business analysis")
    print(f"Biểu đồ bổ sung: {len(cac_bieu_do_bo_sung)}")
    print(f"Đã lưu summary: {DUONG_DAN_TONG_HOP}")
    print(f"Đã lưu notebook: {DUONG_DAN_NOTEBOOK}")
    print(f"Đã lưu report: {DUONG_DAN_BAO_CAO}")


if __name__ == "__main__":
    chay_phan_tich_business()
