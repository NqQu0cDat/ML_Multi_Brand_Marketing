"""Train Revenue Prediction models cho campaign marketing."""

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

from utils.revenue_prediction import (
    CAC_COT_CAM,
    SCENARIO_CONFIG,
    KetQuaScenario,
    du_doan_revenue,
    ghi_json,
    luu_model,
    tai_du_lieu_revenue,
    train_scenario,
)

DUONG_DAN_DU_LIEU = Path("data/processed/marketing_campaigns_features.csv")
THU_MUC_MODEL = Path("models/revenue_prediction")
THU_MUC_BIEU_DO = Path("results/plots/ml")
DUONG_DAN_SUMMARY = Path("results/metrics/revenue_prediction_summary.json")
DUONG_DAN_REPORT = Path("reports/revenue_prediction_report.md")
DUONG_DAN_NOTEBOOK = Path("notebooks/03_revenue_prediction.ipynb")


def phan_tich_tham_so() -> argparse.Namespace:
    """Đọc tham số dòng lệnh cho pipeline ML."""
    bo_phan_tich = argparse.ArgumentParser(
        description="Train Revenue Prediction models cho campaign marketing."
    )
    bo_phan_tich.add_argument(
        "--duong-dan-input",
        "--input-csv",
        default=str(DUONG_DAN_DU_LIEU),
        help="Đường dẫn file marketing_campaigns_features.csv.",
    )
    return bo_phan_tich.parse_args()


def luu_bieu_do(duong_dan: Path) -> None:
    """Lưu biểu đồ hiện tại."""
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(duong_dan, dpi=160, bbox_inches="tight")
    plt.close()


def ve_actual_vs_predicted(
    ket_qua: KetQuaScenario,
    duong_dan: Path,
    tieu_de: str,
) -> None:
    """Vẽ actual vs predicted revenue."""
    mau = pd.DataFrame(
        {
            "Actual Revenue": ket_qua.y_test,
            "Predicted Revenue": ket_qua.y_pred_best,
        }
    )
    if len(mau) > 20000:
        mau = mau.sample(n=20000, random_state=42)
    gioi_han = max(mau["Actual Revenue"].max(), mau["Predicted Revenue"].max())
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=mau, x="Actual Revenue", y="Predicted Revenue", alpha=0.35, linewidth=0)
    plt.plot([0, gioi_han], [0, gioi_han], color="#dc2626", linestyle="--")
    plt.title(tieu_de)
    luu_bieu_do(duong_dan)


def ve_residual_distribution(
    ket_qua: KetQuaScenario,
    duong_dan: Path,
    tieu_de: str,
) -> None:
    """Vẽ phân phối residual."""
    residual = ket_qua.y_test - ket_qua.y_pred_best
    plt.figure(figsize=(10, 6))
    sns.histplot(residual, bins=60, kde=True, color="#2563eb")
    plt.title(tieu_de)
    plt.xlabel("Actual Revenue - Predicted Revenue")
    luu_bieu_do(duong_dan)


def ve_model_comparison(
    ket_qua_scenarios: dict[str, KetQuaScenario],
    duong_dan: Path,
) -> None:
    """Vẽ so sánh RMSE giữa các model."""
    ban_ghi = []
    for scenario, ket_qua in ket_qua_scenarios.items():
        for ten_model, metrics in ket_qua.metrics.items():
            ban_ghi.append(
                {
                    "Scenario": SCENARIO_CONFIG[scenario]["ten_hien_thi"],
                    "Model": ten_model,
                    "RMSE": metrics["rmse"],
                }
            )
    bang = pd.DataFrame(ban_ghi)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=bang, x="Model", y="RMSE", hue="Scenario")
    plt.title("Model Comparison by RMSE")
    plt.xticks(rotation=20, ha="right")
    luu_bieu_do(duong_dan)


def ve_feature_importance(
    ket_qua_scenarios: dict[str, KetQuaScenario],
    duong_dan: Path,
) -> bool:
    """Vẽ feature importance của best model nếu có hỗ trợ."""
    ban_ghi = []
    for scenario, ket_qua in ket_qua_scenarios.items():
        for dong in ket_qua.feature_importance[:15]:
            ban_ghi.append(
                {
                    "Scenario": SCENARIO_CONFIG[scenario]["ten_hien_thi"],
                    "Feature": dong["feature"],
                    "Importance": dong["importance"],
                }
            )
    if not ban_ghi:
        return False
    bang = pd.DataFrame(ban_ghi)
    plt.figure(figsize=(12, max(6, len(bang) * 0.22)))
    sns.barplot(data=bang, x="Importance", y="Feature", hue="Scenario")
    plt.title("Feature Importance of Best Models")
    luu_bieu_do(duong_dan)
    return True


def tao_summary(
    du_lieu: pd.DataFrame,
    ket_qua_scenarios: dict[str, KetQuaScenario],
    plots: dict[str, str],
) -> dict[str, Any]:
    """Tạo summary JSON cho Revenue Prediction."""
    summary: dict[str, Any] = {
        "project_name": "Revenue Prediction for Marketing Campaigns",
        "target": "Revenue",
        "rows": int(len(du_lieu)),
        "columns": int(len(du_lieu.columns)),
        "leakage_prevention": {
            "excluded_columns": CAC_COT_CAM,
            "note": "Không dùng ROI hoặc bất kỳ feature nào được tính trực tiếp từ Revenue.",
        },
        "xgboost_available": "xgboost" in next(iter(ket_qua_scenarios.values())).metrics,
        "plots": plots,
        "scenarios": {},
    }
    for scenario, ket_qua in ket_qua_scenarios.items():
        summary["scenarios"][scenario] = {
            "display_name": SCENARIO_CONFIG[scenario]["ten_hien_thi"],
            "description": SCENARIO_CONFIG[scenario]["mo_ta"],
            "late_stage_prediction": SCENARIO_CONFIG[scenario]["late_stage"],
            "features": ket_qua.feature_columns,
            "categorical_features": ket_qua.categorical_features,
            "numeric_features": ket_qua.numeric_features,
            "split_info": ket_qua.split_info,
            "metrics": ket_qua.metrics,
            "best_model": ket_qua.best_model_name,
            "best_metrics": ket_qua.metrics[ket_qua.best_model_name],
            "feature_importance": ket_qua.feature_importance,
        }
    return summary


def dinh_dang_metric(gia_tri: float) -> str:
    """Định dạng metric trong report."""
    if pd.isna(gia_tri):
        return "N/A"
    return f"{gia_tri:,.4f}"


def tao_bang_metrics_markdown(metrics: dict[str, dict[str, float]]) -> str:
    """Tạo bảng Markdown cho metrics."""
    dong = ["| Model | MAE | RMSE | R2 | MAPE |", "|---|---:|---:|---:|---:|"]
    for ten_model, gia_tri in metrics.items():
        dong.append(
            "| "
            f"{ten_model} | "
            f"{dinh_dang_metric(gia_tri['mae'])} | "
            f"{dinh_dang_metric(gia_tri['rmse'])} | "
            f"{dinh_dang_metric(gia_tri['r2'])} | "
            f"{dinh_dang_metric(gia_tri['mape'])}% |"
        )
    return "\n".join(dong)


def mo_ta_split(split_info: dict[str, Any]) -> str:
    """Mô tả chiến lược split thực tế để đưa vào report."""
    if split_info.get("strategy") == "random_fallback":
        return (
            "Random fallback 80/20 với random_state="
            f"{split_info.get('random_state')}. Lý do: {split_info.get('fallback_reason')}"
        )
    return (
        "Temporal split với ranh giới ngày nghiêm ngặt tại "
        f"{split_info.get('cutoff_date')}; train_date_max={split_info.get('train_date_max')} "
        f"và test_date_min={split_info.get('test_date_min')}."
    )


def tao_report(summary: dict[str, Any]) -> None:
    """Tạo report Revenue Prediction bằng Markdown."""
    DUONG_DAN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pre = summary["scenarios"]["pre_campaign"]
    after = summary["scenarios"]["after_funnel"]
    noi_dung = f"""# Revenue Prediction for Marketing Campaigns

## 1. Project Overview

Đề tài này mở rộng project Multi-Brand Marketing Campaign Performance Analysis sang Machine Learning để dự đoán Revenue của campaign. Pipeline ML được tách riêng và không thay đổi logic EDA / Business Analytics hiện tại.

## 2. ML Problem Definition

- Bài toán: Regression.
- Target: `Revenue`.
- Mục tiêu: dự đoán doanh thu campaign và so sánh hai bối cảnh dự đoán.

## 3. Dataset and Target

- Dataset: `data/processed/marketing_campaigns_features.csv`.
- Số dòng: **{summary['rows']:,}**.
- Số cột: **{summary['columns']:,}**.
- Target được train bằng `log1p(Revenue)` và đổi ngược bằng `expm1()` khi evaluate/predict.

## 4. Feature Sets

### Pre-campaign

Feature dùng trước khi campaign chạy:

```text
{', '.join(pre['features'])}
```

### After-funnel

Feature late-stage, chỉ dùng khi đã có funnel metrics:

```text
{', '.join(after['features'])}
```

## 5. Leakage Prevention

Các cột bị loại khỏi cả hai scenario:

```text
{', '.join(summary['leakage_prevention']['excluded_columns'])}
```

Pre-campaign model không dùng Impressions, Clicks, Leads, Conversions hoặc KPI sinh từ funnel. After-funnel model phải được hiểu là late-stage prediction, không phải dự đoán trước campaign.

## 6. Train/Test Split Strategy

- Pre-campaign: **{mo_ta_split(pre['split_info'])}**
- After-funnel: **{mo_ta_split(after['split_info'])}**
- Pipeline không cho cùng một ngày xuất hiện ở cả train và test khi temporal split.

## 7. Model Comparison

### Pre-campaign

{tao_bang_metrics_markdown(pre['metrics'])}

### After-funnel

{tao_bang_metrics_markdown(after['metrics'])}

![Model Comparison RMSE](../results/plots/ml/35_model_comparison_rmse.png)

## 8. Best Model for Pre-campaign

- Best model: **{pre['best_model']}**.
- RMSE: **{dinh_dang_metric(pre['best_metrics']['rmse'])}**.
- MAE: **{dinh_dang_metric(pre['best_metrics']['mae'])}**.
- R2: **{dinh_dang_metric(pre['best_metrics']['r2'])}**.
- MAPE: **{dinh_dang_metric(pre['best_metrics']['mape'])}%**.

![Pre-campaign Actual vs Predicted](../results/plots/ml/31_actual_vs_predicted_pre_campaign.png)

![Pre-campaign Residual](../results/plots/ml/32_residual_distribution_pre_campaign.png)

## 9. Best Model for After-funnel

- Best model: **{after['best_model']}**.
- RMSE: **{dinh_dang_metric(after['best_metrics']['rmse'])}**.
- MAE: **{dinh_dang_metric(after['best_metrics']['mae'])}**.
- R2: **{dinh_dang_metric(after['best_metrics']['r2'])}**.
- MAPE: **{dinh_dang_metric(after['best_metrics']['mape'])}%**.

![After-funnel Actual vs Predicted](../results/plots/ml/33_actual_vs_predicted_after_funnel.png)

![After-funnel Residual](../results/plots/ml/34_residual_distribution_after_funnel.png)

## 10. Interpretation

Pre-campaign model phù hợp để ước lượng Revenue ở giai đoạn lập kế hoạch vì chỉ dùng thông tin biết trước. After-funnel model thường có nhiều tín hiệu hơn vì đã biết funnel metrics, nên cần diễn giải là dự đoán late-stage.

## 11. Limitations

- Kết quả phụ thuộc vào dữ liệu lịch sử và không chứng minh quan hệ nhân quả.
- Pre-campaign model không thấy hành vi thực tế sau launch nên độ chính xác thường thấp hơn.
- After-funnel model không nên dùng cho quyết định ngân sách trước campaign.
- Nếu dataset thay đổi schema, cần kiểm tra lại feature set và leakage rules.

## 12. How to Use in Streamlit Dashboard

Train model trước:

```bash
venv/bin/python src/train_revenue_prediction.py
```

Chạy dashboard:

```bash
venv/bin/python -m streamlit run app/streamlit_app.py
```

Mở tab **Revenue Prediction ML**, chọn scenario, nhập feature và bấm Predict Revenue.
"""
    DUONG_DAN_REPORT.write_text(noi_dung, encoding="utf-8")


def tao_notebook(summary: dict[str, Any]) -> None:
    """Tạo notebook trình bày workflow Revenue Prediction."""
    DUONG_DAN_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    cac_cell: list[dict[str, Any]] = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Revenue Prediction for Marketing Campaigns\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Project Overview\n", "Notebook này trình bày pipeline Machine Learning dự đoán Revenue cho marketing campaigns.\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 2. ML Problem Definition\n", "Bài toán regression với target `Revenue`, gồm pre-campaign và after-funnel scenarios.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import json\n", "import pandas as pd\n", "from pathlib import Path\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Dataset and Target\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["du_lieu = pd.read_csv('../data/processed/marketing_campaigns_features.csv')\n", "du_lieu[['Revenue']].describe()\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Feature Sets\n", "Feature sets được quản lý trong `src/utils/revenue_prediction.py`.\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Leakage Prevention\n", "Không dùng ROI, Revenue_per_Conversion, Revenue_per_Click hoặc feature được tính trực tiếp từ Revenue.\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 6. Train/Test Split Strategy\n", "Pipeline ưu tiên temporal split theo `Date`; fallback random split nếu không có Date.\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 7. Model Comparison\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["summary = json.loads(Path('../results/metrics/revenue_prediction_summary.json').read_text(encoding='utf-8'))\n", "summary['scenarios'].keys()\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["![Model Comparison](../results/plots/ml/35_model_comparison_rmse.png)\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 8. Best Model for Pre-campaign\n", f"Best model: **{summary['scenarios']['pre_campaign']['best_model']}**\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["![Pre Actual vs Predicted](../results/plots/ml/31_actual_vs_predicted_pre_campaign.png)\n", "![Pre Residual](../results/plots/ml/32_residual_distribution_pre_campaign.png)\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 9. Best Model for After-funnel\n", f"Best model: **{summary['scenarios']['after_funnel']['best_model']}**\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["![After Actual vs Predicted](../results/plots/ml/33_actual_vs_predicted_after_funnel.png)\n", "![After Residual](../results/plots/ml/34_residual_distribution_after_funnel.png)\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 10. Interpretation\n", "So sánh pre-campaign và after-funnel để phân biệt model phục vụ planning và model late-stage.\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 11. Limitations\n", "Mô hình mô tả quan hệ trong dữ liệu lịch sử, không chứng minh nhân quả.\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 12. How to Use in Streamlit Dashboard\n", "Chạy `venv/bin/python src/train_revenue_prediction.py`, sau đó mở dashboard và vào tab Revenue Prediction ML.\n"]},
    ]
    notebook = {
        "cells": cac_cell,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DUONG_DAN_NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


def chay_train_revenue_prediction() -> None:
    """Chạy toàn bộ pipeline train Revenue Prediction."""
    tham_so = phan_tich_tham_so()
    sns.set_theme(style="whitegrid")
    du_lieu = tai_du_lieu_revenue(tham_so.duong_dan_input)

    ket_qua_scenarios = {
        "pre_campaign": train_scenario(du_lieu, "pre_campaign"),
        "after_funnel": train_scenario(du_lieu, "after_funnel"),
    }

    THU_MUC_MODEL.mkdir(parents=True, exist_ok=True)
    THU_MUC_BIEU_DO.mkdir(parents=True, exist_ok=True)
    luu_model(ket_qua_scenarios["pre_campaign"].best_pipeline, THU_MUC_MODEL / "pre_campaign_model.joblib")
    luu_model(ket_qua_scenarios["after_funnel"].best_pipeline, THU_MUC_MODEL / "after_funnel_model.joblib")

    plots = {
        "31_actual_vs_predicted_pre_campaign": str(THU_MUC_BIEU_DO / "31_actual_vs_predicted_pre_campaign.png"),
        "32_residual_distribution_pre_campaign": str(THU_MUC_BIEU_DO / "32_residual_distribution_pre_campaign.png"),
        "33_actual_vs_predicted_after_funnel": str(THU_MUC_BIEU_DO / "33_actual_vs_predicted_after_funnel.png"),
        "34_residual_distribution_after_funnel": str(THU_MUC_BIEU_DO / "34_residual_distribution_after_funnel.png"),
        "35_model_comparison_rmse": str(THU_MUC_BIEU_DO / "35_model_comparison_rmse.png"),
    }
    ve_actual_vs_predicted(
        ket_qua_scenarios["pre_campaign"],
        Path(plots["31_actual_vs_predicted_pre_campaign"]),
        "Pre-campaign: Actual vs Predicted Revenue",
    )
    ve_residual_distribution(
        ket_qua_scenarios["pre_campaign"],
        Path(plots["32_residual_distribution_pre_campaign"]),
        "Pre-campaign Residual Distribution",
    )
    ve_actual_vs_predicted(
        ket_qua_scenarios["after_funnel"],
        Path(plots["33_actual_vs_predicted_after_funnel"]),
        "After-funnel: Actual vs Predicted Revenue",
    )
    ve_residual_distribution(
        ket_qua_scenarios["after_funnel"],
        Path(plots["34_residual_distribution_after_funnel"]),
        "After-funnel Residual Distribution",
    )
    ve_model_comparison(ket_qua_scenarios, Path(plots["35_model_comparison_rmse"]))
    if ve_feature_importance(ket_qua_scenarios, THU_MUC_BIEU_DO / "36_feature_importance_best_model.png"):
        plots["36_feature_importance_best_model"] = str(THU_MUC_BIEU_DO / "36_feature_importance_best_model.png")

    summary = tao_summary(du_lieu, ket_qua_scenarios, plots)
    model_metadata = {
        "target_transform": "log1p",
        "prediction_inverse_transform": "expm1",
        "model_files": {
            "pre_campaign": "pre_campaign_model.joblib",
            "after_funnel": "after_funnel_model.joblib",
        },
        "scenarios": {
            scenario: {
                "best_model": ket_qua.best_model_name,
                "features": ket_qua.feature_columns,
                "categorical_features": ket_qua.categorical_features,
                "numeric_features": ket_qua.numeric_features,
                "late_stage_prediction": SCENARIO_CONFIG[scenario]["late_stage"],
                "test_metrics": ket_qua.metrics[ket_qua.best_model_name],
            }
            for scenario, ket_qua in ket_qua_scenarios.items()
        },
    }
    ghi_json(summary, DUONG_DAN_SUMMARY)
    ghi_json(model_metadata, THU_MUC_MODEL / "model_metadata.json")
    tao_report(summary)
    tao_notebook(summary)

    print("Hoàn tất train Revenue Prediction")
    for scenario, ket_qua in ket_qua_scenarios.items():
        metrics = ket_qua.metrics[ket_qua.best_model_name]
        print(
            f"{SCENARIO_CONFIG[scenario]['ten_hien_thi']}: "
            f"best={ket_qua.best_model_name}, "
            f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
            f"R2={metrics['r2']:.4f}, MAPE={metrics['mape']:.4f}%"
        )
    print(f"Đã lưu model: {THU_MUC_MODEL}")
    print(f"Đã lưu summary: {DUONG_DAN_SUMMARY}")
    print(f"Đã lưu report: {DUONG_DAN_REPORT}")


if __name__ == "__main__":
    chay_train_revenue_prediction()
