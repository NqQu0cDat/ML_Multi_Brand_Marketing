# Revenue Prediction for Marketing Campaigns

## 1. Project Overview

Đề tài này mở rộng project Multi-Brand Marketing Campaign Performance Analysis sang Machine Learning để dự đoán Revenue của campaign. Pipeline ML được tách riêng và không thay đổi logic EDA / Business Analytics hiện tại.

## 2. ML Problem Definition

- Bài toán: Regression.
- Target: `Revenue`.
- Mục tiêu: dự đoán doanh thu campaign và so sánh hai bối cảnh dự đoán.

## 3. Dataset and Target

- Dataset: `data/processed/marketing_campaigns_features.csv`.
- Số dòng: **166,665**.
- Số cột: **28**.
- Target được train bằng `log1p(Revenue)` và đổi ngược bằng `expm1()` khi evaluate/predict.

## 4. Feature Sets

### Pre-campaign

Feature dùng trước khi campaign chạy:

```text
Brand, Campaign_Type, Target_Audience, Duration, Channel_Used, Acquisition_Cost, Language, Customer_Segment, Month, Quarter
```

### After-funnel

Feature late-stage, chỉ dùng khi đã có funnel metrics:

```text
Brand, Campaign_Type, Target_Audience, Duration, Channel_Used, Acquisition_Cost, Language, Customer_Segment, Month, Quarter, Impressions, Clicks, Leads, Conversions, CTR, Lead_Rate, Conversion_Rate, Cost_per_Conversion
```

## 5. Leakage Prevention

Các cột bị loại khỏi cả hai scenario:

```text
Revenue, ROI, Revenue_per_Conversion, Revenue_per_Click
```

Pre-campaign model không dùng Impressions, Clicks, Leads, Conversions hoặc KPI sinh từ funnel. After-funnel model phải được hiểu là late-stage prediction, không phải dự đoán trước campaign.

## 6. Train/Test Split Strategy

- Pre-campaign: **Temporal split với ranh giới ngày nghiêm ngặt tại 2025-04-04; train_date_max=2025-04-03 và test_date_min=2025-04-04.**
- After-funnel: **Temporal split với ranh giới ngày nghiêm ngặt tại 2025-04-04; train_date_max=2025-04-03 và test_date_min=2025-04-04.**
- Pipeline không cho cùng một ngày xuất hiện ở cả train và test khi temporal split.

## 7. Model Comparison

### Pre-campaign

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| dummy | 324,976.0217 | 505,861.0643 | -0.1261 | 116.5509% |
| ridge | 260,587.5661 | 437,869.1057 | 0.1563 | 67.9065% |
| random_forest | 219,099.7392 | 346,651.7584 | 0.4712 | 54.1595% |

### After-funnel

| Model | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|
| dummy | 324,976.0217 | 505,861.0643 | -0.1261 | 116.5509% |
| ridge | 165,593.1476 | 257,800.0949 | 0.7075 | 38.6567% |
| random_forest | 155,464.9901 | 234,513.4553 | 0.7580 | 35.6013% |

![Model Comparison RMSE](../results/plots/ml/35_model_comparison_rmse.png)

## 8. Best Model for Pre-campaign

- Best model: **random_forest**.
- RMSE: **346,651.7584**.
- MAE: **219,099.7392**.
- R2: **0.4712**.
- MAPE: **54.1595%**.

![Pre-campaign Actual vs Predicted](../results/plots/ml/31_actual_vs_predicted_pre_campaign.png)

![Pre-campaign Residual](../results/plots/ml/32_residual_distribution_pre_campaign.png)

## 9. Best Model for After-funnel

- Best model: **random_forest**.
- RMSE: **234,513.4553**.
- MAE: **155,464.9901**.
- R2: **0.7580**.
- MAPE: **35.6013%**.

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
