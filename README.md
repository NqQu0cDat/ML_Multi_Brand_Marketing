# Multi-Brand Marketing Campaign Performance Analysis

Phân tích hiệu quả chiến dịch marketing của Nykaa, Purplle và Tira nhằm đánh giá KPI, tìm business insight và đề xuất hành động.

## Loại bài toán

Đây là project **Data Analysis / EDA / Business Analytics**. Phạm vi hiện tại chưa sử dụng Machine Learning.

## Mục tiêu

- So sánh Revenue, ROI, Acquisition Cost, CTR, CVR và CPA giữa các brand.
- Xác định channel và campaign type hiệu quả.
- Phân tích customer segment mang lại giá trị cao.
- Theo dõi xu hướng campaign theo tháng.
- Đưa ra business recommendations dựa trên dữ liệu.

## Workflow

```text
Raw data
-> Data cleaning
-> Feature engineering
-> EDA / KPI analysis
-> Business insights
-> Recommendations
-> Report / dashboard
```

## Cấu trúc chính

```text
data/raw/                                      Dữ liệu campaign gốc
data/processed/marketing_campaigns_clean.csv   Dữ liệu đã làm sạch
data/processed/marketing_campaigns_features.csv Dữ liệu có feature marketing
notebooks/01_eda.ipynb                         Notebook EDA khám phá dữ liệu
notebooks/02_business_analysis.ipynb            Notebook phân tích business
results/plots/eda/                             Biểu đồ 01-18 của EDA
results/plots/business/                        Biểu đồ 19-30 của Business Analysis
results/metrics/eda_summary.json               Summary EDA
results/metrics/business_summary.json          KPI và bảng tổng hợp business
reports/marketing_campaign_analysis_report.md  Báo cáo business cuối cùng
src/clean_data.py                              Chạy data cleaning
src/create_features.py                         Chạy feature engineering
src/run_eda.py                                 Tạo biểu đồ EDA ban đầu
src/build_business_analysis.py                 Tạo business analysis và report
```

## Chạy project

```bash
venv/bin/python src/clean_data.py
venv/bin/python src/create_features.py
venv/bin/python src/run_eda.py
venv/bin/python src/build_business_analysis.py
```

## KPI chính

- Total Revenue
- Average ROI
- Total Acquisition Cost
- Average CTR
- Average CVR
- Average CPA
- Total Campaigns
- Revenue per Click

## Deliverables

- [Notebook EDA](notebooks/01_eda.ipynb)
- [Notebook Business Analysis](notebooks/02_business_analysis.ipynb)
- [EDA Summary](results/metrics/eda_summary.json)
- [Business Summary](results/metrics/business_summary.json)
- [Marketing Campaign Analysis Report](reports/marketing_campaign_analysis_report.md)

## Streamlit Dashboard

Dashboard theo dõi KPI, hiệu quả theo brand/channel/campaign type/segment, xu hướng tháng, EDA plots và business report.

Cài dependencies và chạy app:

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m streamlit run app/streamlit_app.py
```

App tự tìm CSV trong `data/processed/`, ưu tiên file có chữ `features` trong tên.


## Machine Learning / Revenue Prediction

Đề tài 2 mở rộng project sang bài toán **Revenue Prediction for Marketing Campaigns**. Phần này chạy sau EDA / Business Analytics và không thay đổi logic phân tích cũ.

Mục tiêu:

- Dự đoán `Revenue` của marketing campaign.
- So sánh hai scenario model:
  - **Pre-campaign model**: dùng thông tin biết trước khi campaign chạy, phục vụ planning.
  - **After-funnel model**: dùng thêm funnel metrics, là late-stage prediction để so sánh hiệu suất mô hình.

Train model:

```bash
venv/bin/python src/train_revenue_prediction.py
```

Output chính:

```text
models/revenue_prediction/pre_campaign_model.joblib
models/revenue_prediction/after_funnel_model.joblib
models/revenue_prediction/model_metadata.json
results/metrics/revenue_prediction_summary.json
results/plots/ml/
notebooks/03_revenue_prediction.ipynb
reports/revenue_prediction_report.md
```

Chạy dashboard sau khi train:

```bash
venv/bin/python -m streamlit run app/streamlit_app.py
```

Mở tab **Revenue Prediction ML** để chọn scenario, nhập feature và dự đoán Revenue. Streamlit chỉ load model đã train sẵn, không train model trong app.
