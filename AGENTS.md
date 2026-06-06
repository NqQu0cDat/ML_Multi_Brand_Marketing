# AGENTS.md

## Mục đích

File này quy định cách AI hỗ trợ code trong project:

**Multi-Brand Marketing Campaign Performance Analysis**

Project tập trung vào Data Analysis / EDA / Business Analytics. Chưa triển khai Machine Learning nếu người dùng chưa yêu cầu thay đổi phạm vi.

## Quy tắc viết code

- Tên biến, tên hàm, tên class: dùng tiếng Việt không dấu.
- Comment: dùng tiếng Việt có dấu.
- Ưu tiên `snake_case` cho tên biến và tên hàm.
- Giữ nguyên tên cột dữ liệu gốc để tránh sai schema.
- Không xóa dữ liệu bất thường nếu chưa có lý do rõ ràng.
- Các dòng bị loại phải có báo cáo số lượng và lý do.

## Workflow chuẩn

```text
Raw data
-> Data cleaning
-> Feature engineering
-> EDA / KPI analysis
-> Business insights
-> Recommendations
-> Report / dashboard
```

## Agent: DataCleaner

Input:
```text
data/raw/*.csv
```

Output:
```text
data/processed/marketing_campaigns_clean.csv
results/metrics/data_cleaning_report.json
```

Nhiệm vụ:
- Gộp dữ liệu và thêm `Brand`.
- Chuẩn hóa kiểu dữ liệu, ngày và channel.
- Kiểm tra missing, duplicate và logic funnel.
- Giữ ROI âm vì đây có thể là campaign lỗ.

## Agent: FeatureEngineer

Input:
```text
data/processed/marketing_campaigns_clean.csv
```

Output:
```text
data/processed/marketing_campaigns_features.csv
results/metrics/feature_engineering_report.json
```

Feature chính:
```text
CTR
Lead_Rate
Conversion_Rate
Revenue_per_Conversion
Cost_per_Conversion
Revenue_per_Click
Year
Month
Day
DayOfWeek
Quarter
```

## Agent: EDAAnalyst

Input:
```text
data/processed/marketing_campaigns_features.csv
```

Output:
```text
notebooks/01_eda.ipynb
results/plots/eda/
results/metrics/eda_summary.json
```

## Agent: BusinessAnalyst

Input:
```text
data/processed/marketing_campaigns_features.csv
results/metrics/eda_summary.json
```

Output:
```text
notebooks/02_business_analysis.ipynb
results/plots/business/
results/metrics/business_summary.json
reports/marketing_campaign_analysis_report.md
```

Nhiệm vụ:
- Tổng hợp KPI từ kết quả EDA.
- Phân tích brand, channel, campaign type, customer segment và thời gian theo góc nhìn business.
- Tổng hợp KPI, insight, caveat và business recommendations.
- Không diễn giải tương quan như quan hệ nhân quả.
- Không so sánh trực tiếp tháng chưa hoàn chỉnh với tháng đầy đủ.
- Không ghi đè `notebooks/01_eda.ipynb` hoặc `results/metrics/eda_summary.json`.

## Quy tắc báo cáo sau mỗi bước

```text
Agent:
Input:
Output:
Việc đã làm:
Kết quả chính:
File đã tạo/cập nhật:
Bước tiếp theo:
```
