# AGENTS.md

## Mục đích

File này dùng để quy định cách AI hỗ trợ code trong project Machine Learning này.

Project: Multi-Brand Marketing_ML
Mục tiêu: phân tích và dự đoán hiệu quả chiến dịch marketing của nhiều brand.

---

## Quy tắc viết code

### Ưu tiên cao nhất

- Tên biến, tên hàm, tên class: dùng tiếng Việt không dấu.
- Comment: dùng tiếng Việt có dấu.
- Ưu tiên `snake_case` cho tên biến và tên hàm.
- Tên cột dữ liệu gốc được giữ nguyên theo CSV nếu cần để tránh sai schema.
- Tên file có thể giữ tiếng Anh nếu đã có trong project, nhưng code mới bên trong file phải theo quy tắc trên.

Ví dụ đúng:

```python
doanh_thu = 1000000
chi_phi_quang_cao = 250000
diem_tuong_tac = 85

def tinh_roi(doanh_thu, chi_phi_quang_cao):
    # Tính ROI dựa trên doanh thu và chi phí quảng cáo
    return (doanh_thu - chi_phi_quang_cao) / chi_phi_quang_cao
```

Ví dụ không ưu tiên:

```python
revenue = 1000000
ad_cost = 250000

def calculate_roi(revenue, ad_cost):
    return (revenue - ad_cost) / ad_cost
```

---

## Quy tắc làm việc với dữ liệu

- Không train model trực tiếp từ `data/raw`.
- Luôn tạo dữ liệu sạch trong `data/processed` trước khi EDA hoặc modeling.
- Không xóa dữ liệu bất thường nếu chưa có lý do rõ ràng.
- Các dòng bị loại phải có báo cáo số lượng và lý do.
- `ROI` âm được xem là campaign lỗ, không tự động coi là lỗi dữ liệu.
- Các chỉ số funnel cần được kiểm tra:
  - `Clicks <= Impressions`
  - `Leads <= Clicks`
  - `Conversions <= Leads`

---

## Workflow chuẩn

Thứ tự làm việc mặc định:

```text
Raw data
-> Data cleaning
-> Feature engineering
-> EDA
-> Modeling
-> Evaluation
-> Report / dashboard
```

Không nhảy sang modeling nếu chưa hoàn tất cleaning và feature engineering.

---

## Agent: DataCleaner

### Mục đích

Làm sạch dữ liệu campaign từ nhiều brand.

### Input

```text
data/raw/*.csv
```

### Output

```text
data/processed/marketing_campaigns_clean.csv
results/metrics/data_cleaning_report.json
```

### Nhiệm vụ

- Gộp dữ liệu từ nhiều brand.
- Thêm cột `Brand`.
- Chuẩn hóa kiểu dữ liệu.
- Parse cột `Date`.
- Kiểm tra missing values.
- Kiểm tra duplicate rows và duplicate `Campaign_ID`.
- Chuẩn hóa `Channel_Used` để các tổ hợp kênh khác thứ tự được gom cùng nhóm.
- Kiểm tra logic funnel.
- Xuất báo cáo cleaning.

---

## Agent: FeatureEngineer

### Mục đích

Tạo feature marketing từ dữ liệu sạch.

### Input

```text
data/processed/marketing_campaigns_clean.csv
```

### Output

```text
data/processed/marketing_campaigns_features.csv
results/metrics/feature_engineering_report.json
```

### Feature cần tạo

```text
CTR = Clicks / Impressions
Lead_Rate = Leads / Clicks
Conversion_Rate = Conversions / Leads
Revenue_per_Conversion = Revenue / Conversions
Cost_per_Conversion = Acquisition_Cost / Conversions
Revenue_per_Click = Revenue / Clicks
Year
Month
Day
DayOfWeek
Quarter
```

---

## Agent: EDAAnalyst

### Mục đích

Phân tích dữ liệu để hiểu hiệu quả chiến dịch theo brand, channel, audience và campaign type.

### Input

```text
data/processed/marketing_campaigns_features.csv
```

### Output

```text
notebooks/01_eda.ipynb
results/plots/
results/metrics/eda_summary.json
```

### Nhiệm vụ

- So sánh hiệu quả giữa các brand.
- Phân tích phân phối `Revenue`, `ROI`, `Conversions`, `Engagement_Score`.
- Phân tích channel, audience, language, campaign type.
- Tìm outlier đáng chú ý.
- Rút insight phục vụ modeling.

---

## Agent: ModelTrainer

### Mục đích

Train và đánh giá model dự đoán hiệu quả campaign.

### Input

```text
data/processed/marketing_campaigns_features.csv
config/config.yaml
```

### Output

```text
models/
results/metrics/model_metrics.json
results/predictions/
```

### Nhiệm vụ

- Xác định target phù hợp, mặc định ưu tiên `ROI`.
- Tách train/test.
- Train baseline model.
- So sánh Linear Regression, Random Forest, XGBoost nếu phù hợp.
- Lưu model tốt nhất.
- Lưu metrics và prediction.

---

## Quy tắc báo cáo sau mỗi bước

Sau mỗi bước chính, AI cần báo cáo ngắn gọn:

```text
Agent:
Input:
Output:
Việc đã làm:
Kết quả chính:
File đã tạo/cập nhật:
Bước tiếp theo:
```

Nếu có dữ liệu bị loại hoặc giả định quan trọng, phải nói rõ.
