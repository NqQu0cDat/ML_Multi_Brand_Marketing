# Multi-Brand Marketing ML Project

Dự án Machine Learning để phân tích và dự đoán hiệu suất chiến dịch marketing của các thương hiệu: Nykaa, Purplle, và Tira.

## Cấu trúc Project

```
├── data/
│   ├── raw/                          # Dữ liệu thô từ nguồn
│   │   ├── nykaa_campaign_data.csv
│   │   ├── purplle_campaign_data.csv
│   │   └── tira_campaign_data.csv
│   └── processed/                    # Dữ liệu đã xử lý
├── notebooks/                        # Jupyter notebooks
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       # Xử lý dữ liệu
│   └── 03_modeling.ipynb            # Xây dựng mô hình
├── src/                              # Source code
│   ├── models/                       # Model training
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       └── visualization.py
├── models/                           # Saved trained models
├── results/                          # Output results
│   ├── plots/
│   ├── metrics/
│   └── predictions/
├── config/                           # Configuration files
│   └── config.yaml
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
└── README.md                         # Project documentation
```

## Chuẩn bị

### 1. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu trúc dữ liệu
- Đặt CSV files vào `data/raw/`
- Dữ liệu được xử lý sẽ lưu vào `data/processed/`

### 3. Workflow
1. **EDA (Exploratory Data Analysis)**: Phân tích dữ liệu ban đầu
2. **Preprocessing**: Xử lý missing values, feature engineering
3. **Modeling**: Xây dựng và huấn luyện mô hình
4. **Evaluation**: Đánh giá hiệu suất
5. **Prediction**: Dự đoán trên dữ liệu mới

## Công nghệ sử dụng
- Python 3.x
- Pandas: Xử lý dữ liệu
- NumPy: Tính toán
- Scikit-learn: Machine Learning models
- Matplotlib/Seaborn: Visualization
- Jupyter: Interactive notebooks

## Bắt đầu

Xem `notebooks/01_EDA.ipynb` để bắt đầu phân tích dữ liệu.
