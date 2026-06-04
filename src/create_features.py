"""Chạy bước feature engineering cho dữ liệu chiến dịch đã làm sạch."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from utils.feature_engineering import (
    luu_ket_qua_feature,
    tai_du_lieu_sach,
    tao_feature_marketing,
)


def phan_tich_tham_so() -> argparse.Namespace:
    bo_phan_tich = argparse.ArgumentParser(
        description="Tạo feature marketing từ dữ liệu đã làm sạch."
    )
    bo_phan_tich.add_argument(
        "--duong-dan-input",
        "--input-csv",
        dest="duong_dan_input",
        default="data/processed/marketing_campaigns_clean.csv",
        help="Đường dẫn file CSV đã làm sạch.",
    )
    bo_phan_tich.add_argument(
        "--duong-dan-output",
        "--output-csv",
        dest="duong_dan_output",
        default="data/processed/marketing_campaigns_features.csv",
        help="Đường dẫn lưu file CSV có feature.",
    )
    bo_phan_tich.add_argument(
        "--duong-dan-bao-cao",
        "--report-json",
        dest="duong_dan_bao_cao",
        default="results/metrics/feature_engineering_report.json",
        help="Đường dẫn lưu báo cáo feature engineering dạng JSON.",
    )
    return bo_phan_tich.parse_args()


def chay_tao_feature() -> None:
    tham_so = phan_tich_tham_so()

    du_lieu_sach = tai_du_lieu_sach(tham_so.duong_dan_input)
    du_lieu_feature, bao_cao = tao_feature_marketing(du_lieu_sach)
    luu_ket_qua_feature(
        du_lieu_feature,
        bao_cao,
        tham_so.duong_dan_output,
        tham_so.duong_dan_bao_cao,
    )

    print("Hoàn tất tạo feature")
    print(f"Số dòng: {bao_cao['rows']:,}")
    print(f"Số cột: {bao_cao['columns_before']} -> {bao_cao['columns_after']}")
    print(f"Số feature mới: {len(bao_cao['features_added'])}")
    print(f"Lỗi parse ngày: {bao_cao['date_parse_failures']:,}")
    print(f"Đã lưu dữ liệu feature: {Path(tham_so.duong_dan_output)}")
    print(f"Đã lưu báo cáo feature: {Path(tham_so.duong_dan_bao_cao)}")


if __name__ == "__main__":
    chay_tao_feature()
