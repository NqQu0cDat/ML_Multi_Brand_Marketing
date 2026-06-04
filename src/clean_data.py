"""Chạy bước làm sạch dữ liệu cho các dataset chiến dịch marketing."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from utils.data_cleaning import (
    lam_sach_du_lieu_chien_dich,
    luu_ket_qua_lam_sach,
    tai_du_lieu_chien_dich_tho,
)


def phan_tich_tham_so() -> argparse.Namespace:
    bo_phan_tich = argparse.ArgumentParser(
        description="Làm sạch dữ liệu chiến dịch marketing thô."
    )
    bo_phan_tich.add_argument(
        "--thu-muc-raw",
        "--raw-dir",
        dest="thu_muc_raw",
        default="data/raw",
        help="Thư mục chứa các file CSV thô.",
    )
    bo_phan_tich.add_argument(
        "--duong-dan-csv",
        "--output-csv",
        dest="duong_dan_csv",
        default="data/processed/marketing_campaigns_clean.csv",
        help="Đường dẫn lưu file CSV đã làm sạch.",
    )
    bo_phan_tich.add_argument(
        "--duong-dan-bao-cao",
        "--report-json",
        dest="duong_dan_bao_cao",
        default="results/metrics/data_cleaning_report.json",
        help="Đường dẫn lưu báo cáo làm sạch dạng JSON.",
    )
    return bo_phan_tich.parse_args()


def chay_lam_sach_du_lieu() -> None:
    tham_so = phan_tich_tham_so()

    du_lieu_tho = tai_du_lieu_chien_dich_tho(tham_so.thu_muc_raw)
    du_lieu_sach, bao_cao = lam_sach_du_lieu_chien_dich(du_lieu_tho)
    luu_ket_qua_lam_sach(
        du_lieu_sach,
        bao_cao,
        tham_so.duong_dan_csv,
        tham_so.duong_dan_bao_cao,
    )

    print("Hoàn tất làm sạch dữ liệu")
    print(f"Số dòng thô: {bao_cao['rows_before']:,}")
    print(f"Số dòng sạch: {bao_cao['rows_after']:,}")
    print(f"Số dòng đã loại: {bao_cao['rows_removed']:,}")
    print(
        "Số nhóm kênh: "
        f"{bao_cao['raw_channel_categories']} -> "
        f"{bao_cao['clean_channel_categories']}"
    )
    print(f"Đã lưu dữ liệu sạch: {Path(tham_so.duong_dan_csv)}")
    print(f"Đã lưu báo cáo làm sạch: {Path(tham_so.duong_dan_bao_cao)}")


if __name__ == "__main__":
    chay_lam_sach_du_lieu()
