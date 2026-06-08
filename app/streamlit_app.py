"""Dashboard Streamlit theo dõi hiệu quả chiến dịch marketing đa thương hiệu."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.express as px
import streamlit as st


THU_MUC_GOC = Path(__file__).resolve().parents[1]
THU_MUC_DU_LIEU = THU_MUC_GOC / "data" / "processed"
THU_MUC_BIEU_DO = THU_MUC_GOC / "results" / "plots"
DUONG_DAN_TONG_HOP = THU_MUC_GOC / "results" / "metrics" / "eda_summary.json"
DUONG_DAN_BAO_CAO = (
    THU_MUC_GOC / "reports" / "marketing_campaign_analysis_report.md"
)
THU_MUC_MODEL_ML = THU_MUC_GOC / "models" / "revenue_prediction"
DUONG_DAN_METADATA_ML = THU_MUC_MODEL_ML / "model_metadata.json"
DUONG_DAN_SUMMARY_ML = (
    THU_MUC_GOC / "results" / "metrics" / "revenue_prediction_summary.json"
)

CAC_KPI = {
    "Tổng doanh thu": ("Revenue", "sum", "money"),
    "ROI trung bình": ("ROI", "mean", "decimal"),
    "Tổng chi phí thu hút khách hàng": ("Acquisition_Cost", "sum", "money"),
    "CTR trung bình": ("CTR", "mean", "percent"),
    "CVR trung bình": ("Conversion_Rate", "mean", "percent"),
    "CPA trung bình": ("Cost_per_Conversion", "mean", "decimal"),
    "Tổng số chiến dịch": ("Campaign_ID", "count", "integer"),
    "Doanh thu mỗi lượt nhấp": ("Revenue_per_Click", "mean", "decimal"),
}

TEN_SCENARIO_ML = {
    "pre_campaign": "Pre-campaign",
    "after_funnel": "After-funnel",
}

MO_TA_SCENARIO_ML = {
    "pre_campaign": "Dự đoán Revenue trước khi campaign chạy, phục vụ lập kế hoạch ngân sách.",
    "after_funnel": "Late-stage prediction khi đã có funnel metrics, không dùng cho planning trước campaign.",
}


def tim_file_du_lieu() -> Path | None:
    """Tìm CSV đã xử lý, ưu tiên file có chữ features trong tên."""
    cac_tep = sorted(THU_MUC_DU_LIEU.glob("*.csv"))
    if not cac_tep:
        return None
    cac_tep_feature = [tep for tep in cac_tep if "features" in tep.name.lower()]
    return cac_tep_feature[0] if cac_tep_feature else cac_tep[0]


@st.cache_data(show_spinner=False)
def tai_du_lieu(duong_dan: str) -> pd.DataFrame:
    """Tải dữ liệu và bổ sung các feature cơ bản nếu còn thiếu."""
    du_lieu = pd.read_csv(duong_dan)
    if "Date" in du_lieu.columns:
        du_lieu["Date"] = pd.to_datetime(du_lieu["Date"], errors="coerce")

    cac_phep_chia = {
        "CTR": ("Clicks", "Impressions"),
        "Conversion_Rate": ("Conversions", "Leads"),
        "Cost_per_Conversion": ("Acquisition_Cost", "Conversions"),
        "Revenue_per_Click": ("Revenue", "Clicks"),
    }
    for cot_moi, (tu_so, mau_so) in cac_phep_chia.items():
        if cot_moi not in du_lieu.columns and {tu_so, mau_so}.issubset(du_lieu.columns):
            du_lieu[cot_moi] = du_lieu[tu_so].div(
                du_lieu[mau_so].where(du_lieu[mau_so] != 0)
            ).fillna(0)
    if "Conversion_Rate" not in du_lieu.columns and "CVR" in du_lieu.columns:
        du_lieu["Conversion_Rate"] = du_lieu["CVR"]
    if "Cost_per_Conversion" not in du_lieu.columns and "CPA" in du_lieu.columns:
        du_lieu["Cost_per_Conversion"] = du_lieu["CPA"]
    return du_lieu


@st.cache_data(show_spinner=False)
def tai_tong_hop() -> dict:
    """Đọc EDA summary nếu file tồn tại và hợp lệ."""
    if not DUONG_DAN_TONG_HOP.exists():
        return {}
    try:
        return json.loads(DUONG_DAN_TONG_HOP.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data(show_spinner=False)
def tai_json_an_toan(duong_dan: str) -> dict[str, Any]:
    """Đọc JSON an toàn cho dashboard."""
    tep = Path(duong_dan)
    if not tep.exists():
        return {}
    try:
        return json.loads(tep.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_resource(show_spinner=False)
def tai_model_ml(duong_dan: str) -> Any:
    """Tải model ML đã train."""
    return joblib.load(duong_dan)


def dinh_dang_so(gia_tri: float, kieu: str) -> str:
    """Định dạng KPI để dễ đọc."""
    if pd.isna(gia_tri):
        return "N/A"
    if kieu == "money":
        return f"{gia_tri:,.0f}"
    if kieu == "percent":
        return f"{gia_tri:.2%}"
    if kieu == "integer":
        return f"{gia_tri:,.0f}"
    return f"{gia_tri:,.3f}"


def dinh_dang_tien(gia_tri: float) -> str:
    """Định dạng tiền cho kết quả dự đoán."""
    if pd.isna(gia_tri):
        return "N/A"
    return f"{gia_tri:,.0f}"


def tinh_kpi(du_lieu: pd.DataFrame) -> dict[str, str]:
    """Tính các KPI chính trên dữ liệu sau filter."""
    ket_qua = {}
    for ten_kpi, (cot, phep_tinh, kieu) in CAC_KPI.items():
        if cot not in du_lieu.columns:
            ket_qua[ten_kpi] = "N/A"
            continue
        gia_tri = getattr(du_lieu[cot], phep_tinh)()
        ket_qua[ten_kpi] = dinh_dang_so(float(gia_tri), kieu)
    return ket_qua


def hien_thi_kpi(du_lieu: pd.DataFrame) -> None:
    """Hiển thị tám KPI trong hai hàng."""
    cac_gia_tri = tinh_kpi(du_lieu)
    for nhom_ten in [list(cac_gia_tri)[:4], list(cac_gia_tri)[4:]]:
        cac_cot = st.columns(4)
        for cot, ten_kpi in zip(cac_cot, nhom_ten):
            cot.metric(ten_kpi, cac_gia_tri[ten_kpi])


def tao_bo_loc(du_lieu: pd.DataFrame) -> pd.DataFrame:
    """Tạo sidebar filter và trả về dữ liệu đã lọc."""
    st.sidebar.header("Bộ lọc")
    du_lieu_loc = du_lieu.copy()
    for cot, nhan in [
        ("Brand", "Thương hiệu"),
        ("Channel_Used", "Kênh marketing"),
        ("Campaign_Type", "Loại chiến dịch"),
        ("Customer_Segment", "Phân khúc khách hàng"),
    ]:
        if cot not in du_lieu_loc.columns:
            st.sidebar.caption(f"Không có cột `{cot}`")
            continue
        cac_gia_tri = sorted(du_lieu_loc[cot].dropna().astype(str).unique())
        lua_chon = st.sidebar.multiselect(nhan, cac_gia_tri, placeholder="Tất cả")
        if lua_chon:
            du_lieu_loc = du_lieu_loc[du_lieu_loc[cot].astype(str).isin(lua_chon)]
    st.sidebar.divider()
    st.sidebar.metric("Số chiến dịch sau lọc", f"{len(du_lieu_loc):,}")
    return du_lieu_loc


def tao_bang_tong_hop(du_lieu: pd.DataFrame, cot_nhom: str) -> pd.DataFrame:
    """Tổng hợp KPI theo một chiều phân tích."""
    if cot_nhom not in du_lieu.columns:
        return pd.DataFrame()
    cac_phep_tinh = {
        "Tổng doanh thu": ("Revenue", "sum"),
        "ROI trung bình": ("ROI", "mean"),
        "Tổng chi phí thu hút": ("Acquisition_Cost", "sum"),
        "CTR trung bình": ("CTR", "mean"),
        "CVR trung bình": ("Conversion_Rate", "mean"),
        "CPA trung bình": ("Cost_per_Conversion", "mean"),
        "Số chiến dịch": ("Campaign_ID", "count"),
    }
    cac_phep_hop_le = {
        ten: phep for ten, phep in cac_phep_tinh.items() if phep[0] in du_lieu.columns
    }
    if not cac_phep_hop_le:
        return pd.DataFrame()
    return (
        du_lieu.groupby(cot_nhom, as_index=False)
        .agg(**cac_phep_hop_le)
        .sort_values("Tổng doanh thu" if "Tổng doanh thu" in cac_phep_hop_le else cot_nhom, ascending=False)
    )


def hien_thi_tab_nhom(
    du_lieu: pd.DataFrame,
    cot_nhom: str,
    tieu_de: str,
) -> None:
    """Hiển thị bảng và các biểu đồ KPI theo nhóm."""
    st.subheader(tieu_de)
    bang = tao_bang_tong_hop(du_lieu, cot_nhom)
    if bang.empty:
        st.warning(f"Không đủ cột để phân tích theo `{cot_nhom}`.")
        return

    st.dataframe(bang, width="stretch", hide_index=True)
    cot_trai, cot_phai = st.columns(2)
    if "Tổng doanh thu" in bang.columns:
        cot_trai.plotly_chart(
            px.bar(bang, x=cot_nhom, y="Tổng doanh thu", title=f"Tổng doanh thu theo {tieu_de}"),
            width="stretch",
        )
    if "ROI trung bình" in bang.columns:
        cot_phai.plotly_chart(
            px.bar(
                bang.sort_values("ROI trung bình", ascending=False),
                x=cot_nhom,
                y="ROI trung bình",
                title=f"ROI trung bình theo {tieu_de}",
            ),
            width="stretch",
        )
    cot_trai, cot_phai = st.columns(2)
    if "CTR trung bình" in bang.columns and "CVR trung bình" in bang.columns:
        bang_ty_le = bang.melt(
            id_vars=cot_nhom,
            value_vars=["CTR trung bình", "CVR trung bình"],
            var_name="KPI",
            value_name="Giá trị",
        )
        cot_trai.plotly_chart(
            px.bar(
                bang_ty_le,
                x=cot_nhom,
                y="Giá trị",
                color="KPI",
                barmode="group",
                title=f"CTR và CVR theo {tieu_de}",
            ),
            width="stretch",
        )
    if "CPA trung bình" in bang.columns:
        cot_phai.plotly_chart(
            px.bar(
                bang.sort_values("CPA trung bình"),
                x=cot_nhom,
                y="CPA trung bình",
                title=f"CPA trung bình theo {tieu_de}",
            ),
            width="stretch",
        )


def hien_thi_xu_huong(du_lieu: pd.DataFrame) -> None:
    """Hiển thị xu hướng Revenue, ROI, chi phí và campaign theo tháng."""
    st.subheader("Xu hướng theo tháng")
    cac_cot_can_co = ["Date", "Revenue", "ROI", "Acquisition_Cost", "Campaign_ID"]
    cac_cot_thieu = kiem_tra_cot(du_lieu, cac_cot_can_co)
    if cac_cot_thieu:
        st.warning(f"Không đủ cột để phân tích xu hướng: {', '.join(cac_cot_thieu)}")
        return
    if du_lieu["Date"].isna().all():
        st.warning("Không có giá trị Date hợp lệ để phân tích xu hướng.")
        return
    du_lieu_thang = du_lieu.assign(
        Thang=du_lieu["Date"].dt.to_period("M").astype(str)
    )
    bang = (
        du_lieu_thang.groupby("Thang", as_index=False)
        .agg(
            Tong_doanh_thu=("Revenue", "sum"),
            ROI_trung_binh=("ROI", "mean"),
            Tong_chi_phi_thu_hut=("Acquisition_Cost", "sum"),
            So_chien_dich=("Campaign_ID", "count"),
        )
        .sort_values("Thang")
    )
    st.dataframe(bang, width="stretch", hide_index=True)
    cot_trai, cot_phai = st.columns(2)
    cot_trai.plotly_chart(
        px.line(bang, x="Thang", y="Tong_doanh_thu", markers=True, title="Xu hướng doanh thu"),
        width="stretch",
    )
    cot_phai.plotly_chart(
        px.line(
            bang, x="Thang", y="ROI_trung_binh", markers=True, title="Xu hướng ROI trung bình"
        ),
        width="stretch",
    )
    cot_trai, cot_phai = st.columns(2)
    cot_trai.plotly_chart(
        px.line(
            bang,
            x="Thang",
            y="Tong_chi_phi_thu_hut",
            markers=True,
            title="Xu hướng chi phí thu hút khách hàng",
        ),
        width="stretch",
    )
    cot_phai.plotly_chart(
        px.bar(bang, x="Thang", y="So_chien_dich", title="Số chiến dịch theo tháng"),
        width="stretch",
    )


def tim_anh() -> list[Path]:
    """Tìm toàn bộ PNG trong results/plots và các thư mục con."""
    return sorted(THU_MUC_BIEU_DO.rglob("*.png")) if THU_MUC_BIEU_DO.exists() else []


def hien_thi_anh_eda() -> None:
    """Cho phép chọn và xem từng biểu đồ PNG."""
    st.subheader("Thư viện biểu đồ EDA")
    cac_anh = tim_anh()
    if not cac_anh:
        st.warning("Không tìm thấy file PNG trong `results/plots/`.")
        return
    nhan_anh = {
        str(anh.relative_to(THU_MUC_GOC)): anh
        for anh in cac_anh
    }
    lua_chon = st.selectbox("Chọn biểu đồ", list(nhan_anh))
    st.image(str(nhan_anh[lua_chon]), caption=lua_chon, width="stretch")


def hien_thi_bao_cao() -> None:
    """Đọc và hiển thị báo cáo business dạng Markdown."""
    st.subheader("Báo cáo phân tích kinh doanh")
    if not DUONG_DAN_BAO_CAO.exists():
        st.warning("Không tìm thấy `reports/marketing_campaign_analysis_report.md`.")
        return
    noi_dung = DUONG_DAN_BAO_CAO.read_text(encoding="utf-8")
    # Ảnh local được xem trong tab EDA Plots; bỏ link ảnh để report không hiện link hỏng.
    noi_dung = re.sub(r"!\[[^]]*\]\([^)]*\)\n?", "", noi_dung)
    st.markdown(noi_dung)


def lay_gia_tri_mac_dinh(du_lieu: pd.DataFrame, cot: str) -> float:
    """Lấy median làm giá trị mặc định cho input số."""
    if cot not in du_lieu.columns:
        return 0.0
    gia_tri = pd.to_numeric(du_lieu[cot], errors="coerce").median()
    if pd.isna(gia_tri):
        return 0.0
    return float(gia_tri)


def lay_options_categorical(du_lieu: pd.DataFrame, cot: str) -> list[str]:
    """Lấy options categorical từ dataset."""
    if cot not in du_lieu.columns:
        return [""]
    options = sorted(du_lieu[cot].dropna().astype(str).unique().tolist())
    return options if options else [""]


def tao_form_du_doan_ml(
    du_lieu: pd.DataFrame,
    feature_columns: list[str],
    scenario: str,
) -> pd.DataFrame:
    """Tạo form nhập feature và trả về một dòng dữ liệu predict."""
    numeric_features = [
        cot for cot in feature_columns if cot in du_lieu.columns and is_numeric_dtype(du_lieu[cot])
    ]
    categorical_features = [cot for cot in feature_columns if cot not in numeric_features]
    du_lieu_input: dict[str, Any] = {}

    st.caption(MO_TA_SCENARIO_ML.get(scenario, ""))
    st.markdown("**Thông tin phân loại**")
    cac_cot_cat = st.columns(2)
    for index, cot in enumerate(categorical_features):
        options = lay_options_categorical(du_lieu, cot)
        with cac_cot_cat[index % 2]:
            du_lieu_input[cot] = st.selectbox(cot, options, key=f"ml_{scenario}_{cot}")

    st.markdown("**Thông tin số**")
    cac_cot_num = st.columns(3)
    for index, cot in enumerate(numeric_features):
        mac_dinh = lay_gia_tri_mac_dinh(du_lieu, cot)
        buoc = 1.0 if cot in {"Duration", "Month", "Quarter", "Impressions", "Clicks", "Leads", "Conversions"} else 0.01
        with cac_cot_num[index % 3]:
            du_lieu_input[cot] = st.number_input(
                cot,
                min_value=0.0,
                value=float(mac_dinh),
                step=buoc,
                key=f"ml_{scenario}_{cot}",
            )

    return pd.DataFrame([du_lieu_input], columns=feature_columns)


def hien_thi_metrics_ml(metrics: dict[str, Any]) -> None:
    """Hiển thị metrics test của best model."""
    cac_cot = st.columns(4)
    cac_cot[0].metric("MAE", dinh_dang_tien(float(metrics.get("mae", np.nan))))
    cac_cot[1].metric("RMSE", dinh_dang_tien(float(metrics.get("rmse", np.nan))))
    cac_cot[2].metric("R2", dinh_dang_so(float(metrics.get("r2", np.nan)), "decimal"))
    cac_cot[3].metric("MAPE", f"{float(metrics.get('mape', np.nan)):.2f}%")


def hien_thi_revenue_prediction_ml(du_lieu: pd.DataFrame) -> None:
    """Hiển thị tab Revenue Prediction ML."""
    st.subheader("Revenue Prediction ML")
    metadata = tai_json_an_toan(str(DUONG_DAN_METADATA_ML))
    summary = tai_json_an_toan(str(DUONG_DAN_SUMMARY_ML))

    if not metadata or not summary:
        st.warning("Chưa tìm thấy model metadata hoặc revenue_prediction_summary.json.")
        st.code("venv/bin/python src/train_revenue_prediction.py", language="bash")
        return

    scenario_label = st.radio(
        "Chọn scenario",
        ["Pre-campaign", "After-funnel"],
        horizontal=True,
    )
    scenario = "pre_campaign" if scenario_label == "Pre-campaign" else "after_funnel"
    scenario_meta = metadata.get("scenarios", {}).get(scenario, {})
    feature_columns = scenario_meta.get("features", [])
    model_file = metadata.get("model_files", {}).get(scenario, "")
    duong_dan_model = THU_MUC_MODEL_ML / model_file

    if not feature_columns or not duong_dan_model.exists():
        st.warning("Thiếu model hoặc danh sách feature cho scenario đã chọn.")
        st.code("venv/bin/python src/train_revenue_prediction.py", language="bash")
        return

    cac_cot_thieu = kiem_tra_cot(du_lieu, feature_columns)
    if cac_cot_thieu:
        st.warning(f"Dataset thiếu cột để tạo form: {', '.join(cac_cot_thieu)}")
        return

    try:
        model = tai_model_ml(str(duong_dan_model))
    except Exception as loi:
        st.error(f"Không thể load model: {loi}")
        st.code("venv/bin/python src/train_revenue_prediction.py", language="bash")
        return

    du_lieu_input = tao_form_du_doan_ml(du_lieu, feature_columns, scenario)
    if st.button("Predict Revenue", type="primary"):
        try:
            du_doan_log = model.predict(du_lieu_input)
            du_doan = float(np.maximum(np.expm1(du_doan_log), 0)[0])
        except Exception as loi:
            st.error(f"Không thể dự đoán Revenue: {loi}")
            return

        st.metric("Predicted Revenue", dinh_dang_tien(du_doan))
        st.caption(f"Scenario: {TEN_SCENARIO_ML[scenario]} | Best model: {scenario_meta.get('best_model', 'N/A')}")
        metrics = scenario_meta.get("test_metrics", {})
        if metrics:
            hien_thi_metrics_ml(metrics)

    scenario_summary = summary.get("scenarios", {}).get(scenario, {})
    if scenario_summary:
        st.markdown("**Model comparison trên test set**")
        bang_metrics = pd.DataFrame(scenario_summary.get("metrics", {})).T.reset_index()
        if not bang_metrics.empty:
            bang_metrics = bang_metrics.rename(columns={"index": "Model"})
            st.dataframe(bang_metrics, width="stretch", hide_index=True)


def kiem_tra_cot(du_lieu: pd.DataFrame, cac_cot: Iterable[str]) -> list[str]:
    """Trả về các cột bị thiếu."""
    return [cot for cot in cac_cot if cot not in du_lieu.columns]


def chay_dashboard() -> None:
    """Khởi chạy dashboard EDA, Business Analytics và Revenue Prediction."""
    st.set_page_config(
        page_title="Dashboard Marketing Đa Thương Hiệu",
        layout="wide",
    )
    st.title("Phân Tích Hiệu Quả Chiến Dịch Marketing Đa Thương Hiệu")
    st.caption("Dashboard theo dõi kết quả EDA, phân tích kinh doanh và Revenue Prediction ML.")

    duong_dan_du_lieu = tim_file_du_lieu()
    if duong_dan_du_lieu is None:
        st.error("Không tìm thấy CSV trong `data/processed/`.")
        st.stop()
    try:
        du_lieu = tai_du_lieu(str(duong_dan_du_lieu))
    except Exception as loi:
        st.error(f"Không thể đọc dữ liệu: {loi}")
        st.stop()

    cac_cot_thieu = kiem_tra_cot(
        du_lieu,
        ["Revenue", "ROI", "Acquisition_Cost", "Campaign_ID"],
    )
    if cac_cot_thieu:
        st.warning(f"Thiếu cột KPI quan trọng: {', '.join(cac_cot_thieu)}")

    du_lieu_loc = tao_bo_loc(du_lieu)
    if du_lieu_loc.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")

    st.sidebar.caption(f"Nguồn dữ liệu: {duong_dan_du_lieu.name}")
    tong_hop = tai_tong_hop()
    if tong_hop:
        st.sidebar.success("Đã tải eda_summary.json")
    else:
        st.sidebar.info("Không có EDA summary; KPI được tính trực tiếp từ CSV.")

    cac_tab = st.tabs(
        [
            "Tổng quan",
            "Hiệu quả thương hiệu",
            "Hiệu quả kênh marketing",
            "Loại chiến dịch",
            "Phân khúc khách hàng",
            "Xu hướng theo tháng",
            "Biểu đồ EDA",
            "Báo cáo kinh doanh",
            "Revenue Prediction ML",
        ]
    )
    with cac_tab[0]:
        st.subheader("Tổng quan KPI")
        hien_thi_kpi(du_lieu_loc)
        st.caption(
            f"Đang hiển thị {len(du_lieu_loc):,} trên {len(du_lieu):,} chiến dịch từ "
            f"`{duong_dan_du_lieu.name}`."
        )
        if not du_lieu_loc.empty and {"Revenue", "ROI", "Brand"}.issubset(du_lieu_loc.columns):
            cot_trai, cot_phai = st.columns(2)
            cot_trai.plotly_chart(
                px.histogram(du_lieu_loc, x="Revenue", nbins=50, title="Phân phối doanh thu"),
                width="stretch",
            )
            cot_phai.plotly_chart(
                px.box(du_lieu_loc, x="Brand", y="ROI", title="Phân phối ROI theo thương hiệu"),
                width="stretch",
            )
    with cac_tab[1]:
        hien_thi_tab_nhom(du_lieu_loc, "Brand", "thương hiệu")
    with cac_tab[2]:
        hien_thi_tab_nhom(du_lieu_loc, "Channel_Used", "kênh marketing")
    with cac_tab[3]:
        hien_thi_tab_nhom(du_lieu_loc, "Campaign_Type", "loại chiến dịch")
    with cac_tab[4]:
        hien_thi_tab_nhom(
            du_lieu_loc, "Customer_Segment", "phân khúc khách hàng"
        )
    with cac_tab[5]:
        hien_thi_xu_huong(du_lieu_loc)
    with cac_tab[6]:
        hien_thi_anh_eda()
    with cac_tab[7]:
        hien_thi_bao_cao()
    with cac_tab[8]:
        hien_thi_revenue_prediction_ml(du_lieu)


if __name__ == "__main__":
    chay_dashboard()
