# integrated_ai_analysis_app.py
import os
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# ReportLab (모든 PDF를 ReportLab로 생성: 한글/유니코드 안전)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------- 공통 설정 -----------------------
st.set_page_config(page_title="AI 품질 분석 통합 툴", layout="wide")
st.title("🧪 AI 기반 품질 분석 통합 툴")
sns.set_theme(style="whitegrid")

TODAY = datetime.today().strftime("%Y%m%d")

TABS = st.tabs([
    "체외진단기기 성능 평가",
    "시험 성적서 자동 검토/이상치 분석",
    "의약품 생산 배치 불량 예측",
])

# ----------------------- 유틸 함수 -----------------------
def safe_sample_download_button(label: str, path: str, fallback_filename: str, generator_fn):
    """
    경로의 파일이 존재하면 그 파일을 그대로 제공.
    없으면 generator_fn()으로 DataFrame을 생성해 CSV로 제공.
    """
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(label, f.read(), file_name=os.path.basename(path), mime="text/csv")
        else:
            df = generator_fn()
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(label, csv_bytes, file_name=fallback_filename, mime="text/csv")
            st.info(f"샘플 파일({path})을 찾지 못해, 임시 생성 데이터를 제공합니다: {fallback_filename}")
    except Exception as e:
        st.warning(f"샘플 파일 준비 중 문제가 발생해 임시 데이터를 제공합니다: {e}")
        df = generator_fn()
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(label, csv_bytes, file_name=fallback_filename, mime="text/csv")

def build_pdf_buffer(title: str, paragraphs: list, table_data: list | None = None, col_widths=None):
    """
    간단한 ReportLab PDF 생성 -> BytesIO 반환
    - title: 상단 제목(문자열)
    - paragraphs: 본문 단락 리스트(문자열)
    - table_data: 표 데이터(2차원 리스트) 또는 None
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20
    )
    styles = getSampleStyleSheet()
    elems = [Paragraph(title, styles["Title"]), Spacer(1, 10), Paragraph(f"날짜: {TODAY}", styles["Normal"]), Spacer(1, 10)]

    if table_data is not None:
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elems += [table, Spacer(1, 12)]

    for p in paragraphs:
        elems.append(Paragraph(p.replace("\n", "<br/>"), styles["Normal"]))
        elems.append(Spacer(1, 8))

    doc.build(elems)
    buf.seek(0)
    return buf

# ----------------------- 샘플 생성기 -----------------------
def gen_sample_clia():
    # True_Label(0/1), Test_Result(0/1) 예시
    n = 80
    rng = np.random.default_rng(42)
    true = rng.integers(0, 2, size=n)
    # 대체로 정확도 0.85 정도로 맞추는 샘플
    noise = rng.random(n) < 0.15
    pred = np.where(noise, 1 - true, true)
    return pd.DataFrame({"True_Label": true, "Test_Result": pred})

def gen_sample_qc():
    items = [f"Item{i+1}" for i in range(12)]
    rng = np.random.default_rng(0)
    lows = rng.normal(95, 1.0, len(items)).round(2)
    highs = (lows + rng.normal(4.0, 0.6, len(items))).round(2)
    vals = (lows + rng.normal(2.0, 1.2, len(items))).round(2)
    return pd.DataFrame({
        "항목명": items,
        "측정값": vals,
        "기준하한": lows,
        "기준상한": highs
    })

def gen_sample_batch():
    n = 50
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "Temperature_C": rng.normal(25, 1.5, n).round(3),
        "Pressure_bar": rng.normal(1.0, 0.05, n).round(3),
        "MixingSpeed_rpm": rng.normal(120, 10, n).round(2),
        "pH": rng.normal(7.0, 0.3, n).round(3),
        "Yield_percent": rng.normal(95, 2, n).round(3),
        "Contaminant_ppm": rng.exponential(1.0, n).round(3),
    })

# ==========================================================
# TAB 1: 체외진단기기 성능 평가
# ==========================================================
with TABS[0]:
    st.header("🔬 체외진단기기 성능 평가")

    # 샘플 다운로드 (없으면 생성)
    safe_sample_download_button(
        "📥 샘플 데이터 다운로드",
        path="3.eval sample_data.csv",
        fallback_filename="eval_sample_data.csv",
        generator_fn=gen_sample_clia
    )

    up = st.file_uploader("📁 평가 결과 업로드 (CSV 또는 Excel)", type=["csv", "xlsx"], key="clia")
    if up:
        df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)

        # 컬럼 체크
        required_cols = ["True_Label", "Test_Result"]
        if not all(c in df.columns for c in required_cols):
            st.error(f"필수 컬럼 누락: {required_cols}. 업로드 파일의 컬럼을 확인하세요.")
        else:
            y_true = df["True_Label"]
            y_pred = df["Test_Result"]

            # 지표
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # 가능하면 score 계열 컬럼으로 ROC, 없으면 y_pred 사용
            score_col = next((c for c in df.columns if c.lower() in ["score", "prob", "probability", "y_score"]), None)
            y_score = df[score_col] if score_col else y_pred
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            st.subheader("✅ 성능 지표 요약")
            st.dataframe(pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
                "Value": [acc, prec, rec, f1, roc_auc]
            }), use_container_width=True)

            st.subheader("📊 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            st.subheader("📈 ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

            # PDF (ReportLab): 표 + 코멘트
            st.subheader("📄 PDF 보고서 다운로드")
            metrics_table = [
                ["Metric", "Value"],
                ["Accuracy", f"{acc:.3f}"],
                ["Precision", f"{prec:.3f}"],
                ["Recall", f"{rec:.3f}"],
                ["F1 Score", f"{f1:.3f}"],
                ["AUC", f"{roc_auc:.3f}"],
            ]
            paragraphs = [
                "업로드된 데이터로 체외진단기기의 성능을 평가했습니다.",
                "민감도/특이도 개선이 필요한 경우 임계값 조정 또는 데이터 보강을 고려하세요."
            ]
            pdf_buf = build_pdf_buffer(
                title="체외진단기기 성능 평가 보고서",
                paragraphs=paragraphs,
                table_data=metrics_table,
                col_widths=[120, 100]
            )
            st.download_button(
                "📥 보고서 다운로드",
                pdf_buf,
                file_name=f"체외진단기기_성능_평가_보고서_{TODAY}.pdf",
                mime="application/pdf"
            )

# ==========================================================
# TAB 2: 시험 성적서 자동 검토/이상치 분석
# ==========================================================
with TABS[1]:
    st.header("📄 시험 성적서 자동 검토/이상치 분석")

    safe_sample_download_button(
        "📥 샘플 데이터 다운로드",
        path="2.sample_qc_data.csv",
        fallback_filename="sample_qc_data.csv",
        generator_fn=gen_sample_qc
    )

    up = st.file_uploader("📁 시험 성적서 업로드 (CSV/XLSX)", type=["csv", "xlsx"], key="qc")
    if up:
        df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)

        # 컬럼 한글 → 영문 변환 시도 (사용자 편의)
        rename_dict = {
            "항목명": "Item",
            "측정값": "Value",
            "기준하한": "Lower Limit",
            "기준상한": "Upper Limit",
        }
        df = df.rename(columns=rename_dict)

        required_cols = ["Item", "Value", "Lower Limit", "Upper Limit"]
        if not all(c in df.columns for c in required_cols):
            st.error(f"필수 컬럼 누락: {required_cols}. 업로드 파일의 컬럼을 확인하세요.")
        else:
            # 합부 판정 + 이상치
            df["Result"] = df.apply(lambda r: "Pass" if r["Lower Limit"] <= r["Value"] <= r["Upper Limit"] else "Fail", axis=1)
            df["Z-score"] = zscore(df["Value"])

            st.success("✅ 파일이 정상 처리되었습니다.")
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Z-score 이상치")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(df["Item"].astype(str), df["Z-score"])
            ax.axhline(2, color="red", linestyle="--")
            ax.axhline(-2, color="red", linestyle="--")
            ax.set_ylabel("Z-score")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            total = len(df)
            passed = (df["Result"] == "Pass").sum()
            failed = total - passed
            comment = (
                "모든 항목이 기준 내에 있습니다. 추가 조치는 필요하지 않습니다."
                if failed == 0 else
                f"{failed}개 항목이 기준을 벗어났습니다. 원인 분석과 교정 조치를 검토하세요."
            )

            # PDF (ReportLab)
            st.subheader("📄 PDF 요약 보고서 다운로드")
            table_data = [["Item", "Value", "Spec (Low ~ High)", "Result"]]
            for _, r in df.iterrows():
                table_data.append([str(r["Item"]), str(r["Value"]), f"{r['Lower Limit']} ~ {r['Upper Limit']}", r["Result"]])

            paragraphs = [
                f"총 {total}개 항목 중 Pass: {passed}, Fail: {failed}.",
                comment
            ]
            pdf_buf = build_pdf_buffer(
                title="시험 성적서 자동 검토 보고서",
                paragraphs=paragraphs,
                table_data=table_data,
                col_widths=[110, 70, 130, 70]
            )
            st.download_button(
                "📥 보고서 다운로드",
                pdf_buf,
                file_name=f"시험성적서_자동검토_보고서_{TODAY}.pdf",
                mime="application/pdf"
            )

# ==========================================================
# TAB 3: 의약품 생산 배치 불량 예측
# ==========================================================
with TABS[2]:
    st.header("📉 의약품 생산 배치 불량 예측")

    safe_sample_download_button(
        "📥 샘플 데이터 다운로드",
        path="1.sample_data_100.csv",
        fallback_filename="sample_batch_data.csv",
        generator_fn=gen_sample_batch
    )

    # 간단한 학습용 랜덤포레스트 (데모용)
    @st.cache_resource(show_spinner=False)
    def train_demo_model():
        n = 5000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Temperature_C": rng.normal(25, 1.5, n),
            "Pressure_bar": rng.normal(1.0, 0.05, n),
            "MixingSpeed_rpm": rng.normal(120, 10, n),
            "pH": rng.normal(7.0, 0.3, n),
            "Yield_percent": rng.normal(95, 2, n),
            "Contaminant_ppm": rng.exponential(1.0, n),
        })
        df["Defective"] = ((df["Yield_percent"] < 92) | (df["Contaminant_ppm"] > 5)).astype(int)
        X = df.drop(columns="Defective")
        y = df["Defective"]
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)
        return model

    model = train_demo_model()

    up = st.file_uploader("📁 배치 데이터 업로드 (CSV)", type=["csv"], key="batch")
    if up:
        df = pd.read_csv(up)
        # 단위 표기 제거 (예: 'Temperature_C (℃)' → 'Temperature_C')
        df.columns = df.columns.str.replace(r"\s*\(.*?\)", "", regex=True)

        required_cols = ["Temperature_C", "Pressure_bar", "MixingSpeed_rpm", "pH", "Yield_percent", "Contaminant_ppm"]
        if not all(c in df.columns for c in required_cols):
            st.error(f"필수 컬럼 누락: {required_cols}. 업로드 파일의 컬럼을 확인하세요.")
        else:
            proba = model.predict_proba(df[required_cols])[:, 1]
            df["불량 확률(%)"] = (proba * 100).round(2)
            df["예측 결과"] = np.where(proba >= 0.5, "불량", "정상")
            st.success("예측 완료! 아래 표에서 결과를 확인하세요.")
            st.dataframe(df, use_container_width=True)

            # PDF (ReportLab)
            st.subheader("📄 PDF 보고서 다운로드")
            table_data = [["Index", "예측 결과", "불량 확률(%)"]]
            for i, r in df.iterrows():
                table_data.append([str(i), str(r["예측 결과"]), f"{float(r['불량 확률(%)']):.2f}"])

            paragraphs = [
                "업로드된 배치 데이터로 불량 가능성을 예측했습니다.",
                "불량 확률이 높은 샘플은 공정 점검 및 원인 분석을 권장합니다."
            ]
            pdf_buf = build_pdf_buffer(
                title="의약품 생산 배치 불량 예측 보고서",
                paragraphs=paragraphs,
                table_data=table_data,
                col_widths=[60, 90, 90]
            )
            st.download_button(
                "📥 보고서 다운로드",
                pdf_buf,
                file_name=f"배치_불량_예측_결과_{TODAY}.pdf",
                mime="application/pdf"
            )

# ----------------------- 푸터 -----------------------
st.markdown("""
---
[🔗 GitHub 저장소](https://github.com/kod89/hypermax/tree/main)  
테스트는 각 탭의 **샘플 데이터 다운로드** 버튼으로 파일을 받아 업로드하면 바로 가능합니다.
""")
