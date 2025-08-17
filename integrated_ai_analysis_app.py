import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score,
    recall_score, f1_score
)
from scipy.stats import zscore
from io import BytesIO
from fpdf import FPDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
import os

st.set_page_config(page_title="AI 품질 분석 통합 툴", layout="wide")
st.title("🧪 AI 기반 품질 분석 통합 툴")

tabs = st.tabs([
    "체외진단기기 성능 평가",
    "시험 성적서 자동 검토/이상치 분석",
    "의약품 생산 배치 불량 예측"
])

today = datetime.today().strftime("%Y%m%d")

# --------------------- TAB 1 ---------------------
with tabs[0]:
    st.header("🔬 체외진단기기 성능 평가")
    with open("3.eval sample_data.csv", "rb") as f:
        st.download_button("📥 샘플 데이터 다운로드", f, file_name="3.eval sample_data.csv", mime="text/csv")

    uploaded_file = st.file_uploader("📁 평가 결과 업로드 (CSV 또는 Excel)", type=["csv", "xlsx"], key="clia")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
            y_true = df["True_Label"]
            y_pred = df["Test_Result"]

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            st.subheader("✅ 성능 지표 요약")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
                "Value": [acc, prec, rec, f1, roc_auc]
            })
            st.dataframe(metrics_df)

            st.subheader("📊 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            st.pyplot(fig_cm)

            st.subheader("📈 ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            st.pyplot(fig_roc)

            st.subheader("📄 PDF 보고서 다운로드")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="체외진단기기 성능 평가 보고서", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 8, txt=f"""
[요약 지표]
- Accuracy: {acc:.2f}
- Precision: {prec:.2f}
- Recall: {rec:.2f}
- F1 Score: {f1:.2f}
- AUC: {roc_auc:.2f}

본 성능 평가는 업로드된 데이터에 기반하여 실행되었습니다.
결과에 따라 민감도 및 특이도 향상을 위한 추가 검토가 권장될 수 있습니다.
""")
            pdf_buffer = BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)
            st.download_button(
                label="📥 보고서 다운로드",
                data=pdf_buffer,
                file_name=f"체외진단기기_성능_평가_보고서_{today}.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")

# --------------------- TAB 2 ---------------------
with tabs[1]:
    st.header("📄 시험 성적서 자동 검토/이상치 분석")
    with open("2.sample_qc_data.csv", "rb") as f:
        st.download_button("📥 샘플 데이터 다운로드", f, file_name="2.sample_qc_data.csv", mime="text/csv")

    uploaded_file = st.file_uploader("📁 시험 성적서 업로드 (CSV/XLSX)", type=["csv", "xlsx"], key="qc")
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        df = df.rename(columns={
            "항목명": "Item", "측정값": "Value",
            "기준하한": "Lower Limit", "기준상한": "Upper Limit"
        })
        df["Result"] = df.apply(lambda r: "Pass" if r["Lower Limit"] <= r["Value"] <= r["Upper Limit"] else "Fail", axis=1)
        df["Z-score"] = zscore(df["Value"])

        st.dataframe(df)

        st.subheader("📈 이상치 시각화 (Z-score)")
        fig, ax = plt.subplots()
        ax.bar(df["Item"], df["Z-score"])
        ax.axhline(2, color='red', linestyle='--')
        ax.axhline(-2, color='red', linestyle='--')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        failed = (df["Result"] == "Fail").sum()
        comment = "모든 항목이 기준 내에 있습니다." if failed == 0 else f"{failed}개 항목이 기준을 벗어났습니다."

        st.subheader("📄 PDF 요약 보고서 다운로드")
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("시험 성적서 자동 검토 보고서", styles["Title"]),
            Paragraph(f"날짜: {today}", styles["Normal"]),
            Spacer(1, 12),
        ]
        table_data = [["Item", "Value", "Low ~ High", "Result"]] + [
            [r["Item"], r["Value"], f"{r['Lower Limit']} ~ {r['Upper Limit']}", r["Result"]] for _, r in df.iterrows()
        ]
        table = Table(table_data, colWidths=[100, 60, 120, 60])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        elements.append(table)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(comment, styles["Normal"]))
        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            "📥 보고서 다운로드",
            buffer,
            file_name=f"시험성적서_자동검토_보고서_{today}.pdf",
            mime="application/pdf"
        )

# --------------------- TAB 3 ---------------------
with tabs[2]:
    st.header("📉 의약품 생산 배치 불량 예측")
    with open("1.sample_data_100.csv", "rb") as f:
        st.download_button("📥 샘플 데이터 다운로드", f, file_name="1.sample_data_100.csv", mime="text/csv")

    def train_model():
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "Temperature_C": np.random.normal(25, 1.5, n),
            "Pressure_bar": np.random.normal(1.0, 0.05, n),
            "MixingSpeed_rpm": np.random.normal(120, 10, n),
            "pH": np.random.normal(7.0, 0.3, n),
            "Yield_percent": np.random.normal(95, 2, n),
            "Contaminant_ppm": np.random.exponential(1.0, n),
        })
        df["Defective"] = ((df["Yield_percent"] < 92) | (df["Contaminant_ppm"] > 5)).astype(int)
        X = df.drop("Defective", axis=1)
        y = df["Defective"]
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)
        return model

    model = train_model()
    uploaded_file = st.file_uploader("📁 배치 데이터 업로드", type=["csv"], key="batch")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required = ["Temperature_C", "Pressure_bar", "MixingSpeed_rpm", "pH", "Yield_percent", "Contaminant_ppm"]
        if all(col in df.columns for col in required):
            X = df[required]
            proba = model.predict_proba(X)[:, 1]
            df["불량 확률(%)"] = (proba * 100).round(2)
            df["예측 결과"] = np.where(proba >= 0.5, "불량", "정상")
            st.dataframe(df)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="의약품 생산 배치 불량 예측 보고서", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.ln(10)
            for i, row in df.iterrows():
                pdf.cell(0, 10, txt=f"{i+1}. 예측: {row['예측 결과']}, 확률: {row['불량 확률(%)']}%", ln=True)
            pdf_buffer = BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)

            st.download_button(
                "📥 보고서 다운로드",
                pdf_buffer,
                file_name=f"배치_불량_예측_결과_{today}.pdf",
                mime="application/pdf"
            )
        else:
            st.error(f"❗ 필수 컬럼 누락: {required}")
