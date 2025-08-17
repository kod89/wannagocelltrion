# 🧪 AI 기반 품질 분석 통합 툴

이 Streamlit 애플리케이션은 다음과 같은 세 가지 주요 분석 기능을 하나로 통합한 포트폴리오용 도구입니다:

## 📌 기능 소개

### 1. 체외진단기기 성능 평가
- 평가 결과 파일을 업로드하여 정확도, 정밀도, 재현율, F1 점수, ROC 곡선 등을 계산
- PDF 보고서 자동 생성 및 다운로드

### 2. 시험 성적서 자동 검토/이상치 분석
- 시험 성적서 데이터를 기반으로 합불 판정 및 Z-score 이상치 분석
- 자동 요약 및 PDF 보고서 제공

### 3. 의약품 생산 배치 불량 예측
- 배치 데이터를 기반으로 불량 확률 예측 (Random Forest 모델 사용)
- 주요 변수 중요도 시각화
- 예측 결과 PDF 다운로드

---

## 📂 샘플 데이터

각 기능별 탭 내에서 샘플 CSV 파일을 다운로드 받아 테스트할 수 있습니다.

---

## 🚀 실행 방법

```bash
pip install -r requirements.txt
streamlit run integrated_ai_analysis_app.py
```

---

## 📄 생성되는 보고서 파일명 예시

- 체외진단기기_성능_평가_보고서_YYYYMMDD.pdf  
- 시험성적서_자동검토_보고서_YYYYMMDD.pdf  
- 배치_불량_예측_결과_YYYYMMDD.pdf

---

## 📬 문의

개발자: [kod89](https://github.com/kod89)  
깃허브: https://github.com/kod89/hypermax/tree/main
