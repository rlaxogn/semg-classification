<div align="center">

# 🦾 sEMG 신호 기반 동작 분류 및 성능 비교 연구

<p>
  <b>표면 근전도(sEMG) 시계열 신호의 시간 영역(Time Domain) 특징 추출 기반 Machine Learning 동작 분류 파이프라인</b>
</p>

</div>

---

## 📌 1. 코드 설명
본 연구는 표면 근전도(sEMG) 시계열 신호로부터 시간 영역(Time Domain, TD) 특징을 추출하여 다중 동작(Gesture)을 분류하고, 다양한 머신러닝(ML) 알고리즘 간의 성능을 비교 분석한 프로젝트입니다.

### 🛠️ 사용한 데이터 및 전처리 방법
* **데이터**: 다중 채널 sEMG(표면 근전도) 센서 시계열 데이터셋
* **전처리 및 특징 추출 (`data.py`)**:
  1. <b>구간 자르기 및 윈도잉 (`cut`, `make_windows`)</b>: 연속적인 sEMG 신호를 설정한 슬라이딩 윈도우 크기 및 오버랩 비율에 따라 분할
  2. <b>시간 영역 특징 추출 (`td_features`)</b>: 각 윈도우 구간에서 MAV(Mean Absolute Value), RMS(Root Mean Square), Variance, Zero Crossing 등 신호 특성을 대변하는 시간 영역 통계 특징 수집
  3. <b>데이터 세트 구축 (`build`)</b>: 전체 데이터를 전처리 후 학습 및 검증용 `.npz` 포맷으로 정규화 및 저장 관리

---

### 🤖 사용한 AI/ML 모델 (`model.py`)
* <b>SVM (Support Vector Machine)</b>: 고차원 특징 공간에서 마진(Margin)을 최대화하는 초평면을 학습하여 분류하는 베이스라인 모델
* <b>Random Forest</b>: 다수의 결정 트리(Decision Tree) 앙상블을 통해 특징 간 비선형 상호작용을 포착하고 과적합을 방지하는 모델
* <b>XGBoost (eXtreme Gradient Boosting)</b>: 그래디언트 부스팅 기반으로 오차를 순차적으로 보완하여 높은 분류 정확도와 속도를 제공하는 모델

---

### 🏋️ 학습 및 테스트 방법
* **Train/Test Split**: 학습용과 평가용 데이터셋을 엄격히 격리 분할
* **데이터 관리**: `.npz` 파일 저장을 통한 데이터 로딩 및 실행 재현성 확보
* **하이퍼파라미터 최적화**: 각 모델별 하이퍼파라미터 튜닝을 통한 최적의 분류 성능 도출

---

### 💻 코드 실행 방법

VS Code 터미널 또는 대시보드 환경에서 아래 명령어를 순차적으로 실행합니다.

```bash
# 전체 데이터 전처리, 모델 학습 및 평가 통합 실행
python main.py
```

---

### 📂 코드 파일 구조

<pre>
├── 📄 <b>main.py</b>         : 전체 파이프라인(데이터 로드, 모델 학습, 평가) 실행 스크립트
├── 📄 <b>data.py</b>         : 데이터 전처리, 구간 분할(cut), 윈도잉(make_windows), 특징 추출(td_features) 모듈
├── 📄 <b>model.py</b>        : SVM, Random Forest, XGBoost 등 ML 모델 정의 및 학습 모듈
├── 📄 <b>evaluation.py</b>   : Accuracy, Precision, Recall, F1-Score 산출 및 Confusion Matrix 시각화 모듈
└── 📂 <b>data/</b>           : sEMG 원본 데이터 및 전처리 완료된 .npz 데이터 저장 폴더
</pre>

---

## 📊 2. 모델 성능 비교

### 모델별 성능 비교표

<table>
  <thead>
    <tr>
      <th align="center">Model</th>
      <th align="center">Accuracy</th>
      <th align="center">Precision</th>
      <th align="center">Recall</th>
      <th align="center">F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>XGBoost</b></td>
      <td align="center"><b>0.8945</b></td>
      <td align="center"><b>0.8982</b></td>
      <td align="center"><b>0.8945</b></td>
      <td align="center"><b>0.8958</b></td>
    </tr>
    <tr>
      <td align="center"><b>Random Forest</b></td>
      <td align="center">0.8620</td>
      <td align="center">0.8654</td>
      <td align="center">0.8620</td>
      <td align="center">0.8631</td>
    </tr>
    <tr>
      <td align="center"><b>SVM (RBF Kernel)</b></td>
      <td align="center">0.8115</td>
      <td align="center">0.8180</td>
      <td align="center">0.8115</td>
      <td align="center">0.8132</td>
    </tr>
  </tbody>
</table>

<br>

### 🔍 성능 분석
* **우수 모델**: <b>XGBoost</b> (Accuracy: <b>0.8945</b>, F1-score: <b>0.8958</b>)
* **분석 내용**:
  * 단일 경계 기반의 SVM에 비해 트리 기반 앙상블 모델(Random Forest, XGBoost)이 sEMG 시간 영역 특징(`td_features`) 간의 복잡한 비선형 관계를 더욱 잘 학습함.
  * 특히 **XGBoost**는 그래디언트 부스팅 방식을 통해 이전 트리의 잔차 오차를 연속적으로 보완하여 가장 뛰어난 동작 분류 성능을 기록함.

---

## 🔍 3. Confusion Matrix 분석

<div align="center">
  <img src="./confusion_matrix.png" alt="Confusion Matrix" width="600"/>
</div>

### 📈 모델 성능 상세 분석 (XGBoost 기준)

* **가장 잘 분류된 클래스**: <b>Class 0 (Rest/Idle)</b> (정확도 97.2%) 및 <b>Class 1 (Fist Grip)</b> (정확도 93.5%)
* **가장 많이 오분류된 클래스**: <b>Class 3 (Wrist Extension)</b> (정확도 78.4%)
* **주요 오분류 유형**: <b>True Class 3 → Predicted Class 4 (Wrist Flexion)</b>
* **오분류 원인 분석**:
  * 손목 굴곡(Flexion)과 신전(Extension) 동작은 전완근 주위 인접 근육군을 공통으로 활성화시킴.
  * 이로 인해 단순 시간 영역 통계 특징(MAV, RMS 등)의 분포가 두 클래스 간에 매우 유사하게 추출되어 모델이 명확히 구별하지 못하고 오분류하는 현상이 발생함.

---

## 🎯 4. 최종 결과

### 📝 실험 결과 요약
* **가장 성능이 우수한 모델**: <b>XGBoost</b> (Accuracy: 0.8945, F1-Score: 0.8958)
* **가장 성능이 낮은 모델**: <b>SVM</b> (Accuracy: 0.8115, F1-Score: 0.8132)
* **주요 개선 대상 클래스**: <b>Class 3 및 Class 4</b> (인접 근육군 사용 동작 간 상호 오분류 개선 필요)

---

### 💡 전체적인 실험 결과 및 느낀 점
* **시간 영역 특징(TD Features) 기반 ML 분류의 유효성**
  * `data.py`를 통해 추출한 MAV, RMS 등 시간 영역 특징만으로도 머신러닝 모델을 사용하여 약 89.5%의 높은 F1-Score를 달성할 수 있음을 확인했습니다. 연산량이 적어 실시간 근전도 제어 시스템 적용에 유리하다는 장점이 있습니다.

* **Confusion Matrix 기반의 세부 오분류 원인 도출**
  * 전체 정확도(Accuracy) 지표 외에도 혼동 행렬을 통해 인접 근육군을 사용하는 특정 동작(Class 3 vs Class 4) 간에 신호 특징 유사성으로 인한 오분류가 집중되어 있음을 발견했습니다.

* **향후 개선 방향**
  * 주파수 영역(Frequency Domain) 및 주파수-시간 영역(Wavelet) 특징을 추가 결합(Feature Concatenation)하여 신호 구별력을 높이는 시도가 필요합니다.
  * 오분류가 많은 클래스 간 구별을 위해 특징 선택(Feature Selection) 기법이나 클래스 가중치 손실 함수 적용을 검토할 예정입니다.