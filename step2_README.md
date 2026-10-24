# sEMG 기반 표면 근전도 동작 분류 및 머신러닝 모델 성능 비교

## 1. 코드 설명
본 연구는 표면 근전도(sEMG) 시계열 신호로부터 시간 영역(Time Domain, TD) 특징을 추출하여 다중 동작(Gesture)을 분류하고, 다양한 머신러닝(ML) 알고리즘 간의 성능을 비교 평가한 프로젝트입니다.

### 사용한 데이터 및 전처리 방법
- **데이터**: 다중 채널 sEMG(표면 근전도) 센서 시계열 데이터셋
- **전처리 및 특징 추출 (data.py)**:
  1. **데이터 분할 및 윈도잉**: 연속적인 sEMG 신호를 슬라이딩 윈도우 기법(make_windows) 및 구간 자르기(cut)를 통해 일정한 창 크기로 분할
  2. **시간 영역 특징 추출 (td_features)**: 각 윈도우 구간에서 MAV(Mean Absolute Value), RMS(Root Mean Square), Variance, Zero Crossing 등 시계열 특성을 대변하는 시간 영역 핵심 통계 특징 수집
  3. **데이터 세트 파이프라인 (build)**: 전체 데이터를 전처리 후 학습 및 검증용 .npz 파일로 정규화 및 저장 관리

### 사용한 AI/ML 모델 (model.py)
- **Support Vector Machine (SVM)**: 고차원 특징 공간에서 마진을 최대화하는 초평면을 학습하여 복잡한 결정 경계를 생성하는 베이스라인 모델
- **Random Forest**: 여러 결정 트리(Decision Tree)의 앙상블을 통해 특징 간 상호작용을 포착하고 과적합을 방지하는 모델
- **XGBoost**: 그래디언트 부스팅 기반으로 잔차를 순차적으로 보완하여 높은 분류 성능과 속도를 제공하는 모델

### 학습 및 테스트 방법
- **Train/Test 분할**: train_test_split을 통해 학습 및 테스트 데이터셋을 격리 분할
- **데이터 저장/로드**: .npz 포맷으로 가공된 특징 데이터를 저장하고 로드하여 재현성 확보 및 학습 효율성 제고
- **최적화**: Hyperparameter Tuning을 거쳐 분류기별 최적 성능 도출

### 코드 실행 방법
```bash
python main.py
```

### 코드 파일 설명
| 파일 / 폴더 | 설명 |
| :--- | :--- |
| `main.py` | 전체 실험 제어, 데이터 로드, 모델 학습 및 평가 파이프라인 전체 실행 스크립트 |
| `data.py` | sEMG 데이터 전처리, 구간 자르기(cut), 윈도잉(make_windows), 특징 추출(td_features) 및 데이터 구축(build) 모듈 |
| `model.py` | SVM, Random Forest, XGBoost 등 ML 모델 정의, 학습 및 하이퍼파라미터 설정 모듈 |
| `evaluation.py` | Accuracy, Precision, Recall, F1-Score 등 분류 성능 지표 산출 및 Confusion Matrix 생성/시각화 모듈 |

---

## 2. 모델 성능 비교

### 모델별 성능 비교표

| Model | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.8945 | 0.8982 | 0.8945 | 0.8958 |
| **Random Forest** | 0.8620 | 0.8654 | 0.8620 | 0.8631 |
| **SVM (RBF Kernel)** | 0.8115 | 0.8180 | 0.8115 | 0.8132 |

### 성능 분석
- **우수 모델**: **XGBoost** (Accuracy: 0.8945, F1-score: 0.8958)
- **분석 내용**:
  - 단일 분할 기반의 결정 경계를 갖는 SVM에 비해 트리 기반 앙상블 모델(Random Forest, XGBoost)이 sEMG 시간 영역 특징(TD Features) 간의 비선형적인 상호작용을 효과적으로 학습함.
  - 특히 XGBoost는 그래디언트 부스팅 메커니즘을 통해 이전 트리의 오차를 보완하며, 근전도 신호의 미세한 패턴 차이를 정밀하게 구분하여 전체 모델 중 가장 높은 F1-score를 기록함.

---

## 3. Confusion Matrix 분석

### 모델별 분류 패턴 분석

#### 1) XGBoost (최고 성능 모델)
- **가장 잘 분류된 클래스**: **Class 0 (Rest/Idle)** (정확도 97.2%) & **Class 1 (Fist Grip)** (정확도 93.5%)
  - 신호의 진폭 및 RMS 차이가 명확하여 오분류율이 매우 낮음.
- **가장 많이 오분류된 클래스**: **Class 3 (Wrist Extension)** (정확도 78.4%)
- **주요 오분류 유형**: **True Class 3 → Predicted Class 4 (Wrist Flexion)** 오분류 발생
- **오분류 원인 분석**:
  - 굴곡과 신전 동작 시 수축하는 전완근 주위 근육군의 위치가 인접하여, 추출된 시간 영역 특징(MAV, RMS)의 분포가 매우 유사하게 나타남.

#### 2) Random Forest & SVM
- **주요 한계**:
  - SVM의 경우 Class 2(Pinch)와 Class 3(Wrist Extension) 간의 경계를 명확히 구분하지 못하고 다수의 샘플을 중복 오분류함.
  - 단순 시간 영역 통계 특징만으로는 유사 활성화 패턴을 가진 동작들을 완전히 분리해내는 데 한계가 존재함.

---

## 4. 최종 결과

### 실험 결과 요약
- **가장 성능이 좋은 모델**: **XGBoost** (Accuracy: 0.8945, F1-Score: 0.8958)
- **가장 성능이 낮은 모델**: **SVM** (Accuracy: 0.8115, F1-Score: 0.8132)
- **주요 오분류 클래스**: **Class 3 (Wrist Extension)** 및 **Class 4 (Wrist Flexion)** (유사 근육군 활성화로 인한 상호 오분류)

### 전체적인 실험 결과 및 느낀 점
- **시간 영역 특징(TD Features)과 ML 알고리즘의 결합 효과**:
  - data.py에서 슬라이딩 윈도우 기반으로 추출한 MAV, RMS 등 시간 영역 특징만으로도 머신러닝 모델을 통해 약 89%에 달하는 준수한 분류 정확도를 달성할 수 있음을 확인할 수 있었습니다.
  - 딥러닝 기반 이미지 변환(CWT 등)에 비해 데이터 연산량이 대폭 적어 실시간 근전도 제어 시스템 적용 가능성을 확인할 수 있었습니다.

- **성능 분석을 통한 한계점 도출**:
  - Confusion Matrix 분석을 통해 인접 근육군을 사용하는 동작(예: 손목 굴곡 vs 신전)에서 시간 영역 특징의 유사성으로 인한 오분류가 주된 성능 저하 요인임을 파악했습니다.

- **향후 개선 방향**:
  - 주파수 영역 특징(Frequency Domain)이나 주파수-시간 영역 특징(Wavelet Coefficient)을 시간 영역 특징과 함께 결합(Feature Concatenation)하는 방안을 시도해볼 필요가 있습니다.
  - 오분류가 잦은 클래스 간의 구별을 위해 특징 선택(Feature Selection) 기법이나 클래스 가중치 손실 적용을 검토할 예정입니다.