"""P2-B 멀티모달 응급 융합 — cross-modal attention 트랙 (자체완결 패키지).

구성:
  schema       샘플 스키마 (768 임베딩 + 10 ecg_aux + 12 IMU + 8 SpO2, 5분류)
  priors       클래스 조건부 사전분포 + 실데이터 캘리브레이션 샘플러
  features     실 raw 윈도우 → IMU/SpO2 핸드크래프트 피처
  dataset      조립기(실 P1 임베딩) + Dataset/loader + proxy 빌드
  model        ConcatMLP / GatedFusionModel / CrossModalAttentionFusion
  metrics      2D 평가 (confounder 오경보율 × 결측 강건성)
  confounders  실데이터 앵커 confounder 셋 (만성 ECG·모션·무호흡) — 비순환
  xai          attention 가중치 추출·요약
  train        재사용 학습 루프
  run_dryrun   proxy 드라이런 오케스트레이션 → 2D 리포트 + 아티팩트
"""
__version__ = "0.1.0"
