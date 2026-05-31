"""P2 멀티모달 융합 — 통합 샘플 스키마.

한 개의 조립된 멀티모달 샘플이 어떤 필드/차원으로 구성되는지의 단일 진실원천(single
source of truth). 클래스 조건부 조립기(방법 A), 실데이터 피처 추출기(2단계), 융합 모델이
모두 이 스키마를 공유한다.

설계 메모
- ECG 채널은 P1(ECG 검출기) 출력 계약을 그대로 따른다.
- IMU/SpO2는 1단계에서 핸드크래프트 피처 벡터로 다룬다(raw 1D-CNN은 2단계 옵션).
- `gate_tier`는 문자열 대신 정수 코드로 저장한다(use=0, mask=1, alert=2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# 클래스 taxonomy (P2 출력 5분류)
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "normal_rest",      # 0 정상(안정)
    "normal_active",    # 1 정상(운동/활동)
    "cardiac",          # 2 심혈관계 응급
    "impact",           # 3 외부충격(낙상·충돌)
    "hypoxia",          # 4 저산소
]
NUM_CLASSES = len(CLASS_NAMES)

CARDIAC_PROB_NAMES = ["NSR", "AF", "Ischemia", "Conduction", "Ectopic"]
NUM_CARDIAC = len(CARDIAC_PROB_NAMES)

GATE_TIERS = ["use", "mask", "alert"]  # index = 정수 코드
GATE_TIER_CODE: Dict[str, int] = {t: i for i, t in enumerate(GATE_TIERS)}

# ---------------------------------------------------------------------------
# 모달리티별 피처 이름 (순서 = 벡터 인덱스)
# ---------------------------------------------------------------------------
EMB_DIM = 768  # ECG-FM mean-pool 임베딩 차원 (P1)

# IMU 핸드크래프트 피처 (가속도 3축 + 자이로 3축에서 추출)
IMU_FEATURES: List[str] = [
    "smv_mean",      # 0 신호크기벡터 평균 (정지≈1g)
    "smv_std",       # 1 SMV 표준편차 (활동성)
    "smv_peak",      # 2 SMV 최대 (impact peak, 단위 g)
    "smv_min",       # 3 SMV 최소 (자유낙하 트로프, 낙상≈0.3g)
    "jerk_peak",     # 4 가속도 미분 최대 (충격 급변)
    "gyro_peak",     # 5 각속도 최대 (rad/s)
    "gyro_energy",   # 6 평균 회전율 (∫|ω|dt 근사)
    "tilt_change",   # 7 자세각 변화량 (deg, 중력벡터 기준)
    "act_energy",    # 8 가속도 크기 분산 (활동 에너지)
    "dom_freq",      # 9 지배 주파수 (Hz, 보행 주기성)
    "spec_entropy",  # 10 스펙트럴 엔트로피 (0~1, 규칙성↔무작위)
    "impact_count",  # 11 임계 초과 피크 수 (단발 vs 주기)
]
IMU_DIM = len(IMU_FEATURES)

# SpO2 피처 (저샘플링·서서히 변하는 신호 → 피처 기반)
SPO2_FEATURES: List[str] = [
    "spo2_mean",       # 0 평균 (%)
    "spo2_nadir",      # 1 최저값 (%)
    "spo2_current",    # 2 현재(마지막) 값 (%)
    "desat_rate",      # 3 최대 하강율 (%p/분, 양수)
    "time_below_90",   # 4 90% 미만 시간 비율 (0~1)
    "time_below_88",   # 5 88% 미만 시간 비율 (0~1)
    "recovery_slope",  # 6 회복 기울기 (%p/분)
    "spo2_std",        # 7 변동성 (표준편차)
]
SPO2_DIM = len(SPO2_FEATURES)

MODALITIES = ["ecg", "imu", "spo2"]


@dataclass
class MultimodalSample:
    """조립된 단일 멀티모달 샘플."""

    # --- ECG 채널 (P1 출력) ---
    ecg_embedding: np.ndarray            # [768]
    cardiac_probs: np.ndarray            # [5]
    emergency_score: float
    reliability: float                   # 0~1 (높을수록 불량)
    gate_tier: int                       # 0=use,1=mask,2=alert
    hr_bpm: float
    rhythm_regularity: float             # 0~1

    # --- IMU / SpO2 피처 ---
    imu_feat: np.ndarray                 # [IMU_DIM]
    spo2_feat: np.ndarray                # [SPO2_DIM]

    # --- 레이블 & 가용성 마스크 ---
    label: int                           # 0~4
    modality_mask: np.ndarray = field(   # [3] (ecg,imu,spo2) 1=존재 0=결측
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )

    # --- 출처 추적 (정직성: 모달리티별 real vs synth) ---
    src: Dict[str, str] = field(default_factory=dict)

    def flat_ecg_aux(self) -> np.ndarray:
        """임베딩을 제외한 ECG 보조 피처(P1 score+physio+게이트)를 벡터로."""
        return np.concatenate([
            self.cardiac_probs.astype(np.float32),
            np.array([self.emergency_score, self.reliability,
                      float(self.gate_tier), self.hr_bpm,
                      self.rhythm_regularity], dtype=np.float32),
        ])
