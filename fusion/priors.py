"""클래스 조건부 사전분포 (방법 A).

2단계 실데이터 보정 이력:
  - 클래스 1(ADL) + 클래스 3(낙상): SisFall 실측 통계로 보정 (2026-05-30)
    SisFall: 4505샘플, 200Hz, 허리 착용 (SA01~SA23 청년 + SE01~SE15 노인)
    실측 통계: extract_sisfall_features.py 출력 참조
  - 클래스 0·2·4: 문헌 기반 유지 (PTT-PPG 완료 후 0·1 추가 보정 예정)

분포 표기: (mean, std, lo, hi) = 절단정규(mean±std, [lo,hi] 클립).

단위: 가속도 g, 자이로 rad/s, jerk_peak = Δ(smv)*fs (g/s 단위, fs=200Hz 기준).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from fusion.schema import (CARDIAC_PROB_NAMES, IMU_FEATURES, NUM_CARDIAC,
                           SPO2_FEATURES)

# P1 게이트 임계값 (단일리드 검출기 reliability 게이트 확정값)
T_MASK = 0.2155
T_ALERT = 0.4753

Spec = Tuple[float, float, float, float]  # (mean, std, lo, hi)

# ---------------------------------------------------------------------------
# IMU 피처 사전분포 [클래스 → {피처명: (mean,std,lo,hi)}]
# ★ 통일 프로토콜 보정 (2026-05-30): 200Hz · 3초 윈도우 동일 조건 실데이터
#   0/1/2/4 ← PTT-PPG (500→200Hz 리샘플), 3 ← SisFall (200Hz)
#   calibrate_imu_priors.py 출력. fs·윈도우 불일치 문제 해결 → 전 클래스 동일 스케일.
#   2(cardiac)·4(hypoxia)는 저활동이라 rest(PTT sit) 분포 기반 (IMU로는 rest와 구분 불가
#   = 모달리티 분리 설계상 정상: 심혈관은 ECG, 저산소는 SpO2가 구분).
# ---------------------------------------------------------------------------
IMU_PRIORS: Dict[int, Dict[str, Spec]] = {
    # 0 rest ← PTT-PPG sit (n=2855 windows)
    0: {
        "smv_mean": (0.982, 0.012, 0.962, 1.005), "smv_std": (0.002, 0.002, 0.001, 0.007),
        "smv_peak": (0.990, 0.016, 0.968, 1.023), "smv_min": (0.974, 0.015, 0.935, 0.999),
        "jerk_peak": (1.340, 1.554, 0.500, 4.742), "gyro_peak": (0.001, 0.001, 0.000, 0.004),
        "gyro_energy": (0.001, 0.002, 0.000, 0.007), "tilt_change": (1.069, 1.056, 0.320, 3.791),
        "act_energy": (0.0002, 0.0005, 0.000, 0.003), "dom_freq": (31.070, 16.140, 2.333, 43.667),
        "spec_entropy": (0.719, 0.080, 0.541, 0.849), "impact_count": (0.000, 0.050, 0.000, 0.000),
    },
    # 1 active ← PTT-PPG walk + run (n=5560 windows)
    1: {
        "smv_mean": (1.025, 0.045, 0.958, 1.138), "smv_std": (0.136, 0.100, 0.027, 0.447),
        "smv_peak": (1.362, 0.238, 1.101, 2.049), "smv_min": (0.735, 0.201, 0.190, 0.989),
        "jerk_peak": (9.461, 6.694, 1.868, 27.781), "gyro_peak": (0.037, 0.016, 0.016, 0.080),
        "gyro_energy": (0.048, 0.019, 0.020, 0.098), "tilt_change": (28.057, 13.741, 10.950, 64.688),
        "act_energy": (0.029, 0.047, 0.001, 0.200), "dom_freq": (2.019, 1.202, 0.667, 4.667),
        "spec_entropy": (0.400, 0.083, 0.193, 0.519), "impact_count": (0.055, 0.417, 0.000, 1.000),
    },
    # 2 cardiac ← PTT-PPG sit 기반 (저활동 + 통증/불안 약간의 움직임 → std 확대)
    2: {
        "smv_mean": (0.982, 0.030, 0.92, 1.08), "smv_std": (0.010, 0.020, 0.001, 0.10),
        "smv_peak": (1.05, 0.12, 0.97, 1.5),    "smv_min": (0.94, 0.05, 0.75, 1.0),
        "jerk_peak": (2.0, 2.5, 0.5, 12.0),     "gyro_peak": (0.015, 0.03, 0.0, 0.15),
        "gyro_energy": (0.008, 0.015, 0.0, 0.08),"tilt_change": (3.0, 4.0, 0.3, 18.0),
        "act_energy": (0.002, 0.008, 0.0, 0.04),"dom_freq": (25.0, 16.0, 1.5, 44.0),
        "spec_entropy": (0.71, 0.09, 0.5, 0.88),"impact_count": (0.0, 0.1, 0.0, 1.0),
    },
    # 3 impact ← SisFall fall (n=1798)
    3: {
        "smv_mean": (0.546, 0.042, 0.490, 0.674), "smv_std": (0.328, 0.135, 0.129, 0.667),
        "smv_peak": (3.688, 2.064, 1.174, 9.442), "smv_min": (0.133, 0.071, 0.024, 0.293),
        "jerk_peak": (280.531, 347.865, 27.502, 1445.356), "gyro_peak": (8.260, 3.458, 2.879, 15.900),
        "gyro_energy": (4.017, 0.251, 3.732, 4.778), "tilt_change": (129.791, 28.095, 71.067, 173.343),
        "act_energy": (0.126, 0.107, 0.017, 0.445), "dom_freq": (1.965, 1.083, 0.333, 4.333),
        "spec_entropy": (0.622, 0.074, 0.483, 0.807), "impact_count": (0.940, 0.583, 0.000, 2.000),
    },
    # 4 hypoxia ← PTT-PPG sit 기반 (저활동, rest와 거의 동일 — SpO2가 구분)
    4: {
        "smv_mean": (0.982, 0.020, 0.93, 1.06), "smv_std": (0.004, 0.008, 0.001, 0.04),
        "smv_peak": (1.00, 0.06, 0.97, 1.3),    "smv_min": (0.96, 0.03, 0.85, 1.0),
        "jerk_peak": (1.5, 1.6, 0.5, 7.0),      "gyro_peak": (0.005, 0.01, 0.0, 0.06),
        "gyro_energy": (0.003, 0.006, 0.0, 0.03),"tilt_change": (1.5, 2.0, 0.3, 10.0),
        "act_energy": (0.0005, 0.002, 0.0, 0.01),"dom_freq": (28.0, 16.0, 2.0, 44.0),
        "spec_entropy": (0.715, 0.085, 0.52, 0.85),"impact_count": (0.0, 0.05, 0.0, 0.0),
    },
}

# ---------------------------------------------------------------------------
# SpO2 피처 사전분포
# ---------------------------------------------------------------------------
SPO2_PRIORS: Dict[int, Dict[str, Spec]] = {
    # 0 정상(안정): 96~99%, 안정
    0: {
        "spo2_mean": (97.5, 1.0, 95.0, 100.0), "spo2_nadir": (96.5, 1.0, 94.0, 99.0),
        "spo2_current": (97.5, 1.0, 95.0, 100.0),"desat_rate": (0.2, 0.2, 0.0, 1.0),
        "time_below_90": (0.0, 0.0, 0.0, 0.0),  "time_below_88": (0.0, 0.0, 0.0, 0.0),
        "recovery_slope": (0.1, 0.1, 0.0, 0.5), "spo2_std": (0.5, 0.3, 0.1, 1.5),
    },
    # 1 운동: 94~98%, 일시 2~3%p 하강 가능하나 회복
    1: {
        "spo2_mean": (96.0, 1.2, 93.0, 99.0),  "spo2_nadir": (93.5, 1.5, 90.0, 97.0),
        "spo2_current": (96.0, 1.5, 92.0, 99.0),"desat_rate": (1.0, 0.6, 0.0, 3.0),
        "time_below_90": (0.02, 0.03, 0.0, 0.15),"time_below_88": (0.0, 0.0, 0.0, 0.02),
        "recovery_slope": (1.5, 0.8, 0.2, 4.0), "spo2_std": (1.2, 0.5, 0.4, 2.5),
    },
    # 2 심혈관: 정상~경미 하강(심부전 동반 시 하강 가능)
    2: {
        "spo2_mean": (95.5, 2.0, 90.0, 99.0),  "spo2_nadir": (93.0, 2.5, 86.0, 98.0),
        "spo2_current": (95.0, 2.5, 88.0, 99.0),"desat_rate": (1.0, 0.8, 0.0, 4.0),
        "time_below_90": (0.05, 0.08, 0.0, 0.4),"time_below_88": (0.02, 0.04, 0.0, 0.2),
        "recovery_slope": (0.8, 0.6, 0.0, 3.0), "spo2_std": (1.5, 0.8, 0.4, 3.5),
    },
    # 3 낙상: SpO2 정상 (낙상 자체는 SpO2 영향 적음)
    3: {
        "spo2_mean": (97.0, 1.2, 94.0, 100.0), "spo2_nadir": (95.5, 1.5, 92.0, 99.0),
        "spo2_current": (97.0, 1.5, 93.0, 100.0),"desat_rate": (0.4, 0.3, 0.0, 1.5),
        "time_below_90": (0.0, 0.01, 0.0, 0.05),"time_below_88": (0.0, 0.0, 0.0, 0.0),
        "recovery_slope": (0.3, 0.3, 0.0, 1.0), "spo2_std": (0.8, 0.4, 0.2, 2.0),
    },
    # 4 저산소: 점진 하강 nadir≤88%, desat≥4%, 90%/88% 미만 시간 큼
    4: {
        "spo2_mean": (89.0, 3.0, 80.0, 94.0),  "spo2_nadir": (84.0, 3.5, 70.0, 90.0),
        "spo2_current": (86.0, 3.5, 72.0, 93.0),"desat_rate": (5.0, 2.0, 2.0, 12.0),
        "time_below_90": (0.55, 0.25, 0.15, 1.0),"time_below_88": (0.35, 0.22, 0.05, 1.0),
        "recovery_slope": (0.5, 0.5, 0.0, 2.5), "spo2_std": (2.8, 1.2, 1.0, 6.0),
    },
}

# ---------------------------------------------------------------------------
# ECG 보조값 사전분포 (P1 출력 모사). embedding은 assembler에서 클래스별 가우시안.
# (emergency_score, reliability, hr_bpm, rhythm_regularity) + cardiac_probs 피크
# ---------------------------------------------------------------------------
# ※ emergency_score는 P1의 불완전성(AUROC=0.914)을 반영해 클래스 간 중첩 부여.
#   심혈관(2)이 높지만 깔끔히 분리되진 않음 — 운동/저산소도 일부 상승.
ECG_PRIORS: Dict[int, Dict[str, Spec]] = {
    0: {"emergency_score": (0.10, 0.08, 0.0, 0.45),"reliability": (0.12, 0.07, 0.0, 0.5),
        "hr_bpm": (72.0, 8.0, 50.0, 95.0), "rhythm_regularity": (0.93, 0.05, 0.78, 1.0)},
    1: {"emergency_score": (0.25, 0.15, 0.0, 0.65),"reliability": (0.25, 0.10, 0.05, 0.6),
        "hr_bpm": (135.0, 18.0, 100.0, 175.0),"rhythm_regularity": (0.87, 0.07, 0.65, 0.98)},
    2: {"emergency_score": (0.60, 0.22, 0.10, 0.99),"reliability": (0.18, 0.10, 0.0, 0.55),
        "hr_bpm": (118.0, 32.0, 40.0, 185.0),"rhythm_regularity": (0.55, 0.20, 0.15, 0.92)},
    3: {"emergency_score": (0.28, 0.16, 0.0, 0.70),"reliability": (0.30, 0.15, 0.05, 0.8),
        "hr_bpm": (105.0, 18.0, 75.0, 150.0),"rhythm_regularity": (0.85, 0.09, 0.55, 0.98)},
    4: {"emergency_score": (0.42, 0.20, 0.05, 0.85),"reliability": (0.22, 0.12, 0.0, 0.65),
        "hr_bpm": (115.0, 18.0, 85.0, 160.0),"rhythm_regularity": (0.80, 0.11, 0.5, 0.97)},
}

# cardiac_probs 피크 클래스 (Dirichlet 농도의 우세 인덱스)
#  0 NSR / 1 AF / 2 Ischemia / 3 Conduction / 4 Ectopic
CARDIAC_PEAK: Dict[int, int] = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}  # 심혈관만 비-NSR 피크
CARDIAC_ALPHA_BUMP = 2.5  # 피크 농도 (작을수록 클래스 간 중첩↑, P1 Macro-F1≈0.69 반영)

# 클래스별 임베딩 가우시안 평균의 분리 강도 (synthetic 신호 강도; 2단계서 실임베딩 대체)
EMB_CLASS_SEP = 0.6


def trunc_normal(rng: np.random.Generator, spec: Spec) -> float:
    mean, std, lo, hi = spec
    return float(np.clip(rng.normal(mean, std), lo, hi))


def sample_feature_vector(rng: np.random.Generator, priors: Dict[str, Spec],
                          order) -> np.ndarray:
    return np.array([trunc_normal(rng, priors[name]) for name in order],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# MVN (Multivariate Normal) IMU 샘플러 — 클래스 0·1·3 실데이터 공분산 보존
# ---------------------------------------------------------------------------
# 클래스별 실데이터 소스:
#   0 (rest)   ← imu_calibration["sit"]
#   1 (active) ← imu_calibration["active"]
#   3 (fall)   ← imu_calibration["fall"]
#   2·4        ← 실 IMU 없음 → 독립 trunc_normal fallback
# ---------------------------------------------------------------------------

_MVN_CACHE: Dict[int, tuple] = {}   # cls → (mu, L_chol, lo, hi)
_CALIB_PATH = r"D:\WidU_multimodal_fusion\interim\imu_calibration.npz"
_MVN_CLASSES = (0, 1, 3)   # 실데이터 공분산 보존 대상


def _load_mvn_params(eps: float = 1e-6) -> None:
    """imu_calibration.npz에서 클래스별 MVN 파라미터 로드 (최초 1회)."""
    import os
    if _MVN_CACHE or not os.path.exists(_CALIB_PATH):
        return
    d = np.load(_CALIB_PATH, allow_pickle=True)
    key_map = {0: "sit", 1: "active", 3: "fall"}
    for cls, key in key_map.items():
        X = d[key].astype(np.float64)           # [N, 12]
        mu = X.mean(axis=0)
        # 클립 범위 = 실데이터 3σ (물리적 이상치 방지)
        lo = mu - 3 * X.std(axis=0)
        hi = mu + 3 * X.std(axis=0)
        # 상대적 εI: 각 피처 분산의 eps배 → near-zero 분산 피처 상관 구조 보존
        cov = np.cov(X.T)
        diag_reg = np.maximum(eps * np.diag(cov), 1e-12)
        cov = cov + np.diag(diag_reg)
        L   = np.linalg.cholesky(cov)           # 하삼각 Cholesky
        _MVN_CACHE[cls] = (mu, L, lo, hi)


def sample_imu_mvn(rng: np.random.Generator, cls: int) -> np.ndarray:
    """MVN 다변량 샘플링 — 실데이터 공분산 보존.

    클래스 0·1·3: 실데이터 MVN(μ, Σ) 샘플링 (상관 구조 포함)
    클래스 2·4  : rest(0) MVN — 저활동이라 IMU로 rest와 구분 불가 (설계상 정상)
                  ECG/SpO2가 심혈관·저산소를 구분.
    """
    _load_mvn_params()
    # 2·4 → rest(0) 로 라우팅 (일관성: 모든 저활동 클래스가 동일 IMU 분포)
    src_cls = cls if cls in _MVN_CLASSES else 0
    if src_cls not in _MVN_CACHE:
        return sample_feature_vector(rng, IMU_PRIORS[cls], IMU_FEATURES)

    mu, L, lo, hi = _MVN_CACHE[src_cls]
    z = rng.standard_normal(len(mu))
    x = mu + L @ z
    x = np.clip(x, lo, hi).astype(np.float32)
    return x


# ---------------------------------------------------------------------------
# Bootstrap IMU 샘플러 — 실벡터 직접 리샘플 + 지터 (가우시안 가정 없음)
# ---------------------------------------------------------------------------
_BOOTSTRAP_CACHE: Dict[int, np.ndarray] = {}  # cls → real vectors [N, 12]
_BOOTSTRAP_STD: Dict[int, np.ndarray] = {}    # cls → per-feature std (지터 스케일)


def _load_bootstrap_cache() -> None:
    """imu_calibration.npz에서 실벡터 로드 (최초 1회)."""
    import os
    if _BOOTSTRAP_CACHE or not os.path.exists(_CALIB_PATH):
        return
    d = np.load(_CALIB_PATH, allow_pickle=True)
    key_map = {0: "sit", 1: "active", 3: "fall"}
    for cls, key in key_map.items():
        X = d[key].astype(np.float32)
        _BOOTSTRAP_CACHE[cls] = X
        _BOOTSTRAP_STD[cls] = X.std(axis=0)


def sample_imu_bootstrap(rng: np.random.Generator, cls: int,
                          jitter_frac: float = 0.05) -> np.ndarray:
    """Bootstrap 리샘플링 — 실벡터에서 직접 추출 + 소량 지터.

    jitter_frac: 각 피처 실측 std의 배수로 가우시안 지터 (기본 5%)
    클래스 2·4 → rest(0) 벡터 사용 (MVN과 동일 원칙)
    """
    _load_bootstrap_cache()
    src_cls = cls if cls in _BOOTSTRAP_CACHE else 0
    if src_cls not in _BOOTSTRAP_CACHE:
        return sample_feature_vector(rng, IMU_PRIORS[cls], IMU_FEATURES)

    X   = _BOOTSTRAP_CACHE[src_cls]
    idx = int(rng.integers(0, len(X)))
    vec = X[idx].copy()
    if jitter_frac > 0:
        std = _BOOTSTRAP_STD[src_cls]
        vec = vec + rng.normal(0.0, jitter_frac * std).astype(np.float32)
    return vec


def sample_imu(rng, cls: int, mode: str = "mvn") -> np.ndarray:
    """IMU 샘플링 진입점.

    mode:
      "indep"     — 독립 trunc_normal (v1, 원본)
      "mvn"       — 다변량 가우시안, 공분산 보존 (v2, 기본)
      "bootstrap" — 실벡터 리샘플 + 지터, 가우시안 가정 없음 (v3)
    """
    if mode == "indep":
        return sample_feature_vector(rng, IMU_PRIORS[cls], IMU_FEATURES)
    if mode == "bootstrap":
        return sample_imu_bootstrap(rng, cls)
    # default: mvn
    return sample_imu_mvn(rng, cls)


def sample_spo2(rng, cls: int) -> np.ndarray:
    return sample_feature_vector(rng, SPO2_PRIORS[cls], SPO2_FEATURES)


def sample_cardiac_probs(rng, cls: int) -> np.ndarray:
    """Dirichlet로 cardiac_probs[5] 생성 — 우세 인덱스에 농도 집중."""
    alpha = np.ones(NUM_CARDIAC) * 0.8
    alpha[CARDIAC_PEAK[cls]] += CARDIAC_ALPHA_BUMP
    return rng.dirichlet(alpha).astype(np.float32)


def gate_tier_from_reliability(reliability: float) -> int:
    if reliability < T_MASK:
        return 0  # use
    if reliability < T_ALERT:
        return 1  # mask
    return 2      # alert
