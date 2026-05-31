"""실데이터 raw 윈도우 → 핸드크래프트 피처 추출.

IMU: raw 가속도(3축)+자이로(3축) 윈도우 [T,6] → schema.IMU_FEATURES 순서 [12]
SpO2: raw SpO2 시계열 [T] → schema.SPO2_FEATURES 순서 [8]

단위 가정:
  가속도 g (1g≈9.81 m/s², 정지 시 SMV≈1.0), 자이로 rad/s, 샘플링레이트 fs(Hz).
proxy 조립과 실데이터(2단계·필드)가 동일 피처 정의를 공유한다.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy as scipy_entropy

from fusion.schema import IMU_FEATURES, SPO2_FEATURES

__all__ = ["extract_imu_features", "window_to_imu_feat", "extract_spo2_features"]

_G = 9.81  # m/s²


# ─────────────────────────────────────────────────────────────────────────────
# IMU
# ─────────────────────────────────────────────────────────────────────────────
def _smv(accel: np.ndarray) -> np.ndarray:
    """Signal Magnitude Vector: sqrt(ax²+ay²+az²)"""
    return np.sqrt((accel ** 2).sum(axis=1))


def _dominant_freq(smv: np.ndarray, fs: float) -> float:
    """FFT로 지배 주파수(Hz) 추출."""
    n = len(smv)
    if n < 4:
        return 0.0
    win = smv - smv.mean()
    fft = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if len(fft) < 2:  # DC 제외
        return 0.0
    idx = np.argmax(fft[1:]) + 1
    return float(freqs[idx])


def _spectral_entropy(smv: np.ndarray) -> float:
    """스펙트럴 엔트로피 (0=규칙, 1=무작위)."""
    n = len(smv)
    if n < 4:
        return 1.0
    fft_mag = np.abs(np.fft.rfft(smv - smv.mean())) ** 2
    total = fft_mag.sum()
    if total < 1e-12:
        return 1.0
    p = fft_mag / total
    p = p[p > 0]
    return float(scipy_entropy(p) / np.log(len(p) + 1e-12))


def _tilt_change(accel: np.ndarray) -> float:
    """중력벡터 기준 자세각(tilt) 변화량(°). tilt=arccos(az/smv)의 최대-최소."""
    az = accel[:, 2]
    smv = _smv(accel)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_tilt = np.clip(az / (smv + 1e-8), -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_tilt))
    return float(tilt_deg.max() - tilt_deg.min())


def extract_imu_features(accel: np.ndarray, gyro: np.ndarray,
                         fs: float = 200.0) -> np.ndarray:
    """
    Args:
        accel: [T,3] (ax,ay,az) in g
        gyro:  [T,3] (gx,gy,gz) in rad/s
        fs:    sampling rate (Hz)
    Returns:
        feat: [12] float32, 순서 = schema.IMU_FEATURES
    """
    assert accel.shape[1] == 3 and gyro.shape[1] == 3

    smv = _smv(accel)                            # [T]
    gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))  # [T]

    jerk = np.diff(smv, prepend=smv[0]) * fs     # 가속도 1차 미분
    jerk_peak = float(np.abs(jerk).max())

    # impact peak detection (SMV > 2g 초과 횟수)
    peaks, _ = find_peaks(smv, height=2.0, distance=max(1, int(fs * 0.1)))
    impact_count = float(len(peaks))

    gyro_energy = float(gyro_mag.mean() * len(gyro_mag) / fs)  # ∫|ω|dt 근사

    feat = np.array([
        smv.mean(),               # 0 smv_mean
        smv.std(),                # 1 smv_std
        smv.max(),                # 2 smv_peak
        smv.min(),                # 3 smv_min
        jerk_peak,                # 4 jerk_peak
        gyro_mag.max(),           # 5 gyro_peak
        gyro_energy,              # 6 gyro_energy
        _tilt_change(accel),      # 7 tilt_change
        float(smv.var()),         # 8 act_energy
        _dominant_freq(smv, fs),  # 9 dom_freq
        _spectral_entropy(smv),   # 10 spec_entropy
        impact_count,             # 11 impact_count
    ], dtype=np.float32)

    assert len(feat) == len(IMU_FEATURES), f"{len(feat)} != {len(IMU_FEATURES)}"
    return feat


def window_to_imu_feat(data: np.ndarray, fs: float = 200.0,
                       accel_unit: str = "g") -> np.ndarray:
    """raw 윈도우 [T,6]=[ax,ay,az,gx,gy,gz] → IMU 피처 [12].

    accel_unit: "g"(이미 g단위) | "ms2"(m/s² → g 변환)
    """
    accel = data[:, :3].copy()
    gyro  = data[:, 3:].copy()
    if accel_unit == "ms2":
        accel /= _G
    return extract_imu_features(accel, gyro, fs=fs)


# ─────────────────────────────────────────────────────────────────────────────
# SpO2
# ─────────────────────────────────────────────────────────────────────────────
def extract_spo2_features(spo2: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Args:
        spo2: [T] SpO2 (%) 시계열
        fs:   sampling rate (Hz). 분 단위 변환에 사용.
    Returns:
        feat: [8] float32, 순서 = schema.SPO2_FEATURES
    """
    spo2 = spo2.astype(np.float32)
    n = len(spo2)

    mean_val = float(spo2.mean())
    nadir    = float(spo2.min())
    current  = float(spo2[-1])
    std_val  = float(spo2.std())

    # desaturation rate (%p/분): 최대 하강 (슬라이딩 윈도우 60s)
    win_samples = max(1, int(fs * 60))
    if n > win_samples:
        drops = []
        for i in range(0, n - win_samples, max(1, win_samples // 10)):
            drops.append(spo2[i] - spo2[i:i + win_samples].min())
        desat_rate = float(max(drops)) if drops else 0.0
    else:
        desat_rate = float(max(spo2[0] - nadir, 0.0))

    time_below_90 = float((spo2 < 90.0).mean())
    time_below_88 = float((spo2 < 88.0).mean())

    # recovery slope: 최저점 이후 상승 기울기 (%p/분)
    nadir_idx = int(np.argmin(spo2))
    post = spo2[nadir_idx:]
    if len(post) > 1:
        end_val = float(post[-1])
        duration_min = len(post) / (fs * 60 + 1e-8)
        recovery_slope = max(0.0, (end_val - nadir) / (duration_min + 1e-8))
    else:
        recovery_slope = 0.0

    feat = np.array([
        mean_val,        # 0 spo2_mean
        nadir,           # 1 spo2_nadir
        current,         # 2 spo2_current
        desat_rate,      # 3 desat_rate
        time_below_90,   # 4 time_below_90
        time_below_88,   # 5 time_below_88
        recovery_slope,  # 6 recovery_slope
        std_val,         # 7 spo2_std
    ], dtype=np.float32)

    assert len(feat) == len(SPO2_FEATURES), f"{len(feat)} != {len(SPO2_FEATURES)}"
    return feat
