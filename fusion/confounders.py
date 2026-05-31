"""실데이터 앵커 confounder 셋 — 비순환(no circular synthesis).

합성 confounder를 만들면 "내가 심은 규칙을 모델이 배웠나"의 동어반복이 된다. 따라서 각
confounder의 *정의 모달리티*는 실 신호이고, 맥락 모달리티만 정직한 정상 앵커로 채운다.

| confounder   | 정의 모달(실)                         | 맥락 앵커            | 오발화 대상 |
|--------------|--------------------------------------|---------------------|-------------|
| chronic_ecg  | 실 CPSC Ectopic(label_mc 4, 미학습)   | 실 rest IMU + 정상 SpO2 | cardiac(2)  |
| apnea_spo2   | 실 ucddb 무호흡 desat(nadir·drop·HR)  | rest ECG·IMU(HR 비상승) | hypoxia(4)  |
| motion_ecg   | 실 NSTDB 모션노이즈 ECG(reliability↑) | rest IMU + 정상 SpO2 | cardiac(2)  |

비순환 근거:
  chronic — Ectopic은 proxy 조립에 한 번도 안 쓰임(_ECG_SRC_LABELS) → 모델이 본 적 없는 실 비정상.
            전도(Conduction)는 proxy에서 cardiac로 학습돼 오염 → chronic 앵커에서 제외.
  apnea   — 실 무호흡 desat은 보상성 빈맥을 거의 안 동반(HR>100 0.2%) → SpO2↓∧HR비상승 joint.
  motion  — P1 reliability 헤드가 모션을 표시 → ECG 임베딩은 실 NSTDB 노이즈로 손상. (P1 추론 필요)
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fusion.schema import (CARDIAC_PROB_NAMES, IMU_FEATURES, NUM_CARDIAC,
                           SPO2_FEATURES)

# 외부 데이터 루트 (환경변수 P2_DATA_DIR로 지정, 기본은 상대경로 "data"). 절대경로 미노출.
_DATA_ROOT    = Path(os.environ.get("P2_DATA_DIR", "data"))
P1_CACHE_DIR  = _DATA_ROOT / "p1_cache"
IMU_CALIB     = _DATA_ROOT / "interim" / "imu_calibration.npz"
UCDDB_DIR     = _DATA_ROOT / "raw" / "ucddb"

# P2 클래스 인덱스
CARDIAC_CLASS, HYPOXIA_CLASS, NORMAL_CLASS = 2, 4, 0


@dataclass
class ConfounderSet:
    """실데이터 앵커 confounder 평가 셋.

    arrays: {ecg_embedding[N,768], ecg_aux[N,10], imu_feat[N,12], spo2_feat[N,8],
             modality_mask[N,3]} — P2Dataset 배치와 동일 스키마.
    emergency_class: 이 셋에서 예측되면 오발화(FP)인 응급 클래스.
    true_label: benign 진실 클래스 (보통 0).
    realness: 어느 모달이 실/앵커인지 기록 (정직성).
    """
    name: str
    arrays: Dict[str, np.ndarray]
    emergency_class: int
    true_label: int = NORMAL_CLASS
    realness: str = ""

    @property
    def n(self) -> int:
        return len(self.arrays["ecg_embedding"])


# ─────────────────────────────────────────────────────────────────────────────
# 맥락 앵커 (실 IMU / 정상 SpO2)
# ─────────────────────────────────────────────────────────────────────────────
def _normal_spo2_feat() -> np.ndarray:
    """건강 정상 SpO2 피처 [8] (결정적 앵커, 사전분포 무작위 추출 아님)."""
    # order = SPO2_FEATURES: mean,nadir,current,desat_rate,tb90,tb88,recovery,std
    return np.array([97.5, 96.5, 97.5, 0.2, 0.0, 0.0, 0.1, 0.5], dtype=np.float32)


def _sample_real_imu(n: int, rng: np.random.Generator, context: str = "sit") -> np.ndarray:
    """실 IMU 피처 벡터 [n,12] 를 캘리브레이션에서 추출 (sit=rest / active / fall)."""
    if not IMU_CALIB.exists():
        raise FileNotFoundError(f"IMU 캘리브레이션 없음: {IMU_CALIB}")
    d = np.load(IMU_CALIB, allow_pickle=True)
    pool = d[context].astype(np.float32)            # [M,12]
    idx = rng.integers(0, len(pool), size=n)
    return pool[idx]


def _ecg_aux_from_cache(c: Dict[str, np.ndarray], idx: np.ndarray,
                        hr_override: Optional[np.ndarray] = None) -> np.ndarray:
    """캐시 필드 → ecg_aux [n,10] (schema.flat_ecg_aux 순서).

    [cardiac_probs×5, emergency_score, reliability, gate_tier(float), hr_bpm, rhythm_regularity]
    """
    cp   = c["cardiac_probs"][idx].astype(np.float32)              # [n,5]
    es   = c["emergency_score"][idx].astype(np.float32)
    rel  = c["reliability"][idx].astype(np.float32)
    gt   = c["gate_tier"][idx].astype(np.float32)
    hr   = c["hr_bpm"][idx].astype(np.float32) if hr_override is None else hr_override.astype(np.float32)
    rr   = c["rhythm_regularity"][idx].astype(np.float32)
    aux  = np.concatenate([cp, np.stack([es, rel, gt, hr, rr], axis=1)], axis=1)
    return aux.astype(np.float32)


def _load_cache(splits=("test",)) -> Dict[str, np.ndarray]:
    """P1 캐시 병합 로드."""
    keys = ["embedding", "cardiac_probs", "emergency_score", "reliability",
            "gate_tier", "hr_bpm", "rhythm_regularity", "label_mc"]
    pools = {k: [] for k in keys}
    for sp in splits:
        d = np.load(P1_CACHE_DIR / f"cpsc_mc_{sp}.npz")
        for k in keys:
            pools[k].append(d[k])
    return {k: np.concatenate(v) for k, v in pools.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1) chronic_ecg — 실 CPSC Ectopic (비정상이나 비응급 ECG)
# ─────────────────────────────────────────────────────────────────────────────
def build_chronic_ecg_confounder(splits=("train", "val", "test"),
                                 seed: int = 0,
                                 ecg_label: int = 4) -> ConfounderSet:
    """실 Ectopic(label_mc 4) ECG + 실 rest IMU + 정상 SpO2.

    Ectopic은 proxy 조립에 미사용 → 비순환. emergency_score는 낮으나 cardiac_probs는 비-NSR →
    임베딩/probs만 보는 모델은 cardiac(2) 오발화, emergency_score를 존중하는 모델은 정상.
    """
    rng = np.random.default_rng(seed)
    c = _load_cache(splits)
    idx = np.where(c["label_mc"] == ecg_label)[0]
    n = len(idx)
    arrays = {
        "ecg_embedding": c["embedding"][idx].astype(np.float32),
        "ecg_aux":       _ecg_aux_from_cache(c, idx),
        "imu_feat":      _sample_real_imu(n, rng, context="sit"),
        "spo2_feat":     np.tile(_normal_spo2_feat(), (n, 1)),
        "modality_mask": np.ones((n, 3), dtype=np.float32),
    }
    es_mean = float(c["emergency_score"][idx].mean())
    return ConfounderSet(
        name="chronic_ecg", arrays=arrays, emergency_class=CARDIAC_CLASS,
        realness=f"ECG=real CPSC Ectopic(n={n}, es̄={es_mean:.2f}); IMU=real sit; SpO2=normal-anchor",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2) apnea_spo2 — 실 ucddb 무호흡 desaturation (HR 비상승)
# ─────────────────────────────────────────────────────────────────────────────
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}")


def _parse_respevt(path: Path) -> List[Dict[str, float]]:
    """respevt.txt → [{nadir, drop, hr}] (실 무호흡 desat 이벤트).

    컬럼(공백정렬): Time Type [PB/CS] Duration Low %Drop Snore Arousal Rate Change.
    결측 허용 — nadir(Low, SpO2%) 있는 이벤트만 채택.
    """
    events: List[Dict[str, float]] = []
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return events
    for ln in lines:
        if not _TIME_RE.match(ln.strip()):
            continue
        toks = ln.split()
        # 숫자 토큰만 추출 (부호 포함)
        nums = []
        for t in toks[1:]:
            try:
                nums.append(float(t))
            except ValueError:
                continue
        # nadir = SpO2 범위 [70,100) 의 첫 값
        nadir = next((x for x in nums if 70.0 <= x < 100.0), None)
        if nadir is None:
            continue
        rest = nums[nums.index(nadir) + 1:]
        drop = next((x for x in rest if 0.5 <= x <= 30.0), 3.0)   # %Drop, 결측시 보수값
        hr   = next((x for x in rest if 40.0 <= x <= 140.0), 70.0)  # Rate(HR), 결측시 관측 평균(~70)
        events.append({"nadir": float(nadir), "drop": float(drop), "hr": float(hr)})
    return events


def _spo2_feat_from_event(nadir: float, drop: float) -> np.ndarray:
    """무호흡 이벤트 통계 → SpO2 피처 [8] (반-실: 실 nadir·drop, 형태는 모델링).

    급성 hypoxia와 달리 무호흡은 짧고 자가회복 → current≈baseline, 회복 기울기 큼.
    """
    baseline = min(nadir + drop, 100.0)
    return np.array([
        baseline - 0.4 * drop,                       # mean (윈도우 평균은 baseline서 약간 하강)
        nadir,                                        # nadir (실)
        baseline - 0.1 * drop,                        # current (거의 회복)
        max(drop * 2.0, 0.5),                         # desat_rate (%p/분; 무호흡 ~30s)
        0.30 if nadir < 90 else 0.0,                  # time_below_90
        0.15 if nadir < 88 else 0.0,                  # time_below_88
        max(drop * 3.0, 0.5),                         # recovery_slope (빠른 회복)
        0.4 * drop + 0.5,                             # std
    ], dtype=np.float32)


def build_apnea_confounder(ucddb_dir: Path = UCDDB_DIR, seed: int = 0,
                           max_events: Optional[int] = None) -> ConfounderSet:
    """실 ucddb 무호흡 desat + rest ECG(실 HR, 비상승) + rest IMU.

    SpO2 desat은 실(nadir·drop), HR은 실 이벤트값(비상승) → joint "SpO2↓∧HR정상" 검증.
    ECG 임베딩은 NSR rest 캐시에서, hr_bpm만 실 무호흡 HR로 교체.
    """
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(str(ucddb_dir / "*_respevt.txt")))
    events: List[Dict[str, float]] = []
    for f in files:
        events.extend(_parse_respevt(Path(f)))
    if not events:
        raise RuntimeError(f"ucddb 무호흡 이벤트 파싱 실패: {ucddb_dir}")
    if max_events:
        events = events[:max_events]
    n = len(events)

    nadir = np.array([e["nadir"] for e in events], dtype=np.float32)
    drop  = np.array([e["drop"]  for e in events], dtype=np.float32)
    hr    = np.array([e["hr"]    for e in events], dtype=np.float32)

    spo2 = np.stack([_spo2_feat_from_event(nd, dr) for nd, dr in zip(nadir, drop)])

    # ECG 맥락: NSR rest 캐시(낮은 emergency_score) + hr_bpm을 실 무호흡 HR로 교체
    c = _load_cache(("train", "val", "test"))
    nsr_idx = np.where(c["label_mc"] == 0)[0]
    pick = nsr_idx[rng.integers(0, len(nsr_idx), size=n)]
    arrays = {
        "ecg_embedding": c["embedding"][pick].astype(np.float32),
        "ecg_aux":       _ecg_aux_from_cache(c, pick, hr_override=hr),
        "imu_feat":      _sample_real_imu(n, rng, context="sit"),   # 수면=정지
        "spo2_feat":     spo2.astype(np.float32),
        "modality_mask": np.ones((n, 3), dtype=np.float32),
    }
    return ConfounderSet(
        name="apnea_spo2", arrays=arrays, emergency_class=HYPOXIA_CLASS,
        realness=(f"SpO2=real ucddb desat(n={n}, nadir̄={nadir.mean():.1f}, "
                  f"HR̄={hr.mean():.0f} 비상승); ECG=NSR rest; IMU=real sit"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3) motion_ecg — 실 NSTDB 모션노이즈 ECG (P1 추론 필요)
# ─────────────────────────────────────────────────────────────────────────────
NSTDB_P1_CACHE = P1_CACHE_DIR / "nstdb_motion.npz"


def build_motion_confounder(seed: int = 0) -> ConfounderSet:
    """실 NSTDB 모션 손상 ECG → P1 임베딩 + 정상 맥락. cardiac(2) 오발화 측정.

    P1(fairseq) 추론으로 (CPSC NSR + NSTDB 노이즈)@SNR 의 임베딩·점수를 미리 캐시해야 한다.
    캐시 부재 시 명확한 안내와 함께 중단 — chronic/apnea는 이 의존성 없이 동작.
    """
    if not NSTDB_P1_CACHE.exists():
        raise NotImplementedError(
            f"motion confounder는 P1 추론 캐시가 필요: {NSTDB_P1_CACHE}\n"
            "  생성: CPSC NSR 신호 + NSTDB(em/ma) 노이즈를 SNR로 합성 → P1CardiacChannel.infer →\n"
            "  embedding/cardiac_probs/emergency_score/reliability/... 저장 (별도 P1 추론 패스).\n"
            "  reliability 단독으로도 분리되는 confounder라 P1 레벨에서 이미 분리 검증됨."
        )
    rng = np.random.default_rng(seed)
    d = np.load(NSTDB_P1_CACHE)
    n = len(d["embedding"])
    c = {k: d[k] for k in d.files}
    arrays = {
        "ecg_embedding": c["embedding"].astype(np.float32),
        "ecg_aux":       _ecg_aux_from_cache(c, np.arange(n)),
        "imu_feat":      _sample_real_imu(n, rng, context="sit"),
        "spo2_feat":     np.tile(_normal_spo2_feat(), (n, 1)),
        "modality_mask": np.ones((n, 3), dtype=np.float32),
    }
    return ConfounderSet(
        name="motion_ecg", arrays=arrays, emergency_class=CARDIAC_CLASS,
        realness=f"ECG=real NSTDB-noise@P1(n={n}); IMU=real sit; SpO2=normal-anchor",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 실 페어드(ECG+IMU) 평가. ptt_ppg 운동: 유일 실 시간정렬쌍.
# ─────────────────────────────────────────────────────────────────────────────
PTT_P1_CACHE = P1_CACHE_DIR / "ptt_ppg_p1.npz"
ACTIVE_CLASS = 1   # normal-active (운동, 진실=비응급)


def build_exercise_confounder(seed: int = 0) -> ConfounderSet:
    """실 ptt_ppg 운동(walk/run) ECG+IMU 동일인·시간정렬 페어 + 정상 SpO2 앵커.

    운동은 ECG emergency_score↑(es̄~0.41)·reliability↑(모션)·HR↑를 유발 → 융합이 맥락
    (활동 IMU·정상 SpO2·reliability)으로 응급 오발화(2/3/4)를 억제하는지 측정(진실=normal-active).
    유일 실 ECG+IMU 정렬쌍이라 실 페어드 평가의 핵심. ECG는 CPSC 분포 정합 후 P1 추론(검증 게이트 통과).
    """
    if not PTT_P1_CACHE.exists():
        raise NotImplementedError(
            f"ptt P1 캐시 필요: {PTT_P1_CACHE} (scripts/build_ptt_ppg_p1_cache.py 먼저 실행)")
    rng = np.random.default_rng(seed)
    d = np.load(PTT_P1_CACHE)
    c = {k: d[k] for k in d.files}
    idx = np.where(c["label"] == ACTIVE_CLASS)[0]   # walk/run
    n = len(idx)
    arrays = {
        "ecg_embedding": c["embedding"][idx].astype(np.float32),
        "ecg_aux":       _ecg_aux_from_cache(c, idx),
        "imu_feat":      c["imu_feat"][idx].astype(np.float32),   # 실 ptt IMU (동일 윈도우 페어)
        "spo2_feat":     np.tile(_normal_spo2_feat(), (n, 1)),
        "modality_mask": np.ones((n, 3), dtype=np.float32),
    }
    es_m = float(c["emergency_score"][idx].mean()); rel_m = float(c["reliability"][idx].mean())
    return ConfounderSet(
        name="exercise_pair", arrays=arrays,
        emergency_class=(CARDIAC_CLASS, 3, HYPOXIA_CLASS),   # 임의 응급(2/3/4) 오발화
        true_label=ACTIVE_CLASS,
        realness=(f"ECG+IMU=real ptt_ppg walk/run pair(n={n}, es̄={es_m:.2f}, rel̄={rel_m:.2f}); "
                  f"SpO2=normal-anchor"),
    )


# ─────────────────────────────────────────────────────────────────────────────
def build_all(seed: int = 0, include_motion: bool = True) -> List[ConfounderSet]:
    """가용한 모든 실 confounder 셋. motion은 P1 캐시 있을 때만."""
    sets = [build_chronic_ecg_confounder(seed=seed),
            build_apnea_confounder(seed=seed)]
    if include_motion:
        try:
            sets.append(build_motion_confounder(seed=seed))
        except NotImplementedError as e:
            print(f"[skip motion_ecg] {str(e).splitlines()[0]}")
    return sets
