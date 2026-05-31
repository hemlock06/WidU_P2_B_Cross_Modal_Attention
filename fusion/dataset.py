"""멀티모달 조립 + 데이터셋 — proxy/실데이터 공통.

조립(방법 A, 클래스 조건부): 클래스 라벨이 주어지면 각 모달리티를 해당 클래스 분포에서
독립 추출해 paired 샘플로 조립(conditional-independence 가정).
  ECG  : P1 캐시(실 ECG-FM 임베딩 + 점수)에서 샘플링 — P2는 재인코딩 안 함.
  IMU  : 실데이터 캘리브레이션 MVN/bootstrap (클래스 0·1·3) 또는 문헌 사전분포.
  SpO2 : 사전분포 (저샘플링·서서히 변하는 신호).

★ 조건부 독립 조립의 한계: ECG·IMU·SpO2가 무관한 기록에서 짝지어져 시간 대응이 부재 —
  cross-modal attention의 진짜 우위(joint modeling)는 실 정렬쌍에서만 검증된다. 따라서
  confounder 평가는 합성이 아니라 실데이터에 앵커한다 (fusion/confounders.py 참조).

클래스별 ECG 소스:
  0 정상·1 운동·3 낙상·4 저산소 → CPSC NSR(label_mc 0) → ECG 정상
  2 심혈관 → CPSC AF+Ischemia+Conduction(label_mc 1,2,3) → ECG 비정상
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from fusion.schema import (EMB_DIM, IMU_DIM, IMU_FEATURES, MultimodalSample,
                           NUM_CLASSES, SPO2_DIM, SPO2_FEATURES)
from fusion import priors as cp

# 외부 데이터 경로 (코드 미복사 — 절대경로 참조). 환경변수로 재지정 가능.
P1_CACHE_DIR = Path(r"D:\WidU_multimodal_fusion\p1_cache")

# 클래스별 ECG P1 캐시 소스 매핑 — 키: P2 클래스, 값: cpsc_mc label_mc 값 목록
_ECG_SRC_LABELS: Dict[int, List[int]] = {
    0: [0],        # 정상(안정) → NSR
    1: [0],        # 정상(운동) → NSR (IMU로 구분)
    2: [1, 2, 3],  # 심혈관 응급 → AF + Ischemia + Conduction
    3: [0],        # 낙상 → NSR (충격은 IMU로 구분)
    4: [0],        # 저산소 → NSR (보상성 빈맥은 사전분포 hr_bpm으로 반영)
}

# 혼동쌍: 일부 hard case에서 상대 클래스 분포 사용
PARTNERS = {0: [2], 1: [2, 3], 2: [1, 4], 3: [1], 4: [2]}


def _std_vec(priors, order) -> np.ndarray:
    return np.array([priors[name][1] for name in order], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# P1 캐시 — 클래스별 실 ECG 샘플 공급
# ─────────────────────────────────────────────────────────────────────────────
class P1Cache:
    """P1 실출력 캐시에서 클래스별 ECG 샘플을 공급.

    ECG 누출 방지:
      - P2 train/val 생성: P1 train+val 풀 (splits=["train","val"])
      - P2 test 생성:      P1 test 풀만  (splits=["test"])
      → P2 train/test가 서로 다른 CPSC 레코드 풀 → 임베딩 누출 0%.
    """

    def __init__(self, cache_dir: Path = P1_CACHE_DIR, splits: List[str] = None):
        if splits is None:
            splits = ["train", "val"]

        pools: Dict[str, List] = {
            k: [] for k in ["embedding", "cardiac_probs", "emergency_score",
                            "reliability", "gate_tier", "hr_bpm", "rhythm_regularity"]
        }
        self._label_pool: List[np.ndarray] = []

        for split in splits:
            p = cache_dir / f"cpsc_mc_{split}.npz"
            if not p.exists():
                raise FileNotFoundError(
                    f"P1 캐시 없음: {p}\nP1 임베딩 캐시를 먼저 생성하세요."
                )
            d = np.load(p)
            for k in pools:
                pools[k].append(d[k])
            self._label_pool.append(d["label_mc"])

        self._data = {k: np.concatenate(v) for k, v in pools.items()}
        self._labels = np.concatenate(self._label_pool)
        self._splits = splits

        self._cls_idx: Dict[int, np.ndarray] = {}
        for src_labels in set(tuple(v) for v in _ECG_SRC_LABELS.values()):
            for sl in src_labels:
                if sl not in self._cls_idx:
                    self._cls_idx[sl] = np.where(self._labels == sl)[0]

    def sample(self, rng: np.random.Generator, p2_cls: int) -> dict:
        """p2_cls에 해당하는 CPSC 레코드 중 하나를 무작위 추출."""
        src_labels = _ECG_SRC_LABELS[p2_cls]
        chosen_label = int(rng.choice(src_labels))
        idx_pool = self._cls_idx[chosen_label]
        idx = int(rng.choice(idx_pool))
        return {k: self._data[k][idx] for k in self._data}


# ─────────────────────────────────────────────────────────────────────────────
# 클래스 조건부 조립기
# ─────────────────────────────────────────────────────────────────────────────
class ConditionalAssembler:
    """클래스 조건부 조립기.

    noise_scale: IMU/SpO2 피처 측정 노이즈 (prior std 배수). imu_mode=indep 전용.
    hard_frac:   모호 사례 비율 — IMU/SpO2 일부를 상대 클래스 분포에서 추출.
    p1_cache:    P1Cache 인스턴스. None이면 합성 가우시안 fallback.
    imu_mode:    "indep"|"mvn"|"bootstrap" (mvn=기본, 실데이터 공분산 보존)
    """

    def __init__(self, seed: int = 42, emb_dim: int = EMB_DIM,
                 noise_scale: float = 0.35, hard_frac: float = 0.12,
                 p1_cache: Optional[P1Cache] = None, imu_mode: str = "mvn"):
        self.rng = np.random.default_rng(seed)
        self.emb_dim = emb_dim
        self.noise_scale = noise_scale
        self.hard_frac = hard_frac
        self.p1_cache = p1_cache
        self.imu_mode = imu_mode

        # fallback: 합성 가우시안 임베딩 (캐시 없을 때)
        emb_rng = np.random.default_rng(seed + 1000)
        self._emb_means = emb_rng.normal(
            0.0, cp.EMB_CLASS_SEP, size=(NUM_CLASSES, emb_dim)
        ).astype(np.float32)

    def _sample_ecg(self, cls: int) -> dict:
        if self.p1_cache is not None:
            return self.p1_cache.sample(self.rng, cls)
        ecg_prior = cp.ECG_PRIORS[cls]
        reliability = cp.trunc_normal(self.rng, ecg_prior["reliability"])
        emb = (self._emb_means[cls]
               + self.rng.normal(0.0, 1.0, size=self.emb_dim)).astype(np.float32)
        return {
            "embedding": emb,
            "cardiac_probs": cp.sample_cardiac_probs(self.rng, cls),
            "emergency_score": np.float32(cp.trunc_normal(self.rng, ecg_prior["emergency_score"])),
            "reliability": np.float32(reliability),
            "gate_tier": np.int8(cp.gate_tier_from_reliability(reliability)),
            "hr_bpm": np.float32(cp.trunc_normal(self.rng, ecg_prior["hr_bpm"])),
            "rhythm_regularity": np.float32(cp.trunc_normal(self.rng, ecg_prior["rhythm_regularity"])),
        }

    def _noisy(self, vec: np.ndarray, priors, order) -> np.ndarray:
        if self.noise_scale <= 0:
            return vec
        return (vec + self.rng.normal(
            0.0, self.noise_scale * _std_vec(priors, order)
        )).astype(np.float32)

    def assemble_one(self, cls: int) -> MultimodalSample:
        ecg = self._sample_ecg(cls)
        ecg_tag = "real_p1" if self.p1_cache else "synth_prior"

        imu_cls = spo2_cls = cls
        imu_tag = spo2_tag = ecg_tag.replace("real_p1", "synth_prior")
        if self.rng.random() < self.hard_frac:
            partner = int(self.rng.choice(PARTNERS[cls]))
            if self.rng.random() < 0.5:
                imu_cls = partner; imu_tag = "synth_hard"
            if self.rng.random() < 0.5:
                spo2_cls = partner; spo2_tag = "synth_hard"

        raw_imu = cp.sample_imu(self.rng, imu_cls, mode=self.imu_mode)
        if self.imu_mode == "indep":
            imu = self._noisy(raw_imu, cp.IMU_PRIORS[imu_cls], IMU_FEATURES)
        else:
            imu = raw_imu.astype(np.float32)
        spo2 = self._noisy(cp.sample_spo2(self.rng, spo2_cls),
                           cp.SPO2_PRIORS[spo2_cls], SPO2_FEATURES)

        return MultimodalSample(
            ecg_embedding=ecg["embedding"].astype(np.float32),
            cardiac_probs=ecg["cardiac_probs"].astype(np.float32),
            emergency_score=float(ecg["emergency_score"]),
            reliability=float(ecg["reliability"]),
            gate_tier=int(ecg["gate_tier"]),
            hr_bpm=float(ecg["hr_bpm"]),
            rhythm_regularity=float(ecg["rhythm_regularity"]),
            imu_feat=imu,
            spo2_feat=spo2,
            label=cls,
            modality_mask=np.ones(3, dtype=np.float32),
            src={"ecg": ecg_tag, "imu": imu_tag, "spo2": spo2_tag},
        )

    def assemble_balanced(self, n_per_class: int) -> List[MultimodalSample]:
        out: List[MultimodalSample] = []
        for cls in range(NUM_CLASSES):
            out.extend(self.assemble_one(cls) for _ in range(n_per_class))
        self.rng.shuffle(out)
        return out

    def assemble(self, counts: Optional[List[int]] = None,
                 n_per_class: int = 2000) -> List[MultimodalSample]:
        if counts is None:
            return self.assemble_balanced(n_per_class)
        out: List[MultimodalSample] = []
        for cls, n in enumerate(counts):
            out.extend(self.assemble_one(cls) for _ in range(n))
        self.rng.shuffle(out)
        return out


def samples_to_arrays(samples: List[MultimodalSample]) -> dict:
    emb     = np.stack([s.ecg_embedding for s in samples])
    ecg_aux = np.stack([s.flat_ecg_aux() for s in samples])
    imu     = np.stack([s.imu_feat for s in samples])
    spo2    = np.stack([s.spo2_feat for s in samples])
    mask    = np.stack([s.modality_mask for s in samples])
    y       = np.array([s.label for s in samples], dtype=np.int64)
    return {
        "ecg_embedding": emb.astype(np.float32),
        "ecg_aux": ecg_aux.astype(np.float32),
        "imu_feat": imu.astype(np.float32),
        "spo2_feat": spo2.astype(np.float32),
        "modality_mask": mask.astype(np.float32),
        "label": y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class P2Dataset(Dataset):
    """조립된 .npz 를 읽는 Dataset.

    modality_dropout_p: 학습 시 각 모달리티를 이 확률로 독립 0-마스킹 (val/test=0).
    """

    ECG_AUX_DIM = 10

    def __init__(self, path: Path, modality_dropout_p: float = 0.0, seed: int = 0):
        d = np.load(path)
        self.ecg_emb = torch.from_numpy(d["ecg_embedding"])   # [N,768]
        self.ecg_aux = torch.from_numpy(d["ecg_aux"])          # [N,10]
        self.imu     = torch.from_numpy(d["imu_feat"])          # [N,12]
        self.spo2    = torch.from_numpy(d["spo2_feat"])         # [N,8]
        self.mask    = torch.from_numpy(d["modality_mask"])     # [N,3]
        self.labels  = torch.from_numpy(d["label"])             # [N]
        self.dropout_p = modality_dropout_p
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mask = self.mask[idx].clone()
        if self.dropout_p > 0.0:
            drop = torch.from_numpy(
                (self.rng.random(3) < self.dropout_p).astype(np.float32)
            )
            if drop.sum() == 3:                      # 최소 1개 모달리티 보존
                drop[int(self.rng.integers(3))] = 0.0
            mask = mask * (1.0 - drop)
        return {
            "ecg_emb": self.ecg_emb[idx], "ecg_aux": self.ecg_aux[idx],
            "imu": self.imu[idx], "spo2": self.spo2[idx],
            "mask": mask, "label": self.labels[idx],
        }


def make_loaders(data_dir: Path, batch_size: int = 256,
                 modality_dropout_p: float = 0.15, num_workers: int = 0,
                 version: str = "v1") -> Tuple:
    """train/val/test DataLoader 3개. 파일명: p2_synth_{version}_{split}.npz"""
    from torch.utils.data import DataLoader

    def _loader(split, dropout):
        ds = P2Dataset(data_dir / f"p2_synth_{version}_{split}.npz",
                       modality_dropout_p=dropout)
        return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return (_loader("train", modality_dropout_p),
            _loader("val", 0.0), _loader("test", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# proxy 데이터셋 빌드 (자체완결 — P1 캐시에서 조립)
# ─────────────────────────────────────────────────────────────────────────────
def build_proxy_dataset(out_dir: Path, n_per_class: int = 4000, seed: int = 42,
                        version: str = "vf", imu_mode: str = "mvn",
                        cache_dir: Path = P1_CACHE_DIR) -> Dict[str, Path]:
    """클래스 조건부 조립기로 train/val/test .npz 생성 (ECG 누출 0%).

    train+val: P1 train+val 풀에서 ECG 샘플링 / test: P1 test 풀만 → 완전 분리.
    P1 캐시 없으면 합성 가우시안 ECG fallback (경고).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _make_cache(splits):
        try:
            return P1Cache(cache_dir=cache_dir, splits=splits)
        except FileNotFoundError as e:
            print(f"[warn] {e}\n-> ECG: 합성 가우시안 fallback")
            return None

    # train+val: 합산 조립 후 분리
    cache_tv = _make_cache(["train", "val"])
    asm_tv = ConditionalAssembler(seed=seed, p1_cache=cache_tv, imu_mode=imu_mode)
    tv = samples_to_arrays(asm_tv.assemble_balanced(n_per_class))
    n_tv = len(tv["label"])
    rng = np.random.default_rng(seed + 7)
    idx = rng.permutation(n_tv)
    n_tr = int(n_tv * (0.7 / 0.85))                  # train 비율 = 0.7/0.85 of (train+val)
    idx_tr, idx_va = idx[:n_tr], idx[n_tr:]

    # test: P1 test 풀만 (누출 0%)
    cache_te = _make_cache(["test"])
    asm_te = ConditionalAssembler(seed=seed + 99, p1_cache=cache_te, imu_mode=imu_mode)
    te_n = max(int(n_per_class * 0.15 / 0.85), 100)
    te = samples_to_arrays(asm_te.assemble_balanced(te_n))

    splits = {
        "train": {k: v[idx_tr] for k, v in tv.items()},
        "val":   {k: v[idx_va] for k, v in tv.items()},
        "test":  te,
    }
    paths: Dict[str, Path] = {}
    for name, arrays in splits.items():
        p = out_dir / f"p2_synth_{version}_{name}.npz"
        np.savez_compressed(p, **arrays)
        paths[name] = p
        print(f"[{name:5s}] n={len(arrays['label']):6d} -> {p.name}")
    return paths
