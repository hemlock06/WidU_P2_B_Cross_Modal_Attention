"""multi-SNR 모션 증강 모듈 (자체완결).

목적: clean ECG에 NSTDB 모션 노이즈를 calibrated SNR로 주입해, ECG-FM이 보지 못한
      "모션 강건성"을 학습 단계에서 이식한다.

설계:
  - SNR 분포  : 이산 집합 {24,18,12,6,0} dB, 균등 추출
  - clean 혼합: 샘플의 25%는 clean 유지 (p_noise=0.75)
  - lead별 SNR: 각 lead 독립 샘플링 (wearable lead별 접촉 품질 차이 모사)

핵심 수식:
  SNR_dB = 10·log10(P_signal / P_noise)
  목표 SNR을 맞추는 노이즈 스케일 계수:
    alpha = sqrt( P_signal / (P_noise_raw · 10^(SNR/10)) )
    x_noisy = x + alpha · n
  → alpha는 신호 자기 파워로 보정되므로 절대 단위에 무관 (unit-invariant).

파이프라인 위치:
  clean → [이 모듈: 노이즈 주입] → 리드 마스킹 → ECG-FM+LoRA
  노이즈를 먼저, 마스킹을 나중에 → 각 lead가 {노이즈 신호} 또는 {0=부재}로 깔끔히 분리.

노이즈 종류 (NSTDB): bw=baseline wander, ma=muscle artifact, em=electrode motion(기본 가중치 강조).
경로는 환경변수 NSTDB_DIR로 지정(기본 상대 "data/raw/nstdb").
"""

import os

import numpy as np
import torch
from scipy.signal import resample_poly

NSTDB_DIR_DEFAULT = os.environ.get("NSTDB_DIR", "data/raw/nstdb")
NOISE_TYPES = ("bw", "em", "ma")
NSTDB_FS = 360            # NSTDB 원본 샘플링레이트
TARGET_FS = 500           # ECG-FM 입력 샘플링레이트
# 500/360 = 25/18 (gcd=20) → resample_poly(up=25, down=18)
_UP, _DOWN = 25, 18
_EPS = 1e-8


class MultiSNRNoise:
    """
    NSTDB 노이즈를 미리 500Hz로 리샘플해 메모리에 적재하고,
    배치 텐서에 per-sample·per-lead 노이즈를 주입한다.
    """

    def __init__(
        self,
        nstdb_dir: str = NSTDB_DIR_DEFAULT,
        snr_set=(24, 18, 12, 6, 0),
        noise_weights=(0.25, 0.50, 0.25),  # (bw, em, ma) — em 강조
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        assert len(noise_weights) == len(NOISE_TYPES)
        self.snr_set = np.asarray(snr_set, dtype=np.float64)
        self.noise_weights = np.asarray(noise_weights, dtype=np.float64)
        self.noise_weights /= self.noise_weights.sum()
        self.device = device
        self.rng = np.random.default_rng(seed)

        # 노이즈 레코드를 500Hz로 한 번만 리샘플해 1D pool로 적재
        self.noise_pool = {}     # type -> torch.Tensor (1D, device)
        for t in NOISE_TYPES:
            self.noise_pool[t] = self._load_noise(nstdb_dir, t)

    def _load_noise(self, nstdb_dir: str, noise_type: str) -> torch.Tensor:
        import wfdb
        path = os.path.join(nstdb_dir, noise_type)
        sig, _ = wfdb.rdsamp(path)              # (650000, 2) @360Hz
        res = resample_poly(sig, _UP, _DOWN, axis=0)   # (~902777, 2) @500Hz
        # 2채널을 이어붙여 하나의 긴 1D pool로 → 랜덤 슬라이스 다양성 확보
        flat = res.T.reshape(-1).astype(np.float32)
        return torch.from_numpy(flat).to(self.device)

    def to(self, device: torch.device):
        self.device = device
        for t in self.noise_pool:
            self.noise_pool[t] = self.noise_pool[t].to(device)
        return self

    # ── 학습용: per-sample 게이트 + per-lead 독립 SNR ──────────────────
    def inject(self, x: torch.Tensor, p_noise: float = 0.75) -> torch.Tensor:
        """
        x: (B, C, T) clean 배치 (C=12, T=5000)
        반환: 같은 shape의 노이즈 주입 배치.
        - 각 샘플은 p_noise 확률로만 노이즈 적용 (나머지는 clean 유지)
        - 노이즈 적용 샘플의 각 lead는 독립적으로 SNR·노이즈종류·구간 선택
        """
        B, C, T = x.shape
        out = x.clone()
        for b in range(B):
            if self.rng.random() >= p_noise:
                continue  # clean 유지
            for c in range(C):
                snr = float(self.rng.choice(self.snr_set))
                ntype = NOISE_TYPES[self.rng.choice(len(NOISE_TYPES), p=self.noise_weights)]
                n = self._random_segment(ntype, T)
                out[b, c] = self._add_at_snr(out[b, c], n, snr)
        return out

    # ── 평가용: 전 lead에 고정 SNR 주입 (단계 8 SNR 저하 곡선) ──────────
    def inject_fixed(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        모든 샘플의 모든 lead에 동일 SNR을 주입 (노이즈 종류·구간은 랜덤).
        SNR 저하 곡선 평가 전용 — 학습에는 사용하지 않음.
        """
        B, C, T = x.shape
        out = x.clone()
        for b in range(B):
            for c in range(C):
                ntype = NOISE_TYPES[self.rng.choice(len(NOISE_TYPES), p=self.noise_weights)]
                n = self._random_segment(ntype, T)
                out[b, c] = self._add_at_snr(out[b, c], n, float(snr_db))
        return out

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────
    def _random_segment(self, noise_type: str, length: int) -> torch.Tensor:
        pool = self.noise_pool[noise_type]
        start = int(self.rng.integers(0, pool.shape[0] - length))
        return pool[start:start + length]

    @staticmethod
    def _add_at_snr(sig: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        p_sig = torch.mean(sig * sig)
        p_noise = torch.mean(noise * noise)
        if p_sig < _EPS or p_noise < _EPS:
            return sig  # flat lead → alpha 폭발 방지, 주입 생략
        alpha = torch.sqrt(p_sig / (p_noise * (10.0 ** (snr_db / 10.0))))
        return sig + alpha * noise


# ── 단독 실행: 모듈 자체 검증 (smoke test) ────────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={dev}")
    aug = MultiSNRNoise(device=dev)
    print(f"[smoke] 노이즈 pool 길이: "
          + ", ".join(f"{t}={aug.noise_pool[t].shape[0]:,}" for t in NOISE_TYPES))

    # 가짜 clean 배치로 SNR 검증: 주입 후 실측 SNR이 목표와 일치하는가
    torch.manual_seed(0)
    x = torch.randn(4, 12, 5000, device=dev) * 13.0   # std≈13 (CPSC 유사 스케일)

    print("\n[검증] 고정 SNR 주입 후 실측 SNR (lead 평균)")
    print(f"{'목표SNR(dB)':>10} {'실측SNR(dB)':>10}")
    for target in [24, 18, 12, 6, 0]:
        noisy = aug.inject_fixed(x, target)
        diff = noisy - x
        p_sig = (x ** 2).mean()
        p_noise = (diff ** 2).mean()
        meas = 10 * torch.log10(p_sig / p_noise).item()
        print(f"{target:>10} {meas:>10.2f}")

    # 학습용 inject: clean 유지 비율 확인
    noisy = aug.inject(x, p_noise=0.75)
    n_clean = sum(torch.allclose(noisy[b], x[b]) for b in range(x.shape[0]))
    print(f"\n[검증] inject(p_noise=0.75): 배치 4개 중 clean 유지 {n_clean}개")
    print("[smoke] 통과 — NaN:", torch.isnan(noisy).any().item(),
          "Inf:", torch.isinf(noisy).any().item())
