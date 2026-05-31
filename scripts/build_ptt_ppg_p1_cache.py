"""PTT-PPG 실 ECG → P1 추론 캐시 (ptt_ppg_p1.npz) — 실 페어드(ECG+IMU).

PTT-PPG는 유일 실 ECG+IMU 시간정렬쌍(22명 × sit/walk/run, 500Hz). 단 raw ECG는 센서
원단위(~3e4)라 ECG-FM 입력분포(CPSC lead-II: mean0·std~0.12)와 불일치 → 스케일·폴라리티
정합 필수(미정합 시 LTST식 score 역전). 정합 = bandpass(0.5–40Hz) + per-record z-score×CPSC
타깃std + R-peak 폴라리티 정렬. **검증 게이트**: ptt-sit P1 emergency_score가 NSR급(낮음)이
아니면 정규화 실패로 간주(평가 무효).

per-window 페어: ECG window → P1(emb+aux), 같은 window IMU → imu_feat. label sit=0, walk/run=1.

사용: python scripts/build_ptt_ppg_p1_cache.py  (환경변수 P1_ROOT·P2_DATA_DIR)
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from fusion.features import window_to_imu_feat

_P1   = os.environ.get("P1_ROOT", "../WidU_ecg-fm_emergency-detection")
_DATA = Path(os.environ.get("P2_DATA_DIR", "data"))
CKPT_FM   = os.path.join(_P1, "checkpoints", "ecg-fm", "mimic_iv_ecg_physionet_pretrained.pt")
CKPT_P1   = os.path.join(_P1, "outputs", "lora_multitask_snr_a07", "lora_multitask_snr_best.pt")
CKPT_GATE = os.path.join(_P1, "outputs", "gate", "gate_best.pt")
CPSC_TEST = os.path.join(_P1, "data", "processed", "cpsc2018_mc", "test", "signals.npy")
PTT_DIR   = _DATA / "raw" / "ptt_ppg"
OUT_PATH  = str(_DATA / "p1_cache" / "ptt_ppg_p1.npz")

FS = 500
WIN = 5000              # 10s @ 500Hz (ECG-FM 입력 길이)
MAX_WIN_PER_REC = 16    # 레코드당 최대 윈도우 (크기 제한)
T_MASK, T_ALERT = 0.2155, 0.4753
ACTIVITY_LABEL = {"sit": 0, "walk": 1, "run": 1}
LEAD_SLOT = 1           # 단일리드 → ECG-FM 슬롯 II(index 1)


# ── P1 구성 (build_nstdb_motion_cache.py 와 동일) ────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, lin, r, a, d):
        super().__init__(); self.original = lin; lin.weight.requires_grad_(False)
        if lin.bias is not None: lin.bias.requires_grad_(False)
        self.lora_A = nn.Linear(lin.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, lin.out_features, bias=False)
        self.scaling = a / r; self.dropout = nn.Dropout(d)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
    @property
    def bias(self):   return self.original.bias
    @property
    def weight(self): return self.original.weight
    def forward(self, x): return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(model, rank=8, alpha=16, dropout=0.0,
                suffixes=("self_attn.q_proj", "self_attn.v_proj")):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in suffixes):
            parts = name.split("."); parent = model
            for p in parts[:-1]: parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(mod, rank, alpha, dropout))


class _Head(nn.Module):
    def __init__(self, o): super().__init__(); self.fc = nn.Linear(768, o)
    def forward(self, x): return self.fc(x).squeeze(-1) if self.fc.out_features == 1 else self.fc(x)


def estimate_physio(sig12):
    N = sig12.shape[0]
    hr = np.full(N, 75.0, np.float32); rr = np.full(N, 0.9, np.float32)
    for i in range(N):
        lead = sig12[i, LEAD_SLOT]
        h = max(float(lead.max()) * 0.3, 0.05)
        pk, _ = find_peaks(lead, height=h, distance=int(FS * 0.3))
        if len(pk) >= 2:
            d = np.diff(pk) / FS
            hr[i] = float(60.0 / d.mean())
            rr[i] = float(np.clip(1.0 - (d.std() / (d.mean() + 1e-6)) * 3, 0.0, 1.0))
    return hr, rr


# ── ECG 정규화 (CPSC 분포 정합) ──────────────────────────────────────────────
def cpsc_target_std() -> float:
    s = np.load(CPSC_TEST)               # (N,12,5000)
    per_rec = s[:, LEAD_SLOT].std(axis=1)
    return float(np.median(per_rec))     # ~0.12


def normalize_ecg(ecg_raw: np.ndarray, target_std: float) -> np.ndarray:
    """bandpass 0.5–40Hz + per-record z-score×target_std + R-peak 폴라리티 정렬."""
    b, a = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    x = filtfilt(b, a, ecg_raw).astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-8)
    if skew(x) < 0:                      # R-peak가 음으로 우세하면 반전
        x = -x
    return (x * target_std).astype(np.float32)


def load_record(hea: Path):
    import wfdb
    rec = wfdb.rdrecord(str(hea.with_suffix("")))
    data = rec.p_signal.astype(np.float32); ch = [c.lower() for c in rec.sig_name]
    def idx(names):
        for nm in names:
            if nm in ch: return ch.index(nm)
        return None
    e = idx(["ecg"]); ax, ay, az = idx(["a_x"]), idx(["a_y"]), idx(["a_z"])
    gx, gy, gz = idx(["g_x"]), idx(["g_y"]), idx(["g_z"])
    if None in (e, ax, ay, az): return None
    ecg = data[:, e]
    imu6 = np.stack([data[:, ax], data[:, ay], data[:, az],
                     data[:, gx] if gx else np.zeros(len(data)),
                     data[:, gy] if gy else np.zeros(len(data)),
                     data[:, gz] if gz else np.zeros(len(data))], axis=1)
    imu6[:, 3:] *= (np.pi / 180.0)       # deg/s → rad/s
    return ecg, imu6


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt = cpsc_target_std()
    print(f"device={device}  CPSC target lead-II std={tgt:.4f}")

    # P1 로드
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    res = load_model_and_task(CKPT_FM)
    bb = next(r for r in (res if isinstance(res, (list, tuple)) else [res]) if hasattr(r, "parameters")).to(device)
    for p in bb.parameters(): p.requires_grad_(False)
    inject_lora(bb)
    ck = torch.load(CKPT_P1, map_location=device); bb.load_state_dict(ck["backbone_lora"], strict=False)
    hb = _Head(1).to(device); hb.load_state_dict(ck["head_bin_state"])
    hm = _Head(5).to(device); hm.load_state_dict(ck["head_mc_state"])
    gk = torch.load(CKPT_GATE, map_location=device); hg = _Head(1).to(device); hg.load_state_dict(gk["head_state"])
    bb.eval(); hb.eval(); hm.eval(); hg.eval()

    # 윈도우 조립
    rows = {k: [] for k in ["embedding", "cardiac_probs", "emergency_score", "reliability",
                            "gate_tier", "hr_bpm", "rhythm_regularity", "imu_feat", "label",
                            "subject", "activity"]}
    sig_batch, meta = [], []
    for hea in sorted(PTT_DIR.glob("*.hea")):
        stem = hea.stem; subj, act = stem.split("_", 1)
        rec = load_record(hea)
        if rec is None:
            print(f"  skip {stem}"); continue
        ecg, imu6 = rec
        ecg = normalize_ecg(ecg, tgt)
        n_win = min(len(ecg) // WIN, MAX_WIN_PER_REC)
        for w in range(n_win):
            sl = slice(w * WIN, (w + 1) * WIN)
            sig12 = np.zeros((12, WIN), np.float32); sig12[LEAD_SLOT] = ecg[sl]
            imu_feat = window_to_imu_feat(imu6[sl], fs=FS, accel_unit="ms2")
            sig_batch.append(sig12); meta.append((imu_feat, ACTIVITY_LABEL[act], subj, act))

    sig_batch = np.stack(sig_batch)
    print(f"총 윈도우: {len(sig_batch)} (records×win)")

    # 배치 P1 추론
    BATCH = 32
    embs, cps, ess, rels = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(sig_batch), BATCH):
            x = torch.from_numpy(sig_batch[i:i + BATCH]).to(device)
            emb = bb(source=x, padding_mask=None, features_only=True, mask=False)["x"].mean(1)
            embs.append(emb.cpu().numpy().astype(np.float32))
            cps.append(torch.softmax(hm(emb), -1).cpu().numpy().astype(np.float32))
            ess.append(torch.sigmoid(hb(emb)).cpu().numpy().astype(np.float32))
            rels.append(torch.sigmoid(hg(emb)).cpu().numpy().astype(np.float32))
    emb = np.concatenate(embs); cp = np.concatenate(cps); es = np.concatenate(ess); rel = np.concatenate(rels)
    gt = np.where(rel < T_MASK, 0, np.where(rel < T_ALERT, 1, 2)).astype(np.int8)
    hr, rr = estimate_physio(sig_batch)

    imu_arr = np.stack([m[0] for m in meta]).astype(np.float32)
    lab = np.array([m[1] for m in meta], np.int64)
    subj = np.array([m[2] for m in meta]); actv = np.array([m[3] for m in meta])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, embedding=emb, cardiac_probs=cp, emergency_score=es,
                        reliability=rel, gate_tier=gt, hr_bpm=hr, rhythm_regularity=rr,
                        imu_feat=imu_arr, label=lab, subject=subj, activity=actv)
    print(f"저장: {OUT_PATH}  N={len(emb)}")

    # ── 검증 게이트: sit(정상) P1 출력이 NSR급인가 ──────────────────────────
    print("\n=== 검증 게이트 (정규화 타당성) ===")
    for name, m in [("sit(label0)", lab == 0), ("walk/run(label1)", lab == 1)]:
        if m.sum() == 0: continue
        print(f"  [{name}] n={int(m.sum())}  emergency_score={es[m].mean():.3f}  "
              f"reliability={rel[m].mean():.3f}  NSR_prob={cp[m, 0].mean():.3f}  "
              f"hr={hr[m].mean():.0f}")
    sit_es = es[lab == 0].mean() if (lab == 0).any() else 1.0
    verdict = "PASS (sit emergency 낮음 → 정규화 타당)" if sit_es < 0.4 else \
              "FAIL (sit emergency 높음 → 정규화/폴라리티 의심, 평가 무효)"
    print(f"  → 게이트: {verdict}")


if __name__ == "__main__":
    main()
