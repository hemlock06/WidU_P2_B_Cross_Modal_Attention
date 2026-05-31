"""PPG-DaLiA(PPG_FieldStudy) 실 ECG → P1 추론 캐시 (dalia_p1.npz) — 이상탐지 정상 joint 보강.

PPG-DaLiA: 15명, chest ECG 700Hz + chest ACC 700Hz **동일 길이 = 완벽 시간동기**(단일 기기
ECG+가속도 구조). 8활동(sit/stairs/soccer/cycling/driving/lunch/walking/working)
= 운동 다양성 풍부 → ptt 운동관용 신호(run/sit 경향, 3-seed 분산 큼)를 더 큰 운동
다양성으로 재현·견고성 판정.

ptt 빌더(build_ptt_ppg_p1_cache.py) 재활용 + 차이:
  · 입력 = pickle(.pkl) chest/ECG, chest/ACC (wfdb 아님)
  · 700Hz → 500Hz 리샘플 (ECG-FM 입력 500Hz·5000샘플 정합)
  · ECG 이미 mV단위(std~0.29) → CPSC target std 재스케일만 (센서 원단위 변환 불요)
  · activity.csv 구간(초) 파싱 → rest{BASELINE,LUNCH,WORKING,DRIVING}=0 / active{STAIRS,SOCCER,
    CYCLING,WALKING}=1. NO_ACTIVITY 전이구간 제외.
  · ACC만(자이로 없음) → 자이로 0-fill (ptt 빌더와 동일 처리)

사용: python scripts/build_dalia_p1_cache.py  (환경변수 P1_ROOT·P2_DATA_DIR·DALIA_DIR)
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
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
DALIA_DIR = Path(os.environ.get("DALIA_DIR", "data/raw/dalia"))
OUT_PATH  = str(_DATA / "p1_cache" / "dalia_p1.npz")

FS_SRC, FS = 700, 500    # 원본 700Hz → ECG-FM 500Hz 리샘플
WIN = 5000               # 10s @ 500Hz
MAX_WIN_PER_ACT = 8      # (피험자,활동)당 최대 윈도우 (균형·크기 제한)
T_MASK, T_ALERT = 0.2155, 0.4753
LEAD_SLOT = 1

# activity.csv 라벨명 → rest(0)/active(1). NO_ACTIVITY/전이 제외.
REST_ACTS   = {"BASELINE", "CLEAN_BASELINE", "LUNCH", "WORKING", "DRIVING"}
ACTIVE_ACTS = {"STAIRS", "SOCCER", "CYCLING", "WALKING"}


# ── P1 구성 (ptt 빌더와 동일) ────────────────────────────────────────────────
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
            dd = np.diff(pk) / FS
            hr[i] = float(60.0 / dd.mean())
            rr[i] = float(np.clip(1.0 - (dd.std() / (dd.mean() + 1e-6)) * 3, 0.0, 1.0))
    return hr, rr


def cpsc_target_std() -> float:
    s = np.load(CPSC_TEST)
    return float(np.median(s[:, LEAD_SLOT].std(axis=1)))


def normalize_ecg(ecg_500: np.ndarray, target_std: float) -> np.ndarray:
    """이미 500Hz·mV 단위. bandpass + per-record z-score×target_std + 폴라리티 정렬."""
    b, a = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
    x = filtfilt(b, a, ecg_500).astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-8)
    if skew(x) < 0:
        x = -x
    return (x * target_std).astype(np.float32)


def parse_activity_csv(path: Path):
    """activity.csv → [(label_name, start_sec)] 정렬. 다음 항목 start가 현 항목 end."""
    segs = []
    for ln in path.read_text(errors="ignore").splitlines():
        ln = ln.strip().lstrip("#").strip()
        if "," not in ln or ln.upper().startswith("SUBJECT"):
            continue
        name, val = [t.strip() for t in ln.split(",", 1)]
        try:
            sec = float(val)
        except ValueError:
            continue
        segs.append((name.upper(), sec))
    segs.sort(key=lambda s: s[1])
    return segs


FS_PPG = 64    # wrist BVP 샘플레이트


def ppg_window_feat(bvp_win: np.ndarray) -> np.ndarray:
    """PPG(BVP) 10초窓 → 8-d 피처(SpO2 슬롯 재활용, 모델 구조 무변경).
    박동 기반: 평균/표준편차/박동률(peak)/박동간격 변동/진폭/스펙트럴 등.
    ECG와 같은 심박이라 중복모달 — attention 상보성 시험용(우위 없음 예상, 측정으로 확인)."""
    from scipy.signal import find_peaks
    x = bvp_win.astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-8)
    h = max(float(np.percentile(x, 75)), 0.3)
    pk, _ = find_peaks(x, height=h, distance=int(FS_PPG * 0.4))   # 박동(>~150bpm 배제)
    if len(pk) >= 2:
        ibi = np.diff(pk) / FS_PPG
        hr = 60.0 / (ibi.mean() + 1e-6)
        hrv = ibi.std()
        amp = float(x[pk].mean())
    else:
        hr, hrv, amp = 0.0, 0.0, 0.0
    fft = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1 / FS_PPG)
    dom = float(freqs[1 + np.argmax(fft[1:])]) if len(fft) > 1 else 0.0
    p = fft / (fft.sum() + 1e-8); spec_ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))
    return np.array([x.mean(), x.std(), hr / 100.0, hrv, amp,
                     len(pk) / 10.0, dom, spec_ent], dtype=np.float32)


def load_subject(sdir: Path):
    pkl = sdir / f"{sdir.name}.pkl"
    acsv = sdir / f"{sdir.name}_activity.csv"
    if not pkl.exists() or not acsv.exists():
        return None
    d = pickle.load(open(pkl, "rb"), encoding="latin1")
    ecg700 = d["signal"]["chest"]["ECG"][:, 0].astype(np.float64)
    acc700 = d["signal"]["chest"]["ACC"].astype(np.float64)
    bvp64 = d["signal"]["wrist"]["BVP"][:, 0].astype(np.float64)
    segs = parse_activity_csv(acsv)
    return ecg700, acc700, bvp64, segs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt = cpsc_target_std()
    print(f"device={device}  CPSC target lead-II std={tgt:.4f}  (700→{FS}Hz 리샘플)")

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

    sig_batch, meta = [], []
    for sdir in sorted(DALIA_DIR.glob("S*")):
        loaded = load_subject(sdir)
        if loaded is None:
            print(f"  skip {sdir.name}"); continue
        ecg700, acc700, bvp64, segs = loaded
        # 700→500 리샘플 (전체 신호 한 번)
        ecg500 = resample_poly(ecg700, FS, FS_SRC)
        acc500 = np.stack([resample_poly(acc700[:, j], FS, FS_SRC) for j in range(3)], axis=1)
        ecg500 = normalize_ecg(ecg500, tgt)
        WIN_PPG = int(WIN / FS * FS_PPG)               # 10초 @ 64Hz = 640
        # 활동 구간별 윈도우
        for k, (name, start) in enumerate(segs):
            if name not in REST_ACTS and name not in ACTIVE_ACTS:
                continue
            end = segs[k + 1][1] if k + 1 < len(segs) else (len(ecg500) / FS)
            lab = 0 if name in REST_ACTS else 1
            avail = min(len(ecg500), len(acc500))      # 두 신호 공통 길이 가드
            s0 = int(start * FS); s1 = min(int(end * FS), avail)
            n_win = min(max(s1 - s0, 0) // WIN, MAX_WIN_PER_ACT)
            for w in range(n_win):
                a0 = s0 + w * WIN
                if a0 + WIN > avail:                    # 끝 경계 초과 방지
                    break
                sig12 = np.zeros((12, WIN), np.float32); sig12[LEAD_SLOT] = ecg500[a0:a0 + WIN]
                imu6 = np.zeros((WIN, 6), np.float32); imu6[:, :3] = acc500[a0:a0 + WIN]
                imu_feat = window_to_imu_feat(imu6, fs=FS, accel_unit="g")  # DaLiA ACC 단위=g
                # PPG: 같은 시각窓을 64Hz로 변환해 절취 (시간정합)
                p0 = int(a0 / FS * FS_PPG)
                ppg_win = bvp64[p0:p0 + WIN_PPG]
                ppg_feat = ppg_window_feat(ppg_win) if len(ppg_win) == WIN_PPG else np.zeros(8, np.float32)
                sig_batch.append(sig12)
                meta.append((imu_feat, ppg_feat, lab, sdir.name, name.lower()))
        print(f"  {sdir.name}: {sum(1 for m in meta if m[3]==sdir.name)} win 누적")

    sig_batch = np.stack(sig_batch)
    print(f"총 윈도우: {len(sig_batch)}")

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
    ppg_arr = np.stack([m[1] for m in meta]).astype(np.float32)
    lab = np.array([m[2] for m in meta], np.int64)
    subj = np.array([m[3] for m in meta]); actv = np.array([m[4] for m in meta])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, embedding=emb, cardiac_probs=cp, emergency_score=es,
                        reliability=rel, gate_tier=gt, hr_bpm=hr, rhythm_regularity=rr,
                        imu_feat=imu_arr, ppg_feat=ppg_arr, label=lab, subject=subj, activity=actv)
    print(f"저장: {OUT_PATH}  N={len(emb)}  피험자={len(set(subj.tolist()))}")

    print("\n=== 검증 게이트 (정규화 타당성) ===")
    for name, m in [("rest(label0)", lab == 0), ("active(label1)", lab == 1)]:
        if m.sum() == 0: continue
        print(f"  [{name}] n={int(m.sum())}  emergency_score={es[m].mean():.3f}  "
              f"reliability={rel[m].mean():.3f}  NSR_prob={cp[m, 0].mean():.3f}  hr={hr[m].mean():.0f}")
    rest_es = es[lab == 0].mean() if (lab == 0).any() else 1.0
    print(f"  → 게이트: {'PASS (rest emergency 낮음 → 정규화 타당)' if rest_es < 0.4 else 'FAIL (정규화/폴라리티 의심)'}")


if __name__ == "__main__":
    main()
