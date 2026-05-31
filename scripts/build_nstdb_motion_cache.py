"""NSTDB 모션 노이즈 ECG → P1 추론 캐시 (nstdb_motion.npz).

CPSC NSR 신호(label_mc==0, 전 split)에 NSTDB 전극모션·근전도(em·ma) 노이즈를
0dB SNR로 합성한 뒤 P1 모델을 돌려 embedding·cardiac_probs·emergency_score·
reliability·gate_tier·hr_bpm·rhythm_regularity를 cpsc_mc_*.npz 와 동일
스키마로 저장한다.

motion confounder 목적:
  "ECG 신호품질 불량(reliability↑=bad) + 맥락은 정상 IMU·SpO2"
  → 모델이 신뢰도 낮은 ECG를 cardiac(2) 오발화하는지 측정.
  P1 reliability가 이미 모션을 플래그하므로 within-modality 거부 가능 confounder.
  (chronic·apnea는 cross-modal 맥락 필요 → 전혀 다른 실패 모드)

사용:
    cd D:/WidU_P2_B_Cross_Modal_Attention
    D:/conda_envs/py39/python.exe scripts/build_nstdb_motion_cache.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, r"D:\WidU_ecg-fm_emergency-detection_git\scripts")

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

from multisnr import MultiSNRNoise

# ── 경로 ─────────────────────────────────────────────────────────────────────
CKPT_FM   = r"D:\WidU_ecg-fm_emergency-detection\checkpoints\ecg-fm\mimic_iv_ecg_physionet_pretrained.pt"
CKPT_P1   = r"D:\WidU_ecg-fm_emergency-detection\outputs\lora_multitask_snr_a07\lora_multitask_snr_best.pt"
CKPT_GATE = r"D:\WidU_ecg-fm_emergency-detection\outputs\gate\gate_best.pt"
DATA_DIR  = r"D:\WidU_ecg-fm_emergency-detection\data\processed\cpsc2018_mc"
NSTDB_DIR = r"D:\WidU_ecg-fm_emergency-detection\data\raw\nstdb"
OUT_PATH  = r"D:\WidU_multimodal_fusion\p1_cache\nstdb_motion.npz"

T_MASK  = 0.2155
T_ALERT = 0.4753
FS      = 500
SNR_DB  = 0.0   # 0dB: 신호파워=노이즈파워 → reliability 확실히 alert 플래그

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ── P1 구성요소 (build_p1_cache.py 와 동일) ──────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.original = linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def bias(self):   return self.original.bias
    @property
    def weight(self): return self.original.weight
    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(model, rank=8, alpha=16, dropout=0.0,
                target_suffixes=("self_attn.q_proj", "self_attn.v_proj")):
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))


class BinaryHead(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(768, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

class MulticlassHead(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(768, 5)
    def forward(self, x): return self.fc(x)

class GateHead(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(768, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)


def estimate_physio(signals_np: np.ndarray):
    """signals_np: [N, 12, 5000] → hr_bpm[N], rhythm_regularity[N]"""
    N = signals_np.shape[0]
    hr_bpm = np.full(N, 75.0, dtype=np.float32)
    rhythm_reg = np.full(N, 0.9, dtype=np.float32)
    for i in range(N):
        lead = signals_np[i, 0]
        height = max(float(lead.max()) * 0.3, 0.1)
        peaks, _ = find_peaks(lead, height=height, distance=int(FS * 0.3))
        if len(peaks) >= 2:
            rr = np.diff(peaks) / FS
            hr_bpm[i] = float(60.0 / rr.mean())
            cv = float(rr.std()) / (float(rr.mean()) + 1e-6)
            rhythm_reg[i] = float(np.clip(1.0 - cv * 3, 0.0, 1.0))
    return hr_bpm, rhythm_reg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  SNR_DB={SNR_DB}")

    # 1) NSR 신호 수집 (label_mc==0, 전 split)
    print("\n[1] NSR 신호 로드...")
    parts = []
    for split in ["train", "val", "test"]:
        sigs   = np.load(os.path.join(DATA_DIR, split, "signals.npy"))
        labels = np.load(os.path.join(DATA_DIR, split, "labels.npy"))
        mask = labels == 0
        parts.append(sigs[mask])
        print(f"  {split}: NSR {mask.sum()}/{len(labels)}")
    sigs_nsr = np.concatenate(parts)   # [M, 12, 5000]
    print(f"  총 NSR: {len(sigs_nsr)}")

    # 2) NSTDB 노이즈 합성 (em 위주 — 전극모션)
    print("\n[2] NSTDB 노이즈 합성...")
    aug = MultiSNRNoise(
        nstdb_dir=NSTDB_DIR,
        snr_set=(SNR_DB,),
        noise_weights=(0.15, 0.55, 0.30),   # bw·em·ma; em 강조
        device=torch.device("cpu"),
        seed=7,
    )
    x_clean = torch.from_numpy(sigs_nsr.astype(np.float32))   # [M, 12, 5000]
    x_noisy = aug.inject_fixed(x_clean, SNR_DB).numpy()       # [M, 12, 5000]
    print(f"  noisy shape: {x_noisy.shape}")

    # 3) P1 로드
    print("\n[3] P1 모델 로드...")
    from fairseq_signals.utils.checkpoint_utils import load_model_and_task
    result = load_model_and_task(CKPT_FM)
    backbone = next(r for r in (result if isinstance(result, (list, tuple)) else [result])
                    if hasattr(r, "parameters"))
    backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank=8, alpha=16, dropout=0.0)

    p1_ckpt = torch.load(CKPT_P1, map_location=device)
    backbone.load_state_dict(p1_ckpt["backbone_lora"], strict=False)
    head_bin = BinaryHead().to(device);     head_bin.load_state_dict(p1_ckpt["head_bin_state"])
    head_mc  = MulticlassHead().to(device); head_mc.load_state_dict(p1_ckpt["head_mc_state"])

    gate_ckpt = torch.load(CKPT_GATE, map_location=device)
    gate_head = GateHead().to(device);     gate_head.load_state_dict(gate_ckpt["head_state"])

    backbone.eval(); head_bin.eval(); head_mc.eval(); gate_head.eval()

    # 4) 배치 추론
    print("\n[4] 추론 중...")
    BATCH = 32
    all_emb, all_cp, all_es, all_rel = [], [], [], []
    M = len(x_noisy)
    with torch.no_grad():
        for i in range(0, M, BATCH):
            x = torch.from_numpy(x_noisy[i:i+BATCH]).to(device)
            out  = backbone(source=x, padding_mask=None, features_only=True)
            emb  = out["x"].mean(dim=1)
            all_emb.append(emb.cpu().numpy().astype(np.float32))
            all_cp.append(torch.softmax(head_mc(emb), dim=-1).cpu().numpy().astype(np.float32))
            all_es.append(torch.sigmoid(head_bin(emb)).cpu().numpy().astype(np.float32))
            all_rel.append(torch.sigmoid(gate_head(emb)).cpu().numpy().astype(np.float32))
            if (i // BATCH) % 10 == 0:
                print(f"  {i}/{M}")

    emb_arr = np.concatenate(all_emb)
    cp_arr  = np.concatenate(all_cp)
    es_arr  = np.concatenate(all_es)
    rel_arr = np.concatenate(all_rel)
    gt_arr  = np.where(rel_arr < T_MASK, 0,
               np.where(rel_arr < T_ALERT, 1, 2)).astype(np.int8)

    # 5) 생리지표 (노이즈 신호 기반 — 모션 시 추정 정확도 낮음이 실제와 일치)
    print("\n[5] 생리지표 추정...")
    hr_bpm, rhythm_reg = estimate_physio(x_noisy)

    # 6) 저장
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        embedding=emb_arr,
        cardiac_probs=cp_arr,
        emergency_score=es_arr,
        reliability=rel_arr,
        gate_tier=gt_arr,
        hr_bpm=hr_bpm,
        rhythm_regularity=rhythm_reg,
    )
    print(f"\n저장: {OUT_PATH}")
    print(f"N={len(emb_arr)}")
    print(f"reliability  mean={rel_arr.mean():.3f}  alert%={100*(gt_arr==2).mean():.1f}%")
    print(f"emergency_score mean={es_arr.mean():.3f}")
    print(f"cardiac_probs   mean={cp_arr.mean(0).round(3)}")
    print("완료.")


if __name__ == "__main__":
    main()
