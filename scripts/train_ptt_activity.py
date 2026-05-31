"""ptt_ppg 정상 3분류(sit/walk/run) — 학습 융합 3모델 + 모달 ablation.

운동 오경보 억제 평가와 별개: 여기는 정상상태 분류력(택소노미 normal_rest/active 근거).
실데이터 100%·합성 0. 피험자단위 분할 정본(ptt_subject_split.json) 사용 — 윈도우 섞기 금지.
모델: concat/gated/cross_attn(num_classes=3). 지표: macro-F1 + 혼동행렬 + 모달 ablation(IMU/ECG/둘다).
사전 예측(사후합리화 방지): 활동=가속도 직접 → IMU 우세 예상. attention 우위 선험 가정 안 함.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fusion.model import ConcatMLP, GatedFusionModel, CrossModalAttentionFusion
from fusion.metrics import macro_f1, model_logits
from fusion.confounders import _ecg_aux_from_cache, _normal_spo2_feat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DATA  = Path(os.environ.get("P2_DATA_DIR", "data"))
PTT    = _DATA / "p1_cache" / "ptt_ppg_p1.npz"
SPLIT  = _DATA / "interim" / "ptt_subject_split.json"
ARTIFACTS = ROOT / "fusion" / "artifacts"
CLS = ["sit", "walk", "run"]; NCLS = 3
ACT2LAB = {"sit": 0, "walk": 1, "run": 2}
EPOCHS, LR, SEED = 80, 3e-4, 42
MOD_DROP = 0.2   # 학습 중 ECG/IMU 결측 → unimodal 경로 학습 → ablation 유의미

FACTORY = {
    "concat":     lambda: ConcatMLP(hidden_dims=(256, 128), dropout_p=0.3, num_classes=NCLS),
    "gated":      lambda: GatedFusionModel(fusion_hidden=(128,), dropout=0.3, aux_loss_weight=0.3,
                                           reliability_mode="feature", num_classes=NCLS),
    "cross_attn": lambda: CrossModalAttentionFusion(d_model=128, n_heads=4, n_layers=2, dropout=0.3,
                                                    aux_loss_weight=0.3, emb_bottleneck=16, num_classes=NCLS),
}


def load():
    d = np.load(PTT); c = {k: d[k] for k in d.files}
    n = len(c["embedding"])
    feats = {
        "ecg_emb": c["embedding"].astype(np.float32),
        "ecg_aux": _ecg_aux_from_cache(c, np.arange(n)),
        "imu":     c["imu_feat"].astype(np.float32),
        "spo2":    np.tile(_normal_spo2_feat(), (n, 1)).astype(np.float32),  # 상수(ptt SpO2 없음) → 마스크
        "label":   np.array([ACT2LAB[a] for a in c["activity"]], dtype=np.int64),
    }
    return feats, c["subject"]


def batch(feats, idx, mask_vec, rng=None, drop=0.0):
    b = {"ecg_emb": torch.tensor(feats["ecg_emb"][idx]), "ecg_aux": torch.tensor(feats["ecg_aux"][idx]),
         "imu": torch.tensor(feats["imu"][idx]), "spo2": torch.tensor(feats["spo2"][idx]),
         "label": torch.tensor(feats["label"][idx])}
    m = np.tile(np.array(mask_vec, np.float32), (len(idx), 1))
    if drop > 0 and rng is not None:                       # ECG/IMU 독립 드롭(둘 중 최소 1개 보존)
        for j in range(len(idx)):
            dr = (rng.random(2) < drop)
            if dr.all(): dr[rng.integers(2)] = False
            m[j, 0] *= (1 - dr[0]); m[j, 1] *= (1 - dr[1])
    b["mask"] = torch.tensor(m)
    return {k: v.to(DEVICE) for k, v in b.items()}


def confusion(preds, labels):
    cm = np.zeros((NCLS, NCLS), int)
    for t, p in zip(labels, preds): cm[t, p] += 1
    return cm


def fit_eval(name, feats, tr_idx, te_idx):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = FACTORY[name]().to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
    ce = nn.CrossEntropyLoss()
    bs = 128
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), bs):
            bi = tr_idx[i:i + bs]
            bt = batch(feats, bi, [1, 1, 0], rng=rng, drop=MOD_DROP)
            opt.zero_grad()
            if isinstance(model, (GatedFusionModel, CrossModalAttentionFusion)):
                out = model(bt); loss = model.loss(bt, out)
            else:
                loss = ce(model(bt), bt["label"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

    # eval: full / IMU-only / ECG-only
    model.eval()
    res = {}
    with torch.no_grad():
        for tag, mv in [("both", [1, 1, 0]), ("imu_only", [0, 1, 0]), ("ecg_only", [1, 0, 0])]:
            bt = batch(feats, te_idx, mv)
            preds = model_logits(model, bt).argmax(-1).cpu().numpy()
            labs = feats["label"][te_idx]
            res[tag] = {"f1": macro_f1(preds, labs, NCLS),
                        "cm": confusion(preds, labs).tolist() if tag == "both" else None}
    return res


def main():
    feats, subj = load()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tr_idx = np.where(np.isin(subj, split["train"]))[0]
    te_idx = np.where(np.isin(subj, split["test"]))[0]
    print(f"split seed={split['seed']}  train={len(tr_idx)}w/{len(split['train'])}명  "
          f"test={len(te_idx)}w/{len(split['test'])}명")
    print(f"3분류 macro-F1 (피험자분할, 모달 ablation):\n")
    print(f"{'model':<12}{'both':>8}{'imu_only':>10}{'ecg_only':>10}")
    results = {}
    for name in ["concat", "gated", "cross_attn"]:
        t0 = time.time(); r = fit_eval(name, feats, tr_idx.copy(), te_idx)
        results[name] = r
        print(f"{name:<12}{r['both']['f1']:>8.3f}{r['imu_only']['f1']:>10.3f}"
              f"{r['ecg_only']['f1']:>10.3f}   ({time.time()-t0:.0f}s)")
    print("\n혼동행렬 (both, 행=true[sit,walk,run] 열=pred):")
    for name in results:
        print(f"  [{name}]")
        for i, row in enumerate(results[name]["both"]["cm"]):
            print(f"    {CLS[i]:<5} {row}")

    # 리포트 저장
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "ptt_activity_report.json"
    out.write_text(json.dumps({"split": split, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
