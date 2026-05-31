"""ptt_ppg 멀티모달 이상탐지 — 정상성 모델링(질문1): 운동(walk/run)을 '정상으로 관용'.

방향: "응급을 양성으로"가 데이터 가용성 제약에 막혀, "정상을 정밀 학습하고 분포 밖을
이탈로" 재정의(one-class). 핵심 요건 = 운동 같은 생리 변동을 **응급으로 오발화 안 하기**.

질문1 (1차 척도 혼선 교정): "운동을 정상으로 관용하나?"
  학습 = train 피험자의 sit+walk+run **전부 정상**(운동 관용 학습).
  정상 테스트 = held-out 피험자 sit/walk/run → 셋 다 낮은 이탈점수여야(관용 성공).
  이탈 테스트 = 통제 perturbation(정상 분포 밖 방향) → 높은 이탈점수여야(민감도).
  ★ 좋은 모델 = 운동 관용(walk/run 점수 ≈ sit, 낮음) AND 이탈 민감(perturb 점수 높음).
    → 분리지표 = AUROC(perturb vs 전체 정상[sit+walk+run]). 운동을 정상으로 묶는 게 핵심.

IMU robust 표준화: jerk_peak·tilt_change 등 긴 꼬리(최대 185) → log1p + 1~99분위 클리핑 후
  표준화(전체 정상 기준). 1차 run 재구성오차 1e10 폭발(sit기준 표준화 부작용) 교정.

비교: late-fusion AE vs cross-attention AE. perturbation은 합성이라 순환성 주의 —
  "정상성 모델이 분포 밖에 반응하나"의 민감도·배관 측정이지 응급 효능 증명 아님.

사용: python -m scripts.anomaly_ptt  (환경변수 P2_DATA_DIR)
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

from fusion.schema import EMB_DIM, IMU_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS = ROOT / "fusion" / "artifacts"
_DATA = Path(os.environ.get("P2_DATA_DIR", "data"))   # 외부 데이터 루트 (절대경로 미노출)

# 데이터셋 레지스트리: 경로 + 정상기준(rest) 활동 + 운동(active) 활동 라벨
DATASETS = {
    "ptt": {
        "path": _DATA / "p1_cache" / "ptt_ppg_p1.npz",
        "split": _DATA / "interim" / "ptt_subject_split.json",
        "rest": ["sit"], "active": ["walk", "run"],
    },
    "dalia": {
        "path": _DATA / "p1_cache" / "dalia_p1.npz",
        "split": None,    # 즉석 피험자분할(고정 seed)
        "rest": ["baseline", "clean_baseline", "lunch", "working", "driving"],
        "active": ["stairs", "soccer", "cycling", "walking"],
    },
}

EPOCHS, LR, BS = 120, 1e-3, 64
SEEDS = [42, 7, 123]       # 3-seed 견고성 (1차 단일 seed 미확정 교정)
PERTURB_SIGMAS = [1.0, 2.0, 3.0]   # 이탈 강도 다단계 (1.000 포화 풀어 민감도 차이 측정)
EMB_BN = 16
LATENT = 16
ECG_TOK_IN = EMB_BN + 4    # 병목 임베딩 + aux(es,rel,hr,rhythm) 4


# ─────────────────────────────────────────────────────────────────────────────
def load(path):
    d = np.load(path, allow_pickle=True)
    aux = np.stack([d["emergency_score"], d["reliability"],
                    d["hr_bpm"] / 100.0, d["rhythm_regularity"]], axis=1).astype(np.float32)
    feats = {
        "ecg_emb": d["embedding"].astype(np.float32),
        "ecg_aux": aux,
        "imu":     d["imu_feat"].astype(np.float32),
        "activity": d["activity"],
    }
    return feats, d["subject"]


# ─── 모델 (1차와 동일 구조) ───────────────────────────────────────────────────
class LateFusionAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_bn = nn.Sequential(nn.Linear(EMB_DIM, EMB_BN), nn.Dropout(0.5))
        self.ecg_enc = nn.Sequential(nn.Linear(ECG_TOK_IN, 32), nn.GELU(), nn.Linear(32, 16))
        self.imu_enc = nn.Sequential(nn.Linear(IMU_DIM, 32), nn.GELU(), nn.Linear(32, 16))
        self.to_latent = nn.Linear(32, LATENT)
        self.from_latent = nn.Linear(LATENT, 32)
        self.ecg_dec = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, ECG_TOK_IN))
        self.imu_dec = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, IMU_DIM))

    def forward(self, b, mask):
        ecg_in = torch.cat([self.ecg_bn(b["ecg_emb"]), b["ecg_aux"]], -1)
        imu_in = b["imu"]
        e = self.ecg_enc(ecg_in) * mask[:, 0:1]
        i = self.imu_enc(imu_in) * mask[:, 1:2]
        z = self.to_latent(torch.cat([e, i], -1))
        h = self.from_latent(z)
        rec_ecg = self.ecg_dec(h[:, :16]); rec_imu = self.imu_dec(h[:, 16:32])
        return (ecg_in, imu_in), (rec_ecg, rec_imu), mask


class CrossAttnAE(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.ecg_bn = nn.Sequential(nn.Linear(EMB_DIM, EMB_BN), nn.Dropout(0.5))
        self.ecg_proj = nn.Sequential(nn.Linear(ECG_TOK_IN, d_model), nn.LayerNorm(d_model))
        self.imu_proj = nn.Sequential(nn.Linear(IMU_DIM, d_model), nn.LayerNorm(d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout=0.3,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.to_latent = nn.Linear(d_model * 2, LATENT)
        self.from_latent = nn.Linear(LATENT, d_model * 2)
        self.ecg_dec = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, ECG_TOK_IN))
        self.imu_dec = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, IMU_DIM))

    def forward(self, b, mask):
        ecg_in = torch.cat([self.ecg_bn(b["ecg_emb"]), b["ecg_aux"]], -1)
        imu_in = b["imu"]
        et = self.ecg_proj(ecg_in) * mask[:, 0:1]
        it = self.imu_proj(imu_in) * mask[:, 1:2]
        tok = torch.stack([et, it], 1)
        pad = (mask < 0.5)
        allm = pad.all(1, keepdim=True); pad = pad & ~allm.expand_as(pad)
        ctx = self.tr(tok, src_key_padding_mask=pad)
        z = self.to_latent(ctx.reshape(ctx.size(0), -1))
        h = self.from_latent(z).reshape(ctx.size(0), 2, -1)
        rec_ecg = self.ecg_dec(h[:, 0]); rec_imu = self.imu_dec(h[:, 1])
        return (ecg_in, imu_in), (rec_ecg, rec_imu), mask


def recon_error(inp, rec, mask):
    (ecg_in, imu_in), (rec_ecg, rec_imu) = inp, rec
    e = ((ecg_in - rec_ecg) ** 2).mean(-1) * mask[:, 0]
    i = ((imu_in - rec_imu) ** 2).mean(-1) * mask[:, 1]
    return (e + i) / (mask[:, 0] + mask[:, 1] + 1e-6)


def auroc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def to_dev(feats, idx):
    return {k: torch.tensor(feats[k][idx]).to(DEVICE) for k in ("ecg_emb", "ecg_aux", "imu")}


# ─── 통제 perturbation: 정상 분포 밖 이탈 생성 (합성 — 민감도/배관 측정용) ───────
def make_perturbations(feats, base_idx, rng, sigma):
    """held-out 정상 윈도우를 복제해 '분포 밖' 방향으로 교란(강도=sigma σ). 세 유형:
      ecg_dev  : ECG 임베딩에 가우시안 + aux 비정상화(es↑rel↑) — 'ECG 이상' 모사
      imu_dev  : IMU에 비활동성 이상(낙상류 충격) 모사
      joint_dev: 둘 다
    표준화 공간이라 sigma=표준편차 배수. 합성 → 민감도·배관 측정(응급효능 아님).
    """
    out = {}
    emb = feats["ecg_emb"][base_idx].copy()
    aux = feats["ecg_aux"][base_idx].copy()
    imu = feats["imu"][base_idx].copy()
    e_emb = emb + rng.normal(0, sigma, emb.shape).astype(np.float32)
    e_aux = aux.copy(); e_aux[:, 0] += sigma; e_aux[:, 1] += sigma
    out["ecg_dev"] = (e_emb, e_aux, imu.copy())
    i_imu = imu.copy(); i_imu[:, [2, 4, 11]] += sigma
    out["imu_dev"] = (emb.copy(), aux.copy(), i_imu)
    out["joint_dev"] = (e_emb.copy(), e_aux.copy(), i_imu.copy())
    return out


def score_arrays(model, emb, aux, imu, mv):
    model.eval()
    with torch.no_grad():
        b = {"ecg_emb": torch.tensor(emb).to(DEVICE),
             "ecg_aux": torch.tensor(aux).to(DEVICE),
             "imu": torch.tensor(imu).to(DEVICE)}
        mask = torch.tensor(np.tile(mv, (len(emb), 1)), dtype=torch.float32, device=DEVICE)
        inp, rec, m = model(b, mask)
        return recon_error(inp, rec, m).cpu().numpy()


def fit_eval(ModelCls, feats, tr_idx, te_idx, te_act, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = ModelCls().to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
    tr = tr_idx.copy()
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(tr)
        for i in range(0, len(tr), BS):
            bi = tr[i:i + BS]
            b = to_dev(feats, bi)
            mask = torch.ones(len(bi), 2, device=DEVICE)
            inp, rec, m = model(b, mask)
            loss = recon_error(inp, rec, m).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

    sc_norm = score_arrays(model, feats["ecg_emb"][te_idx], feats["ecg_aux"][te_idx],
                           feats["imu"][te_idx], [1, 1])
    is_rest = np.isin(te_act, REST_ACTS); is_active = np.isin(te_act, ACTIVE_ACTS)
    rest_sc = sc_norm[is_rest]; active_sc = sc_norm[is_active]
    all_norm = sc_norm

    # 이탈: 다단계 sigma × 3유형, AUROC(이탈 vs 전체정상)
    pert_rng = np.random.default_rng(seed + 1000)
    dev = {}
    for sigma in PERTURB_SIGMAS:
        pset = make_perturbations(feats, te_idx, pert_rng, sigma)
        for dname, (emb, aux, imu) in pset.items():
            sc = score_arrays(model, emb, aux, imu, [1, 1])
            dev[f"{dname}_s{sigma:g}"] = auroc(sc, all_norm)

    return {
        # 운동관용비: active 재구성오차 / rest 재구성오차 (1.0=완전관용, ↑=운동을 이탈로 봄)
        "tol_active": float(active_sc.mean() / (rest_sc.mean() + 1e-9)),
        "dev_auroc": dev,
    }


def robust_standardize(feats, fit_idx):
    """전체 정상(fit_idx) 기준. IMU는 log1p + 1~99분위 클리핑 후 표준화(긴 꼬리 교정)."""
    # IMU: 비음수 가정 → log1p, 분위 클리핑
    imu = feats["imu"]
    lo = np.percentile(imu[fit_idx], 1, axis=0)
    hi = np.percentile(imu[fit_idx], 99, axis=0)
    imu_c = np.clip(imu, lo, hi)
    imu_l = np.log1p(imu_c - lo + 1e-6)
    mu = imu_l[fit_idx].mean(0, keepdims=True); sd = imu_l[fit_idx].std(0, keepdims=True) + 1e-6
    feats["imu"] = ((imu_l - mu) / sd).astype(np.float32)
    # ECG emb / aux: 표준 z-score (전체 정상 기준)
    for key in ("ecg_emb", "ecg_aux"):
        m = feats[key][fit_idx].mean(0, keepdims=True); s = feats[key][fit_idx].std(0, keepdims=True) + 1e-6
        feats[key] = ((feats[key] - m) / s).astype(np.float32)


def _ms(vals):
    a = np.array(vals, float)
    return f"{a.mean():.3f}±{a.std():.3f}"


REST_ACTS = ACTIVE_ACTS = None    # main에서 데이터셋별 설정(전역)


def split_indices(cfg, subj):
    """ptt=분할정본 / dalia=즉석 피험자분할(고정 seed, ~70/30)."""
    if cfg["split"] is not None:
        sp = json.loads(cfg["split"].read_text(encoding="utf-8"))
        return (np.where(np.isin(subj, sp["train"]))[0],
                np.where(np.isin(subj, sp["test"]))[0], sp.get("seed", 42))
    subs = sorted(set(subj.tolist()))
    rng = np.random.default_rng(42); rng.shuffle(subs)
    n_tr = int(round(len(subs) * 0.7))
    tr_s, te_s = set(subs[:n_tr]), set(subs[n_tr:])
    return (np.where(np.isin(subj, list(tr_s)))[0],
            np.where(np.isin(subj, list(te_s)))[0], 42)


def main():
    global REST_ACTS, ACTIVE_ACTS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptt", choices=list(DATASETS.keys()))
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    REST_ACTS = cfg["rest"]; ACTIVE_ACTS = cfg["active"]

    feats, subj = load(cfg["path"])
    tr_idx, te_idx, sp_seed = split_indices(cfg, subj)
    # 학습 = train 피험자의 정상(rest+active 전부; 운동 관용 학습)
    te_act = feats["activity"][te_idx]
    n_rest = int(np.isin(te_act, REST_ACTS).sum()); n_act = int(np.isin(te_act, ACTIVE_ACTS).sum())

    robust_standardize(feats, tr_idx)

    print(f"[{args.dataset}] 이상탐지(질문1·운동관용) — {len(SEEDS)}-seed {SEEDS}, perturb σ {PERTURB_SIGMAS}")
    print(f"학습=train 전체정상 {len(tr_idx)}w / test {len(te_idx)}w (rest {n_rest} / active {n_act})")
    print(f"rest={REST_ACTS} active={ACTIVE_ACTS}")
    print("좋은 모델 = 운동관용(active/rest→1.0) AND 이탈민감(AUROC↑). 선험우위 가정 안 함.\n")

    agg = {}
    for name, Cls in [("late_fusion", LateFusionAE), ("cross_attn", CrossAttnAE)]:
        t0 = time.time()
        runs = [fit_eval(Cls, feats, tr_idx, te_idx, te_act, s) for s in SEEDS]
        agg[name] = {
            "tol_active": [r["tol_active"] for r in runs],
            "dev_auroc": {k: [r["dev_auroc"][k] for r in runs] for k in runs[0]["dev_auroc"]},
        }
        print(f"[{name}]  ({time.time()-t0:.0f}s, {len(SEEDS)} seeds)")
        print(f"  운동관용비 active/rest(↓ 좋음) = {_ms(agg[name]['tol_active'])}")
        for dn in agg[name]["dev_auroc"]:
            print(f"    AUROC {dn:<14} {_ms(agg[name]['dev_auroc'][dn])}")
        print()

    print("=== late_fusion vs cross_attn 대조 ===")
    print(f"  active/rest 관용비(↓ 좋음):  late {_ms(agg['late_fusion']['tol_active'])}  |  cross {_ms(agg['cross_attn']['tol_active'])}")
    for dn in agg["late_fusion"]["dev_auroc"]:
        print(f"  AUROC {dn:<14}: late {_ms(agg['late_fusion']['dev_auroc'][dn])}  |  cross {_ms(agg['cross_attn']['dev_auroc'][dn])}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"anomaly_{args.dataset}_report.json"
    out.write_text(json.dumps({"dataset": args.dataset, "split_seed": sp_seed, "seeds": SEEDS,
                               "perturb_sigmas": PERTURB_SIGMAS, "epochs": EPOCHS,
                               "rest_acts": REST_ACTS, "active_acts": ACTIVE_ACTS,
                               "design": "Q1 motion-tolerant one-class AE; train=all-normal; "
                                         "robust IMU std(log1p+clip); multi-sigma perturbation; 3-seed",
                               "agg": agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
