"""DaLiA 3-modal (ECG+IMU+PPG) anomaly detection — last available combination: ECG+PPG joint.

SSOT 12.10: ECG+IMU 2-modal attention = no robust advantage (ptt+dalia 3-seed). User asked to
try other combos. DaLiA has wrist BVP (PPG 64Hz) -> ECG+PPG and ECG+IMU+PPG testable here.

Prior prediction (NOT assumed — close by measurement): ECG & PPG = two views of the same
heartbeat (electrical vs peripheral pulse) = redundant, not complementary -> PPG largely
derivable from ECG -> expect no attention advantage, same as ECG+IMU. Measure to confirm/refute.

Design (Q1 motion-tolerant, same as anomaly_ptt): train all-normal one-class AE with 3 tokens;
metrics = motion tolerance (active/rest recon ratio) per ablation + controlled perturbation
AUROC. late_fusion vs cross_attn, 3-seed.
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
DALIA = Path(os.environ.get("P2_DATA_DIR", "data")) / "p1_cache" / "dalia_p1.npz"
ARTIFACTS = ROOT / "fusion" / "artifacts"

REST = ["baseline", "clean_baseline", "lunch", "working", "driving"]
ACTIVE = ["stairs", "soccer", "cycling", "walking"]

EPOCHS, LR, BS = 120, 1e-3, 64
SEEDS = [42, 7, 123]
PERTURB_SIGMAS = [1.0, 2.0, 3.0]
EMB_BN, LATENT = 16, 16
ECG_TOK_IN = EMB_BN + 4
PPG_DIM = 8

ABLATIONS = {"all": [1, 1, 1], "ecg_ppg": [1, 0, 1], "ecg_imu": [1, 1, 0],
             "ecg_only": [1, 0, 0], "ppg_only": [0, 0, 1], "imu_only": [0, 1, 0]}


def load():
    d = np.load(DALIA, allow_pickle=True)
    aux = np.stack([d["emergency_score"], d["reliability"],
                    d["hr_bpm"] / 100.0, d["rhythm_regularity"]], axis=1).astype(np.float32)
    return ({"ecg_emb": d["embedding"].astype(np.float32), "ecg_aux": aux,
             "imu": d["imu_feat"].astype(np.float32), "ppg": d["ppg_feat"].astype(np.float32),
             "activity": d["activity"]}, d["subject"])


class LateFusionAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_bn = nn.Sequential(nn.Linear(EMB_DIM, EMB_BN), nn.Dropout(0.5))
        self.ecg_enc = nn.Sequential(nn.Linear(ECG_TOK_IN, 32), nn.GELU(), nn.Linear(32, 16))
        self.imu_enc = nn.Sequential(nn.Linear(IMU_DIM, 32), nn.GELU(), nn.Linear(32, 16))
        self.ppg_enc = nn.Sequential(nn.Linear(PPG_DIM, 32), nn.GELU(), nn.Linear(32, 16))
        self.to_latent = nn.Linear(48, LATENT)
        self.from_latent = nn.Linear(LATENT, 48)
        self.ecg_dec = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, ECG_TOK_IN))
        self.imu_dec = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, IMU_DIM))
        self.ppg_dec = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, PPG_DIM))

    def forward(self, b, mask):
        ecg_in = torch.cat([self.ecg_bn(b["ecg_emb"]), b["ecg_aux"]], -1)
        imu_in, ppg_in = b["imu"], b["ppg"]
        e = self.ecg_enc(ecg_in) * mask[:, 0:1]
        i = self.imu_enc(imu_in) * mask[:, 1:2]
        p = self.ppg_enc(ppg_in) * mask[:, 2:3]
        h = self.from_latent(self.to_latent(torch.cat([e, i, p], -1)))
        return ((ecg_in, imu_in, ppg_in),
                (self.ecg_dec(h[:, :16]), self.imu_dec(h[:, 16:32]), self.ppg_dec(h[:, 32:48])), mask)


class CrossAttnAE(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.ecg_bn = nn.Sequential(nn.Linear(EMB_DIM, EMB_BN), nn.Dropout(0.5))
        self.ecg_proj = nn.Sequential(nn.Linear(ECG_TOK_IN, d_model), nn.LayerNorm(d_model))
        self.imu_proj = nn.Sequential(nn.Linear(IMU_DIM, d_model), nn.LayerNorm(d_model))
        self.ppg_proj = nn.Sequential(nn.Linear(PPG_DIM, d_model), nn.LayerNorm(d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout=0.3,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)
        self.to_latent = nn.Linear(d_model * 3, LATENT)
        self.from_latent = nn.Linear(LATENT, d_model * 3)
        self.ecg_dec = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, ECG_TOK_IN))
        self.imu_dec = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, IMU_DIM))
        self.ppg_dec = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, PPG_DIM))

    def forward(self, b, mask):
        ecg_in = torch.cat([self.ecg_bn(b["ecg_emb"]), b["ecg_aux"]], -1)
        imu_in, ppg_in = b["imu"], b["ppg"]
        et = self.ecg_proj(ecg_in) * mask[:, 0:1]
        it = self.imu_proj(imu_in) * mask[:, 1:2]
        pt = self.ppg_proj(ppg_in) * mask[:, 2:3]
        tok = torch.stack([et, it, pt], 1)
        pad = (mask < 0.5); allm = pad.all(1, keepdim=True); pad = pad & ~allm.expand_as(pad)
        ctx = self.tr(tok, src_key_padding_mask=pad)
        h = self.from_latent(self.to_latent(ctx.reshape(ctx.size(0), -1))).reshape(ctx.size(0), 3, -1)
        return ((ecg_in, imu_in, ppg_in),
                (self.ecg_dec(h[:, 0]), self.imu_dec(h[:, 1]), self.ppg_dec(h[:, 2])), mask)


def recon_error(inp, rec, mask):
    (ecg_in, imu_in, ppg_in), (re, ri, rp) = inp, rec
    e = ((ecg_in - re) ** 2).mean(-1) * mask[:, 0]
    i = ((imu_in - ri) ** 2).mean(-1) * mask[:, 1]
    p = ((ppg_in - rp) ** 2).mean(-1) * mask[:, 2]
    return (e + i + p) / (mask.sum(-1) + 1e-6)


def auroc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score(model, feats, idx, mv, override=None):
    model.eval()
    with torch.no_grad():
        b = {}
        for k in ("ecg_emb", "ecg_aux", "imu", "ppg"):
            arr = override[k] if (override and k in override) else feats[k][idx]
            b[k] = torch.tensor(arr).to(DEVICE)
        mask = torch.tensor(np.tile(mv, (len(idx), 1)), dtype=torch.float32, device=DEVICE)
        inp, rec, m = model(b, mask)
        return recon_error(inp, rec, m).cpu().numpy()


def perturb_set(feats, idx, rng, sigma):
    emb = feats["ecg_emb"][idx].copy(); aux = feats["ecg_aux"][idx].copy(); ppg = feats["ppg"][idx].copy()
    e_emb = emb + rng.normal(0, sigma, emb.shape).astype(np.float32)
    e_aux = aux.copy(); e_aux[:, 0] += sigma; e_aux[:, 1] += sigma
    p_dev = ppg.copy(); p_dev[:, [2, 3, 4]] += sigma
    return {"ecg_dev": {"ecg_emb": e_emb, "ecg_aux": e_aux}, "ppg_dev": {"ppg": p_dev},
            "ecgppg_dev": {"ecg_emb": e_emb, "ecg_aux": e_aux, "ppg": p_dev}}


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
            b = {k: torch.tensor(feats[k][bi]).to(DEVICE) for k in ("ecg_emb", "ecg_aux", "imu", "ppg")}
            inp, rec, m = model(b, torch.ones(len(bi), 3, device=DEVICE))
            loss = recon_error(inp, rec, m).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    is_rest = np.isin(te_act, REST); is_active = np.isin(te_act, ACTIVE)
    out = {"tol_active": {}, "dev_auroc": {}}
    for ab, mv in ABLATIONS.items():
        sc = score(model, feats, te_idx, mv)
        out["tol_active"][ab] = float(sc[is_active].mean() / (sc[is_rest].mean() + 1e-9))
    prng = np.random.default_rng(seed + 1000)
    for sigma in PERTURB_SIGMAS:
        ps = perturb_set(feats, te_idx, prng, sigma)
        base = score(model, feats, te_idx, [1, 1, 1])
        for dn, ov in ps.items():
            out["dev_auroc"][f"{dn}_s{sigma:g}"] = auroc(score(model, feats, te_idx, [1, 1, 1], override=ov), base)
    return out


def robust_standardize(feats, fit_idx):
    for key, nonneg in [("imu", True), ("ppg", False)]:
        x = feats[key]
        lo = np.percentile(x[fit_idx], 1, axis=0); hi = np.percentile(x[fit_idx], 99, axis=0)
        xc = np.clip(x, lo, hi)
        if nonneg:
            xc = np.log1p(xc - lo + 1e-6)
        mu = xc[fit_idx].mean(0, keepdims=True); sd = xc[fit_idx].std(0, keepdims=True) + 1e-6
        feats[key] = ((xc - mu) / sd).astype(np.float32)
    for key in ("ecg_emb", "ecg_aux"):
        m = feats[key][fit_idx].mean(0, keepdims=True); s = feats[key][fit_idx].std(0, keepdims=True) + 1e-6
        feats[key] = ((feats[key] - m) / s).astype(np.float32)


def _ms(v):
    a = np.array(v, float); return f"{a.mean():.3f}+-{a.std():.3f}"


def split_idx(subj):
    subs = sorted(set(subj.tolist())); rng = np.random.default_rng(42); rng.shuffle(subs)
    n = int(round(len(subs) * 0.7))
    return (np.where(np.isin(subj, subs[:n]))[0], np.where(np.isin(subj, subs[n:]))[0])


def main():
    feats, subj = load()
    tr_idx, te_idx = split_idx(subj)
    te_act = feats["activity"][te_idx]
    robust_standardize(feats, tr_idx)
    print(f"[dalia 3modal ECG+IMU+PPG] {len(SEEDS)}-seed, perturb sigma {PERTURB_SIGMAS}")
    print(f"train all-normal {len(tr_idx)}w / test {len(te_idx)}w")
    print("last combo (ECG+PPG). prior: redundant modal -> no advantage. close by measurement.\n")
    agg = {}
    for name, Cls in [("late_fusion", LateFusionAE), ("cross_attn", CrossAttnAE)]:
        t0 = time.time()
        runs = [fit_eval(Cls, feats, tr_idx, te_idx, te_act, s) for s in SEEDS]
        agg[name] = {"tol_active": {ab: [r["tol_active"][ab] for r in runs] for ab in ABLATIONS},
                     "dev_auroc": {k: [r["dev_auroc"][k] for r in runs] for k in runs[0]["dev_auroc"]}}
        print(f"[{name}]  ({time.time()-t0:.0f}s)")
        for ab in ABLATIONS:
            print(f"    tol {ab:<9} {_ms(agg[name]['tol_active'][ab])}")
        for dn in agg[name]["dev_auroc"]:
            print(f"    AUROC {dn:<14} {_ms(agg[name]['dev_auroc'][dn])}")
        print()
    print("=== late vs cross (key: ecg_ppg) ===")
    for ab in ["ecg_ppg", "all", "ecg_imu"]:
        print(f"  tol[{ab}]: late {_ms(agg['late_fusion']['tol_active'][ab])}  |  cross {_ms(agg['cross_attn']['tol_active'][ab])}")
    for dn in agg["late_fusion"]["dev_auroc"]:
        print(f"  AUROC {dn:<14}: late {_ms(agg['late_fusion']['dev_auroc'][dn])}  |  cross {_ms(agg['cross_attn']['dev_auroc'][dn])}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "anomaly_dalia3_report.json"
    out.write_text(json.dumps({"dataset": "dalia_3modal", "seeds": SEEDS, "perturb_sigmas": PERTURB_SIGMAS,
                               "epochs": EPOCHS, "modalities": "ECG+IMU+PPG", "ablations": ABLATIONS,
                               "agg": agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
