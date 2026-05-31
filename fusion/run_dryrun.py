"""proxy 드라이런 — 3 모델 학습 + 2D 비교(confounder-FP × 결측강건성) + attention XAI.

목적(정직한 기대): 필드 전에 파이프라인을 끝까지 돌려 ① 배관 작동 ② 바닥값 ③ 구조 검증.
산출 fusion_synth_v1.pt = 예행용·버림(비배포). 조건부 독립 proxy에선 attention ≈ concat/gated
동급 예상이며, confounder-FP·결측강건성도 ECG 임베딩 용량 정합 시 아키텍처-비특이로 측정됨
(용량 readout). 실 변별은 시간정렬 실데이터(필드), 고유가치는 XAI로 한정된다.

사용:
    python -m fusion.run_dryrun --epochs 60
    python -m fusion.run_dryrun --epochs 60 --build-data   # 프록시 재조립(자체완결)
출력: fusion/artifacts/ {fusion_synth_v1.pt, dryrun_report.txt, dryrun_results.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from fusion.dataset import (P2Dataset, build_proxy_dataset, make_loaders,
                            P1_CACHE_DIR)
from fusion.model import ConcatMLP, GatedFusionModel, CrossModalAttentionFusion
from fusion.metrics import evaluate_2d, format_2d_table, macro_f1, model_logits
from fusion.confounders import build_all
from fusion import xai
from fusion.schema import CLASS_NAMES

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS = ROOT / "fusion" / "artifacts"
DATA_ROOT = Path(os.environ.get("P2_DATA_DIR", "data"))

MODEL_FACTORY = {
    "concat":     lambda: ConcatMLP(hidden_dims=(512, 256, 128), dropout_p=0.3),
    "gated":      lambda: GatedFusionModel(fusion_hidden=(256, 128), dropout=0.3,
                                           aux_loss_weight=0.3, reliability_mode="feature"),
    "cross_attn": lambda: CrossModalAttentionFusion(d_model=128, n_heads=4, n_layers=2,
                                                    dropout=0.3, aux_loss_weight=0.3,
                                                    emb_bottleneck=16),
}


def fit(model, train_loader, val_loader, epochs: int, lr: float, log_every: int = 15):
    """compact 학습 루프 — best(val macro-F1) 모델 반환."""
    model = model.to(DEVICE)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    ce = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            opt.zero_grad()
            if isinstance(model, (GatedFusionModel, CrossModalAttentionFusion)):
                out = model(batch); loss = model.loss(batch, out)
            else:
                loss = ce(model(batch), batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # val
        model.eval()
        P, L = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                P.append(model_logits(model, batch).argmax(-1).cpu().numpy())
                L.append(batch["label"].cpu().numpy())
        f1 = macro_f1(np.concatenate(P), np.concatenate(L))
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % log_every == 0 or ep == epochs:
            print(f"    [{ep:03d}/{epochs}] val_macroF1={f1:.4f}{'  *best' if f1 == best_f1 else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def ensure_data(version: str, build: bool, n_per_class: int, seed: int) -> Path:
    """프록시 데이터 확보 — 기존 synthetic 재사용, 없거나 --build-data면 조립(자체완결)."""
    if build:
        out = ARTIFACTS / "synthetic"
        print(f"프록시 재조립 → {out} (version={version}, n/class={n_per_class})")
        build_proxy_dataset(out, n_per_class=n_per_class, seed=seed, version=version,
                            cache_dir=P1_CACHE_DIR)
        return out
    ext = DATA_ROOT / "synthetic"
    if (ext / f"p2_synth_{version}_test.npz").exists():
        print(f"기존 프록시 재사용 → {ext} (version={version})")
        return ext
    out = ARTIFACTS / "synthetic"
    print(f"기존 프록시 없음 → 조립 → {out}")
    build_proxy_dataset(out, n_per_class=n_per_class, seed=seed, version=version,
                        cache_dir=P1_CACHE_DIR)
    return out


def main():
    ap = argparse.ArgumentParser(description="P2-B proxy 드라이런 — 2D 비교 + XAI")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dropout-mod", type=float, default=0.15,
                    help="학습 중 modality dropout (결측 강건성 학습)")
    ap.add_argument("--version", default="vf", help="데이터셋 버전 (vf=누출없음)")
    ap.add_argument("--build-data", action="store_true", help="프록시 재조립(자체완결)")
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="concat,gated,cross_attn")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}  |  epochs={args.epochs}  |  version={args.version}")

    # ── 데이터 ──
    data_dir = ensure_data(args.version, args.build_data, args.n_per_class, args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        data_dir, batch_size=args.batch_size,
        modality_dropout_p=args.dropout_mod, version=args.version)

    # ── 실데이터 앵커 confounder ──
    print("\n실데이터 앵커 confounder 구성 (비순환):")
    conf_sets = build_all(seed=args.seed)
    for cs in conf_sets:
        print(f"  - {cs.name}: n={cs.n}  | {cs.realness}")

    # ── 3 모델 학습 + 2D 평가 ──
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results, trained, val_f1s = {}, {}, {}
    for name in model_names:
        print(f"\n=== {name} 학습 ===")
        torch.manual_seed(args.seed)               # 동일 init seed
        model = MODEL_FACTORY[name]()
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        model, vf1 = fit(model, train_loader, val_loader, args.epochs, args.lr)
        val_f1s[name] = vf1; trained[name] = model
        results[name] = evaluate_2d(model, test_loader, conf_sets, DEVICE)
        print(f"  params={n_par:,}  val_F1={vf1:.4f}  ({time.time()-t0:.0f}s)")

    # ── 2D 표 ──
    table = format_2d_table(results)
    print("\n" + table)

    # ── attention XAI (cross_attn) ──
    xai_block = ""
    if "cross_attn" in trained:
        print("\n[attention XAI — cross_attn, confounder 케이스 쏠림]")
        xai_block = xai.confounder_attention_report(trained["cross_attn"], conf_sets, DEVICE)
        print(xai_block)

    # ── 산출물 저장 ──
    report_path = ARTIFACTS / "dryrun_report.txt"
    json_path   = ARTIFACTS / "dryrun_results.json"
    ckpt_path   = ARTIFACTS / "fusion_synth_v1.pt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("P2-B proxy 드라이런 리포트 (예행·버림, 비배포)\n")
        f.write(f"version={args.version} epochs={args.epochs} seed={args.seed} device={DEVICE}\n\n")
        f.write("confounder 앵커(비순환):\n")
        for cs in conf_sets:
            f.write(f"  - {cs.name}: n={cs.n} | {cs.realness}\n")
        f.write("\n" + table + "\n")
        if xai_block:
            f.write("\n[attention XAI — cross_attn]\n" + xai_block + "\n")
        f.write("\n해석 지침 (예행 — 수치는 버린다):\n")
        f.write(" - clean 정확도는 proxy·미튜닝 베이스라인 기준이라 필드 전 우열 판정 불가.\n")
        f.write(" - confounder-FP 차이는 융합 아키텍처가 아니라 ECG 임베딩 용량의 readout이다(용량-정합 측정):\n")
        f.write("   동일 임베딩 병목을 concat·gated에 주면 chronic-FP가 cross_attn 수준으로 수렴 → 아키텍처-비특이.\n")
        f.write(" - apnea류 joint('desat ∧ HR 비상승')는 조건부 독립 proxy에 부재 → 어느 모델도 학습 불가.\n")
        f.write("   이 축은 실 시간정렬 데이터(train_on_field.py) 필요성의 실측 근거다.\n")

    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean({"args": vars(args), "val_macro_f1": val_f1s,
                          "results": results,
                          "confounders": [{"name": c.name, "n": c.n,
                                           "emergency_class": int(c.emergency_class),
                                           "realness": c.realness} for c in conf_sets]}),
                  f, ensure_ascii=False, indent=2)

    torch.save({"note": "proxy dry-run — disposable, not for deployment",
                "version": args.version, "args": vars(args),
                "val_macro_f1": val_f1s,
                "model_states": {n: m.state_dict() for n, m in trained.items()},
                "class_names": CLASS_NAMES}, ckpt_path)

    print(f"\n저장: {ckpt_path.name}, {report_path.name}, {json_path.name}  (→ {ARTIFACTS})")


if __name__ == "__main__":
    main()
