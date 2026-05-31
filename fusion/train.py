"""융합 모델 학습 — 재사용 루프 (concat / gated / cross_attn 공통).

사용:
    python -m fusion.train --model concat     --epochs 60
    python -m fusion.train --model gated      --epochs 80
    python -m fusion.train --model cross_attn --epochs 80

데이터: P2_DATA_DIR/synthetic/p2_synth_{version}_{split}.npz (외부 데이터 디렉토리)
출력  : P2_CKPT_DIR/p2_{model}_{run_id}/  {best_model.pt, last_model.pt, train_log.csv}
        기본 출력은 레포 내 fusion/artifacts/checkpoints/ (재생성 가능, gitignore).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.dataset import make_loaders
from fusion.model import ConcatMLP, GatedFusionModel, CrossModalAttentionFusion
from fusion.schema import CLASS_NAMES, NUM_CLASSES

DATA_DIR  = Path(os.environ.get("P2_DATA_DIR", "D:/WidU_multimodal_fusion")) / "synthetic"
CKPT_ROOT = Path(os.environ.get("P2_CKPT_DIR", str(ROOT / "fusion" / "artifacts" / "checkpoints")))
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 평가 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1(preds: np.ndarray, labels: np.ndarray, n_cls: int = NUM_CLASSES) -> float:
    f1s = []
    for c in range(n_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate(model, loader, device, no_emb: bool = False) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batch = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if no_emb:
            batch["ecg_emb"] = torch.zeros_like(batch["ecg_emb"])
        if isinstance(model, (GatedFusionModel, CrossModalAttentionFusion)):
            out  = model(batch)
            loss = model.loss(batch, out)
            logits = out["logits"]
        else:
            logits = model(batch)
            loss   = nn.CrossEntropyLoss()(logits, batch["label"])

        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        total_loss += loss.item()
        n_batch += 1

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    mf1    = macro_f1(preds, labels)

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = ((preds == c) & (labels == c)).sum()
        denom = (labels == c).sum()
        per_class[name] = float(tp / denom) if denom > 0 else 0.0

    return {
        "loss":      total_loss / max(n_batch, 1),
        "macro_f1":  mf1,
        "per_class": per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    run_id   = f"{args.model}_{int(time.time()) % 100000}"
    ckpt_dir = CKPT_ROOT / f"p2_{run_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")
    print(f"Checkpoint: {ckpt_dir}")

    train_loader, val_loader, test_loader = make_loaders(
        DATA_DIR,
        batch_size=args.batch_size,
        modality_dropout_p=args.dropout_mod,
        version=args.dataset_version,
    )

    # ── 모델 선택 ──
    if args.model == "concat":
        model = ConcatMLP(hidden_dims=(512, 256, 128),
                          dropout_p=args.dropout).to(DEVICE)
    elif args.model == "gated":
        model = GatedFusionModel(fusion_hidden=(256, 128),
                                 dropout=args.dropout,
                                 aux_loss_weight=args.aux_weight,
                                 reliability_mode=args.reliability_mode,
                                 gate_input_norm=True,
                                 fusion_level=args.fusion_level,
                                 gate_mode=args.gate_mode,
                                 temperature=args.temperature,
                                 emb_bottleneck=getattr(args, "emb_bottleneck", 0)).to(DEVICE)
    elif args.model == "cross_attn":
        model = CrossModalAttentionFusion(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            aux_loss_weight=args.aux_weight,
            emb_bottleneck=getattr(args, "emb_bottleneck", 16),
        ).to(DEVICE)
    else:
        raise ValueError(f"--model must be 'concat', 'gated', or 'cross_attn', got '{args.model}'")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    no_emb = getattr(args, "no_embedding", False)
    print(f"Model: {args.model}  |  파라미터: {n_params:,}  |  no_embedding={no_emb}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    log_path = ckpt_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1"] + CLASS_NAMES

    best_f1  = -1.0
    best_ep  = -1

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss, n_steps = 0.0, 0

            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                if no_emb:
                    batch["ecg_emb"] = torch.zeros_like(batch["ecg_emb"])
                optimizer.zero_grad()

                if isinstance(model, (GatedFusionModel, CrossModalAttentionFusion)):
                    out  = model(batch)
                    loss = model.loss(batch, out)
                else:
                    logits = model(batch)
                    loss   = nn.CrossEntropyLoss()(logits, batch["label"])

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1

            scheduler.step()
            train_loss = epoch_loss / n_steps

            val_metrics = evaluate(model, val_loader, DEVICE, no_emb=no_emb)

            row = {
                "epoch":        epoch,
                "train_loss":   f"{train_loss:.4f}",
                "val_loss":     f"{val_metrics['loss']:.4f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.4f}",
                **{n: f"{val_metrics['per_class'][n]:.3f}" for n in CLASS_NAMES},
            }
            writer.writerow(row)
            f.flush()

            is_best = val_metrics["macro_f1"] > best_f1
            if is_best:
                best_f1 = val_metrics["macro_f1"]
                best_ep = epoch
                torch.save({"epoch": epoch,
                            "model_state": model.state_dict(),
                            "val_macro_f1": best_f1,
                            "args": vars(args)},
                           ckpt_dir / "best_model.pt")

            if epoch % args.log_every == 0 or epoch == args.epochs:
                pc_str = "  ".join(
                    f"{n[:3]}={val_metrics['per_class'][n]:.3f}"
                    for n in CLASS_NAMES
                )
                print(f"[{epoch:03d}/{args.epochs}] "
                      f"train={train_loss:.4f}  "
                      f"val_loss={val_metrics['loss']:.4f}  "
                      f"val_F1={val_metrics['macro_f1']:.4f}"
                      f"{'*' if is_best else ' '}  |  {pc_str}")

    torch.save({"epoch": args.epochs,
                "model_state": model.state_dict(),
                "args": vars(args)},
               ckpt_dir / "last_model.pt")

    # ── 테스트 평가 ──
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, DEVICE, no_emb=no_emb)

    print(f"\n=== 테스트 결과 (best epoch={best_ep}, val_F1={best_f1:.4f}) ===")
    print(f"test macro-F1: {test_metrics['macro_f1']:.4f}")
    for name in CLASS_NAMES:
        print(f"  {name:<20s}: recall={test_metrics['per_class'][name]:.3f}")
    print(f"체크포인트: {ckpt_dir}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="P2 융합 모델 학습")
    ap.add_argument("--model",       default="gated", choices=["concat", "gated", "cross_attn"])
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch-size",  type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--dropout",     type=float, default=0.3,
                    help="expert/MLP dropout 비율")
    ap.add_argument("--dropout-mod", type=float, default=0.15,
                    help="modality dropout 비율 (학습 중 결측 시뮬레이션)")
    ap.add_argument("--aux-weight",  type=float, default=0.3,
                    help="unimodal 보조손실 가중치 (gated only)")
    ap.add_argument("--reliability-mode", default="feature",
                    choices=["none", "feature", "hard_mult"],
                    help="reliability 통합 방식 (gated only)")
    ap.add_argument("--fusion-level", default="feature",
                    choices=["feature", "logit"],
                    help="feature: feature-weighted sum+MLP, logit: MoE probability mixing")
    ap.add_argument("--gate-mode", default="learned",
                    choices=["learned", "conf_routed"],
                    help="learned: gate_net MLP, conf_routed: softmax(conf/τ) 직접 라우팅")
    ap.add_argument("--temperature", type=float, default=0.15,
                    help="conf_routed 모드 softmax 온도 (낮을수록 winner-take-all)")
    ap.add_argument("--log-every",   type=int,   default=10)
    ap.add_argument("--dataset-version", default="v1",
                    help="데이터셋 버전 (v1=독립샘플링, v2_mvn=MVN다변량, vf=누출없음)")
    ap.add_argument("--no-embedding", action="store_true",
                    help="raw 768 ECG 임베딩 zero-out — ecg_aux(10) 점수만 사용. "
                         "임베딩 과적합 진단용.")
    ap.add_argument("--emb-bottleneck", type=int, default=0,
                    help="ECG 임베딩 병목 차원 (0=기존 768→256→128, >0=768→N→128 + dropout0.5). "
                         "외울 용량 줄여 과적합 vs cardiac 신호 tradeoff.")
    # ── Cross-Modal Attention 전용 인자 ──
    ap.add_argument("--d-model",  type=int, default=128,
                    help="Transformer hidden dim (cross_attn)")
    ap.add_argument("--n-heads",  type=int, default=4,
                    help="Attention heads (cross_attn)")
    ap.add_argument("--n-layers", type=int, default=2,
                    help="Transformer encoder layers (cross_attn)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
