"""기존 체크포인트 로드 → motion confounder 포함 2D 평가 재실행.

이미 80-에폭 훈련된 fusion_synth_v1.pt 가 있으므로 재훈련 없이
evaluate_2d (confounder-FP × 결측강건성) 만 빠르게 다시 돌린다.
nstdb_motion.npz 가 추가된 뒤 최초 실행 → motion-FP 열 완성.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch

from fusion.dataset import make_loaders, P1_CACHE_DIR
from fusion.model import ConcatMLP, GatedFusionModel, CrossModalAttentionFusion
from fusion.metrics import evaluate_2d, format_2d_table, model_logits, macro_f1
from fusion.confounders import build_all
from fusion import xai

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS = ROOT / "fusion" / "artifacts"
DATA_ROOT = Path(os.environ.get("P2_DATA_DIR", r"D:\WidU_multimodal_fusion"))
CKPT_PATH = ARTIFACTS / "fusion_synth_v1.pt"

MODEL_FACTORY = {
    "concat":     lambda: ConcatMLP(hidden_dims=(512, 256, 128), dropout_p=0.3),
    "gated":      lambda: GatedFusionModel(fusion_hidden=(256, 128), dropout=0.3,
                                           aux_loss_weight=0.3, reliability_mode="feature"),
    "cross_attn": lambda: CrossModalAttentionFusion(d_model=128, n_heads=4, n_layers=2,
                                                    dropout=0.3, aux_loss_weight=0.3,
                                                    emb_bottleneck=16),
}


def main():
    if not CKPT_PATH.exists():
        print(f"체크포인트 없음: {CKPT_PATH}"); return

    print(f"체크포인트 로드: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    version = ckpt.get("version", "vf")
    seed    = ckpt.get("args", {}).get("seed", 42)
    val_f1s = ckpt.get("val_macro_f1", {})
    print(f"version={version}  seed={seed}  val_macro_f1={val_f1s}")

    # 데이터 (test_loader 만 필요 — 결측강건성 평가용)
    data_ext = DATA_ROOT / "synthetic"
    if (data_ext / f"p2_synth_{version}_test.npz").exists():
        data_dir = data_ext
    else:
        data_dir = ARTIFACTS / "synthetic"
    print(f"데이터: {data_dir}")
    _, _, test_loader = make_loaders(data_dir, batch_size=256,
                                     modality_dropout_p=0.0, version=version)

    # confounder (motion 포함)
    print("\nconfounder 구성 (motion 포함):")
    conf_sets = build_all(seed=seed, include_motion=True)
    for cs in conf_sets:
        print(f"  - {cs.name}: n={cs.n} | {cs.realness}")

    # 모델 복원
    trained = {}
    model_states = ckpt.get("model_states", {})
    for name, factory in MODEL_FACTORY.items():
        if name not in model_states:
            print(f"  [skip] {name} — 체크포인트에 없음"); continue
        m = factory().to(DEVICE)
        m.load_state_dict(model_states[name])
        m.eval()
        trained[name] = m
        print(f"  복원: {name}")

    # 2D 평가
    print("\n2D 평가 중...")
    results = {}
    for name, model in trained.items():
        results[name] = evaluate_2d(model, test_loader, conf_sets, DEVICE)

    table = format_2d_table(results)
    print("\n" + table)

    # XAI
    if "cross_attn" in trained:
        print("\n[attention XAI — cross_attn, confounder 케이스]")
        xai_block = xai.confounder_attention_report(trained["cross_attn"], conf_sets, DEVICE)
        print(xai_block)

    # 리포트 업데이트
    report_path = ARTIFACTS / "dryrun_report.txt"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n\n=== motion confounder 추가 평가 (기존 체크포인트 재사용) ===\n")
        f.write(f"nstdb_motion.npz 생성 후 재실행. 재훈련 없음(val_macro_f1={val_f1s}).\n\n")
        f.write("confounder 앵커(비순환):\n")
        for cs in conf_sets:
            f.write(f"  - {cs.name}: n={cs.n} | {cs.realness}\n")
        f.write("\n" + table + "\n")
        if "cross_attn" in trained:
            f.write("\n[attention XAI — cross_attn]\n" + xai_block + "\n")
    print(f"\n리포트 업데이트: {report_path}")


if __name__ == "__main__":
    main()
