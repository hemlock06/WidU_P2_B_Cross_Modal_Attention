"""실 페어드 평가 — ptt_ppg 운동(ECG+IMU) 응급 오발화율(suppression).

기존 fusion_synth_v1.pt 재사용(재훈련 없음). 질문: 합성 앵커서 본 융합 억제가 실
시간정렬 페어에도 일반화되나? 운동은 ECG es↑·reliability↑·HR↑ → 응급(2/3/4) 오발화
유혹. 진실=normal-active. 대조: reliability 헤드는 rel̄~0.67(>T_ALERT 0.4753)=alert라 단독으로도
응급 보류 → 융합 FP가 0보다 크면 '융합<reliability헤드'(정보가치 있는 실측).
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

from fusion.model import ConcatMLP, GatedFusionModel, CrossModalAttentionFusion
from fusion.metrics import confounder_fp
from fusion.confounders import build_exercise_confounder

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS = ROOT / "fusion" / "artifacts"
CKPT      = ARTIFACTS / "fusion_synth_v1.pt"

FACTORY = {
    "concat":     lambda: ConcatMLP(hidden_dims=(512, 256, 128), dropout_p=0.3),
    "gated":      lambda: GatedFusionModel(fusion_hidden=(256, 128), dropout=0.3,
                                           aux_loss_weight=0.3, reliability_mode="feature"),
    "cross_attn": lambda: CrossModalAttentionFusion(d_model=128, n_heads=4, n_layers=2,
                                                    dropout=0.3, aux_loss_weight=0.3, emb_bottleneck=16),
}


def main():
    if not CKPT.exists():
        print(f"체크포인트 없음: {CKPT}"); return
    ck = torch.load(CKPT, map_location=DEVICE)
    states = ck.get("model_states", {})

    ex = build_exercise_confounder(seed=42)
    rel_mean = float(ex.arrays["ecg_aux"][:, 6].mean())   # reliability = aux idx 6
    print(f"실 페어드(운동) 평가: {ex.realness}")
    print(f"진실=normal-active(1)  |  응급 오발화 = pred ∈ {{cardiac,impact,hypoxia}}  |  n={ex.n}")
    print(f"reliabilitȳ={rel_mean:.3f} (>T_ALERT 0.4753 → reliability 헤드 단독 응급 보류)\n")

    lines = [f"{'model':<12}{'emergency-FP(↓)':>16}   pred 분포(비영)"]
    for name in ["concat", "gated", "cross_attn"]:
        if name not in states:
            continue
        m = FACTORY[name]().to(DEVICE); m.load_state_dict(states[name]); m.eval()
        r = confounder_fp(m, ex, DEVICE)
        dist = {k: v for k, v in r["pred_dist"].items() if v > 0}
        lines.append(f"{name:<12}{r['fp_rate']:>16.3f}   {dist}")
    table = "\n".join(lines)
    print(table)

    report = ARTIFACTS / "dryrun_report.txt"
    with open(report, "a", encoding="utf-8") as f:
        f.write("\n\n=== 실 페어드 평가 — ptt_ppg 운동 ECG+IMU (재훈련 없음) ===\n")
        f.write(f"{ex.realness}\n진실=normal-active, 응급 오발화=pred∈{{2,3,4}}, "
                f"reliabilitȳ={rel_mean:.3f}(alert)\n\n" + table + "\n")
        f.write("해석: 실 ECG+IMU 정렬쌍서 융합이 운동 유발 ECG 상승을 맥락(활동 IMU·정상 SpO2)으로\n"
                "억제하는지. 확인급 — reliability 헤드·규칙 결합기와 중복. FP 높으면 융합<reliability헤드.\n")
    print(f"\n리포트 업데이트: {report}")


if __name__ == "__main__":
    main()
