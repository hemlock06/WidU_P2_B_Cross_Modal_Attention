"""Attention 가중치 추출·요약 — cross-modal attention의 XAI (P2-B 고유 기여).

CrossModalAttentionFusion은 forward마다 마지막 레이어 어텐션 [B,3,3] (토큰순 ECG·IMU·SpO2,
행=query 모달, 열=key 모달, head 평균)을 출력한다. 이를 모아 "어떤 판정에서 어떤 모달리티가
어디에 주목했나"를 요약한다 — confounder 케이스에서 특히, 모델이 오답 모달에 쏠렸는지 본다.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from fusion.metrics import _to_batch
from fusion.schema import CLASS_NAMES

MOD = ["ECG", "IMU", "SpO2"]


@torch.no_grad()
def collect_attention(model, arrays: Dict[str, np.ndarray], device,
                      batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """arrays → (attention[N,3,3], preds[N]). cross_attn 전용(attention_weights 출력 필요)."""
    model.eval()
    n = len(arrays["ecg_embedding"])
    A, P = [], []
    for i in range(0, n, batch_size):
        sub = {k: v[i:i + batch_size] for k, v in arrays.items()}
        out = model(_to_batch(sub, device))
        if "attention_weights" not in out:
            raise ValueError("모델이 attention_weights를 출력하지 않음 (cross_attn 전용 XAI)")
        A.append(out["attention_weights"].cpu().numpy())
        P.append(out["logits"].argmax(-1).cpu().numpy())
    return np.concatenate(A), np.concatenate(P)


def summarize(attn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """attn[N,3,3] → (평균 어텐션 행렬[3,3], 모달별 받은 어텐션[3] 정규화)."""
    mean = attn.mean(axis=0)                 # [3,3] query→key
    received = mean.sum(axis=0)              # 열 합 = 각 모달이 받은 총 어텐션
    received = received / received.sum()
    return mean, received


def format_attention(mean: np.ndarray, received: np.ndarray, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append("  attention[query→key] 평균 (행=query, 열=key):")
    lines.append("           " + "".join(f"{m:>8}" for m in MOD))
    for i, m in enumerate(MOD):
        lines.append(f"    {m:>5}  " + "".join(f"{mean[i, j]:>8.3f}" for j in range(3)))
    lines.append("  받은 어텐션(정규화): "
                 + "  ".join(f"{m}={received[j]:.3f}" for j, m in enumerate(MOD)))
    return "\n".join(lines)


def confounder_attention_report(model, confounder_sets, device) -> str:
    """각 confounder 케이스에서의 어텐션 쏠림 요약 — 오답 모달 지배 여부 진단."""
    blocks = []
    for cs in confounder_sets:
        try:
            attn, preds = collect_attention(model, cs.arrays, device)
        except ValueError as e:
            return f"(XAI 불가: {e})"
        mean, received = summarize(attn)
        fp = float((preds == cs.emergency_class).mean())
        title = f"[{cs.name}] FP({CLASS_NAMES[cs.emergency_class]})={fp:.3f}  n={cs.n}"
        blocks.append(format_attention(mean, received, title))
    return "\n\n".join(blocks)
