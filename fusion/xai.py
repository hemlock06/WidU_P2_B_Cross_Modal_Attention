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

# 5분류 한국어 레이블
_CLASS_KO = ["정상(안정)", "정상(활동 중)", "심혈관 응급 의심", "낙상·충격 의심", "저산소 의심"]
# 분류별 1차 모달리티 (기대값)
_CLASS_PRIMARY_MOD = [None, "IMU", "ECG", "IMU", "SpO2"]
# ecg_aux 인덱스
_AUX_CP = slice(0, 5)   # cardiac_probs
_AUX_ES, _AUX_REL, _AUX_GT = 5, 6, 7   # emergency_score, reliability, gate_tier
_CARD_TYPE = ["NSR", "AF", "급성 허혈", "전도 장애", "이소성 박동"]


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


def generate_attention_explanation(pred_class: int,
                                   attention: np.ndarray,
                                   ecg_aux: np.ndarray) -> str:
    """P3 자연어 설명 — attention [3,3] + P1 보조 정보 → 한국어 판정 근거.

    pred_class: 0=정상(안정) 1=정상(활동) 2=심혈관 3=낙상 4=저산소
    attention:  [3,3] query→key (행=query 모달, 열=key 모달)
    ecg_aux:    [10] flat_ecg_aux 순서 (cardiac_probs×5, es, rel, gate_tier, hr, rhythm_reg)
    """
    # ── 어텐션 수신량 (열 합산 정규화 = 어느 모달이 집중받았나) ──────────────
    received = attention.sum(axis=0)
    received = received / (received.sum() + 1e-8)
    dom_idx = int(np.argmax(received))
    dom_mod = MOD[dom_idx]

    # ── ecg_aux 파싱 ─────────────────────────────────────────────────────────
    cp        = ecg_aux[_AUX_CP]                          # [5]
    rel       = float(ecg_aux[_AUX_REL])                  # 0~1 (높을수록 불량)
    gate_tier = int(round(float(ecg_aux[_AUX_GT])))       # 0/1/2
    card_type = _CARD_TYPE[int(np.argmax(cp))]
    card_conf = float(cp.max())

    # ── 판정 헤더 ─────────────────────────────────────────────────────────────
    label_ko = _CLASS_KO[pred_class]
    lines = [f"[판정] {label_ko}"]

    # ── 어텐션 근거 ───────────────────────────────────────────────────────────
    focus_str = "  ".join(f"{m} {received[i]:.0%}" for i, m in enumerate(MOD))
    lines.append(f"[모달 집중도] {focus_str}")

    # 1차 모달과 어텐션 dominant 모달 일치 여부 (균등 어텐션 = max-min < 0.10 → 전 채널)
    expected = _CLASS_PRIMARY_MOD[pred_class]
    balanced = (received.max() - received.min()) < 0.10
    if balanced:
        lines.append("  → 전 채널 균등 검토하여 판정.")
    elif expected and dom_mod == expected:
        lines.append(f"  → {dom_mod} 신호 중심으로 판정 (기대 모달 일치).")
    elif expected and dom_mod != expected:
        lines.append(f"  → {dom_mod} 신호 중심으로 판정 (기대 모달: {expected} — 교차 맥락 반영).")
    else:
        lines.append(f"  → {dom_mod} 신호 중심으로 판정.")

    # ── 분류별 세부 설명 ──────────────────────────────────────────────────────
    if pred_class == 2:   # cardiac
        ecg_q = "신호 불량(모션 등)" if gate_tier == 2 else ("신호 보통" if gate_tier == 1 else "신호 양호")
        lines.append(f"  → 심전도: {card_type} 의심(확률 {card_conf:.0%}), {ecg_q}.")
        if gate_tier == 2:
            lines.append("     ECG 신뢰도 낮음 — 재측정 또는 안정 후 확인 권고.")
        if received[0] < 0.25 and gate_tier == 2:
            lines.append("     (모델이 손상된 ECG 채널 어텐션을 낮춤 — reliability 인지 작동.)")

    elif pred_class == 3:  # impact
        ecg_q = "" if gate_tier < 2 else " ECG 신호 불량(움직임)."
        lines.append(f"  → 충격·낙상 패턴 감지.{ecg_q}")
        if received[1] < 0.35:
            lines.append("     IMU 어텐션 낮음 — 다른 채널 교차 맥락으로 판정 (불확실성 ↑).")

    elif pred_class == 4:  # hypoxia
        lines.append(f"  → 산소포화도 저하 패턴 감지.")
        if received[2] < 0.30:
            lines.append("     SpO2 어텐션 낮음 — ECG/IMU 교차 맥락으로 보완 판정.")

    elif pred_class in (0, 1):  # normal
        lines.append("  → 모든 채널 정상 범위.")

    # ── 신뢰도 주의문구 (신호불량 시) ────────────────────────────────────────
    if gate_tier == 2 and pred_class != 3:
        lines.append("[주의] 심전도 신호 불량 — 활동 중이거나 전극 접촉 문제일 수 있습니다.")

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
