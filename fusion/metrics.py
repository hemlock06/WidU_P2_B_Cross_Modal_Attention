"""2D 평가 — {confounder 오경보율} × {결측 강건성}.

이 두 축이 P2-B의 고유 기여다. proxy 정확도(clean macro-F1)는 세 모델이 동급일 것으로
예상되므로(조건부 독립 → attention이 게이팅으로 퇴화), 변별은 다음에서 측정한다:

  축1  confounder-FP (↓ 좋음): 실데이터 앵커 confounder(만성 ECG·모션·무호흡)에서
       '응급 오발화율'. benign이 진실인데 해당 응급 클래스로 예측 = false positive.
       (가설: cross-modal attention이 한 모달의 강한 오답을 맥락으로 거부. 이 축이 그 가설을
        시험한다 — ECG 임베딩 용량 정합 시 proxy에선 아키텍처-비특이로 측정됨: attention-free
        대조군이 동급 재현, confounder-FP는 융합 구조가 아니라 ECG 임베딩 용량의 readout.)
  축2  결측 강건성 (↑ 좋음): 모달리티를 하나씩 결측시켰을 때 clean 대비 macro-F1 유지.

선험 우열 가정 금지 — 표로 판정한다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from fusion.schema import CLASS_NAMES, NUM_CLASSES

MODALITY_NAMES = ["ecg", "imu", "spo2"]


# ─────────────────────────────────────────────────────────────────────────────
# 기본 지표
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


def per_class_recall(preds: np.ndarray, labels: np.ndarray) -> List[float]:
    rec = []
    for c in range(NUM_CLASSES):
        denom = (labels == c).sum()
        rec.append(float(((preds == c) & (labels == c)).sum() / denom) if denom > 0 else 0.0)
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# 통일 forward (concat=tensor / gated·cross=dict)
# ─────────────────────────────────────────────────────────────────────────────
def model_logits(model, batch: Dict[str, Tensor]) -> Tensor:
    out = model(batch)
    return out["logits"] if isinstance(out, dict) else out


def _to_batch(arrays: Dict[str, np.ndarray], device) -> Dict[str, Tensor]:
    """arrays(np) → batch dict(tensor). ecg_embedding/ecg_aux/imu_feat/spo2_feat/modality_mask[/label]."""
    b = {
        "ecg_emb": torch.as_tensor(arrays["ecg_embedding"], dtype=torch.float32),
        "ecg_aux": torch.as_tensor(arrays["ecg_aux"],       dtype=torch.float32),
        "imu":     torch.as_tensor(arrays["imu_feat"],      dtype=torch.float32),
        "spo2":    torch.as_tensor(arrays["spo2_feat"],     dtype=torch.float32),
        "mask":    torch.as_tensor(arrays["modality_mask"], dtype=torch.float32),
    }
    if "label" in arrays:
        b["label"] = torch.as_tensor(arrays["label"], dtype=torch.long)
    return {k: v.to(device) for k, v in b.items()}


@torch.no_grad()
def predict_arrays(model, arrays: Dict[str, np.ndarray], device,
                   drop_modality: Optional[int] = None, batch_size: int = 1024) -> np.ndarray:
    """arrays 전체에 대한 argmax 예측. drop_modality: None|0|1|2 (해당 모달 mask=0)."""
    model.eval()
    n = len(arrays["label"]) if "label" in arrays else len(arrays["ecg_embedding"])
    preds = []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        sub = {k: v[sl] for k, v in arrays.items()}
        batch = _to_batch(sub, device)
        if drop_modality is not None:
            batch["mask"] = batch["mask"].clone()
            batch["mask"][:, drop_modality] = 0.0
        preds.append(model_logits(model, batch).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def predict_loader(model, loader, device, drop_modality: Optional[int] = None):
    model.eval()
    P, L = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if drop_modality is not None:
            batch["mask"] = batch["mask"].clone()
            batch["mask"][:, drop_modality] = 0.0
        P.append(model_logits(model, batch).argmax(-1).cpu().numpy())
        L.append(batch["label"].cpu().numpy())
    return np.concatenate(P), np.concatenate(L)


# ─────────────────────────────────────────────────────────────────────────────
# 축1 — confounder 오발화율
# ─────────────────────────────────────────────────────────────────────────────
def confounder_fp(model, conf_set, device, batch_size: int = 1024) -> Dict:
    """conf_set: .name, .arrays(dict np), .emergency_class(int), [.true_label].

    반환: fp_rate(해당 응급 클래스로 예측한 비율), pred_dist(클래스별 예측 분포), n.
    """
    preds = predict_arrays(model, conf_set.arrays, device, batch_size=batch_size)
    em = conf_set.emergency_class
    em_set = [em] if isinstance(em, (int, np.integer)) else list(em)   # int 또는 집합(임의 응급)
    fp = float(np.isin(preds, em_set).mean())
    dist = {CLASS_NAMES[c]: int((preds == c).sum()) for c in range(NUM_CLASSES)}
    em_name = "+".join(CLASS_NAMES[c] for c in em_set)
    return {"name": conf_set.name, "emergency_class": em_name,
            "fp_rate": fp, "n": int(len(preds)), "pred_dist": dist}


# ─────────────────────────────────────────────────────────────────────────────
# 축2 — 결측 강건성
# ─────────────────────────────────────────────────────────────────────────────
def missing_robustness(model, test_loader, device) -> Dict:
    """clean macro-F1 + 각 모달리티 결측 시 macro-F1 + 유지율."""
    p, l = predict_loader(model, test_loader, device)
    clean = macro_f1(p, l)
    drops = {}
    for m_idx, m_name in enumerate(MODALITY_NAMES):
        pm, lm = predict_loader(model, test_loader, device, drop_modality=m_idx)
        f1 = macro_f1(pm, lm)
        drops[m_name] = {"macro_f1": f1, "retention": f1 / clean if clean > 0 else 0.0}
    worst = min(drops.values(), key=lambda d: d["macro_f1"])["macro_f1"]
    return {"clean_macro_f1": clean, "per_modality_drop": drops,
            "worst_drop_macro_f1": worst,
            "recall_clean": per_class_recall(p, l)}


# ─────────────────────────────────────────────────────────────────────────────
# 2D 종합
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_2d(model, test_loader, confounder_sets: Sequence, device) -> Dict:
    """한 모델의 2D 평가 행 (결측 강건성 + confounder-FP 전체)."""
    robust = missing_robustness(model, test_loader, device)
    conf = [confounder_fp(model, cs, device) for cs in confounder_sets]
    return {"robustness": robust, "confounders": conf,
            "mean_confounder_fp": float(np.mean([c["fp_rate"] for c in conf])) if conf else None}


def format_2d_table(results: Dict[str, Dict]) -> str:
    """results: {model_name: evaluate_2d(...)} → 텍스트 표."""
    lines = []
    # 헤더: confounder 이름 동적
    any_model = next(iter(results.values()))
    conf_names = [c["name"] for c in any_model["confounders"]]
    h = f"{'model':<12}{'clean':>8}{'-ecg':>7}{'-imu':>7}{'-spo2':>7}{'worst':>7}  | confounder-FP(↓): " \
        + "  ".join(f"{n}" for n in conf_names) + f"   meanFP"
    lines.append("=" * len(h)); lines.append(h); lines.append("-" * len(h))
    for mname, r in results.items():
        rob = r["robustness"]; d = rob["per_modality_drop"]
        row = (f"{mname:<12}{rob['clean_macro_f1']:>8.3f}"
               f"{d['ecg']['macro_f1']:>7.3f}{d['imu']['macro_f1']:>7.3f}{d['spo2']['macro_f1']:>7.3f}"
               f"{rob['worst_drop_macro_f1']:>7.3f}  | ")
        fps = {c["name"]: c["fp_rate"] for c in r["confounders"]}
        row += "  ".join(f"{fps[n]:>{max(len(n),5)}.3f}" for n in conf_names)
        row += f"   {r['mean_confounder_fp']:.3f}" if r["mean_confounder_fp"] is not None else "     -"
        lines.append(row)
    lines.append("=" * len(h))
    lines.append("축1 confounder-FP ↓ (응급 오발화율, benign 진실) / 축2 결측 macro-F1 ↑ (clean 대비 유지)")
    return "\n".join(lines)
