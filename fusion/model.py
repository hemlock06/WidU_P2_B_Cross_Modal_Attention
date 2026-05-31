"""융합 모델 3 변형 — 동일 batch dict 인터페이스로 2D 비교.

  ConcatMLP                 : 단순 concat → MLP 베이스라인 (late fusion concat 대응)
  GatedFusionModel          : 모달리티별 expert + confidence-aware 게이팅 네트워크
  CrossModalAttentionFusion : Transformer 교차모달 융합 (attention weights → XAI)

세 모델 공통 입력:
  batch = {ecg_emb[B,768], ecg_aux[B,10], imu[B,12], spo2[B,8], mask[B,3], label[B]}
  ecg_aux = [cardiac_probs×5, emergency_score, reliability, gate_tier, hr_bpm, rhythm_regularity]
출력: ConcatMLP은 logits[B,5]. Gated/Cross는 dict(logits + gate/attention 등 분석용 부가출력).

cross-modal attention의 동기 (conf-routed 게이트 한계):
  단일 모달이 강하게 오답을 가리켜도 이기는 confounder 오경보를 joint modeling으로 완화 —
    운동 중 잠깐 낙상 → IMU 단독 낙상 / 만성 비정상 ECG → ECG 단독 심혈관 /
    수면무호흡 → SpO2 단독 저산소. 다른 모달에 attend → 거부권(veto)·협의 + attention_weights XAI.
주의: 조건부 독립 proxy에선 시간 대응이 부재해 attention이 게이팅으로 퇴화할 수 있다 —
      우열은 선험 가정이 아니라 confounder-FP × 결측강건성 2D 측정으로 판정한다.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fusion.schema import EMB_DIM, IMU_DIM, SPO2_DIM, NUM_CLASSES

# ecg_aux 구성 (schema.flat_ecg_aux 순서)
#   [cardiac_probs×5, emergency_score, reliability, gate_tier, hr_bpm, rhythm_regularity]
ECG_AUX_DIM = 10


def _mlp(in_dim: int, hidden_dims: tuple, out_dim: int,
         dropout: float = 0.2, norm: bool = True) -> nn.Sequential:
    """Linear→(LayerNorm)→GELU→Dropout 스택 + 최종 Linear."""
    layers: list = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        if norm:
            layers.append(nn.LayerNorm(h))
        layers += [nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ═════════════════════════════════════════════════════════════════════════════
# 1) ConcatMLP — 모든 모달리티를 단순 연결한 MLP 베이스라인
# ═════════════════════════════════════════════════════════════════════════════
#   ecg_emb 768 + ecg_aux 10 + imu 12 + spo2 8 = 798
#   모달리티 드롭아웃: mask(0/1)를 해당 피처 벡터에 곱해 결측 시뮬레이션.
INPUT_DIM = EMB_DIM + ECG_AUX_DIM + IMU_DIM + SPO2_DIM  # 798


class ConcatMLP(nn.Module):
    """단순 concat → LayerNorm → MLP → 5분류 (게이트·attention 없음).

    emb_bottleneck>0: ECG 768 임베딩을 bottleneck차원으로 투영(+Dropout) 후 concat —
      gated/cross_attn의 ECG 용량 정규화와 정합시켜 '용량 vs 아키텍처'를 분리하는 통제용.
    """

    def __init__(self, hidden_dims=(512, 256, 128), dropout_p: float = 0.3,
                 num_classes: int = NUM_CLASSES, emb_bottleneck: int = 0,
                 emb_dropout: float = 0.5):
        super().__init__()
        self.emb_bottleneck = emb_bottleneck
        if emb_bottleneck > 0:
            self.ecg_bn = nn.Sequential(nn.Linear(EMB_DIM, emb_bottleneck),
                                        nn.Dropout(emb_dropout))
            in_dim = emb_bottleneck + ECG_AUX_DIM + IMU_DIM + SPO2_DIM
        else:
            self.ecg_bn = None
            in_dim = INPUT_DIM
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = _mlp(in_dim, hidden_dims, num_classes, dropout=dropout_p)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        ecg_emb = batch["ecg_emb"]                            # [B,768]
        if self.ecg_bn is not None:
            ecg_emb = self.ecg_bn(ecg_emb)                    # [B,bn]
        # 마스크 적용 (결측 모달리티 → 0벡터; 병목은 마스크 전에 통과해야 0이 0으로 유지)
        ecg_emb = ecg_emb        * batch["mask"][:, 0:1]
        ecg_aux = batch["ecg_aux"] * batch["mask"][:, 0:1]    # [B,10]
        imu     = batch["imu"]     * batch["mask"][:, 1:2]    # [B,12]
        spo2    = batch["spo2"]    * batch["mask"][:, 2:3]    # [B,8]

        x = torch.cat([ecg_emb, ecg_aux, imu, spo2], dim=-1)
        x = self.input_norm(x)
        return self.mlp(x)                                     # [B,5]


# ═════════════════════════════════════════════════════════════════════════════
# 2) GatedFusionModel — 모달리티별 expert + confidence-aware 게이팅
# ═════════════════════════════════════════════════════════════════════════════
#   conf_m = max(softmax(unimodal_logits_m)).detach()
#     결측 expert → 0-feat → 균등 softmax → conf 낮음 → 게이트 자동 down-weight (+ -inf 마스킹 이중보호)
#   보조손실: 각 unimodal head CrossEntropy (α). XAI: gate_weights·unimodal_logits·conf.
EXPERT_DIM  = 128                      # 모달리티별 공통 expert 출력 차원
GATE_IN_DIM = ECG_AUX_DIM + 3 + 3      # ecg_aux + modality_mask + conf(3)
AUX_RELIABILITY = 6                    # ecg_aux 내 reliability 인덱스

# reliability 사용 모드
#   "none"      : 미사용 (게이트 입력 0 처리, ECG 하드곱 없음)
#   "feature"   : 게이트넷 입력으로만 사용 (하드곱 없음) ← 권장 가설
#   "hard_mult" : ECG 피처에 (1-reliability) 직접 곱 + 게이트넷 입력
RELIABILITY_MODES = ("none", "feature", "hard_mult")


class GatedFusionModel(nn.Module):
    """게이팅 Late Fusion 5분류 모델."""

    def __init__(
        self,
        fusion_hidden: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        aux_loss_weight: float = 0.3,
        num_classes: int = NUM_CLASSES,
        reliability_mode: str = "feature",
        gate_input_norm: bool = True,
        fusion_level: str = "feature",
        gate_mode: str = "learned",
        temperature: float = 0.15,
        emb_bottleneck: int = 0,
    ):
        """
        gate_mode:
          "learned"     : 학습 gate_net (ecg_aux+mask+conf → softmax)
          "conf_routed" : conf/τ → softmax (학습 파라미터 없음, 붕괴 불가)
        temperature: conf_routed 모드 softmax 온도 (낮을수록 winner-take-all)
        """
        super().__init__()
        if reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"reliability_mode must be one of {RELIABILITY_MODES}")
        if fusion_level not in ("feature", "logit"):
            raise ValueError("fusion_level must be 'feature' or 'logit'")
        if gate_mode not in ("learned", "conf_routed"):
            raise ValueError("gate_mode must be 'learned' or 'conf_routed'")
        self.aux_loss_weight = aux_loss_weight
        self.num_classes = num_classes
        self.reliability_mode = reliability_mode
        self.gate_input_norm = gate_input_norm
        self.fusion_level = fusion_level
        self.gate_mode = gate_mode
        self.temperature = temperature
        self.emb_bottleneck = emb_bottleneck

        # ── ECG expert ──
        # emb_bottleneck>0: 768→bottleneck→128 (병목으로 외울 용량 제한) / ==0: 768→256→128
        if emb_bottleneck > 0:
            self.ecg_proj = nn.Sequential(
                nn.Linear(EMB_DIM, emb_bottleneck),
                nn.Dropout(0.5),
                nn.Linear(emb_bottleneck, EXPERT_DIM),
                nn.LayerNorm(EXPERT_DIM),
            )
        else:
            self.ecg_proj = nn.Sequential(
                nn.Linear(EMB_DIM, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, EXPERT_DIM),
                nn.LayerNorm(EXPERT_DIM),
            )

        # ── IMU / SpO2 expert ──
        self.imu_expert  = _mlp(IMU_DIM,  (64,), EXPERT_DIM, dropout)
        self.spo2_expert = _mlp(SPO2_DIM, (32,), EXPERT_DIM, dropout)

        # ── 게이팅 네트워크 (learned 모드 전용) ──
        if gate_mode == "learned":
            self.gate_in_norm = nn.LayerNorm(GATE_IN_DIM) if gate_input_norm else nn.Identity()
            self.gate_net = nn.Sequential(
                nn.Linear(GATE_IN_DIM, 32),
                nn.GELU(),
                nn.Linear(32, 3),
            )
        else:
            self.gate_in_norm = None
            self.gate_net = None

        # ── Fusion MLP (feature 모드 전용; logit 모드는 MoE라 불필요) ──
        self.fusion_mlp = _mlp(EXPERT_DIM, fusion_hidden, num_classes, dropout) \
                          if fusion_level == "feature" else None

        # ── Unimodal 보조 헤드 (각 모달리티 단독 예측력) ──
        self.ecg_head  = nn.Linear(EXPERT_DIM, num_classes)
        self.imu_head  = nn.Linear(EXPERT_DIM, num_classes)
        self.spo2_head = nn.Linear(EXPERT_DIM, num_classes)

    def forward(self, batch: Dict[str, Tensor],
                return_aux: bool = False) -> Dict[str, Tensor]:
        """Returns dict: logits[B,5], gate_weights[B,3], unimodal_logits[B,3,5], conf_per_modality[B,3]."""
        ecg_emb = batch["ecg_emb"]   # [B,768]
        ecg_aux = batch["ecg_aux"]   # [B,10]
        imu     = batch["imu"]       # [B,12]
        spo2    = batch["spo2"]      # [B,8]
        mask    = batch["mask"]      # [B,3]

        reliability = ecg_aux[:, AUX_RELIABILITY:AUX_RELIABILITY + 1]  # [B,1]

        # ── Step 1: Expert 표현 ──
        if self.reliability_mode == "hard_mult":
            ecg_soft_w = (1.0 - reliability) * mask[:, 0:1]
        else:
            ecg_soft_w = mask[:, 0:1]
        ecg_feat  = self.ecg_proj(ecg_emb) * ecg_soft_w          # [B,128]
        imu_feat  = self.imu_expert(imu  * mask[:, 1:2])         # [B,128]
        spo2_feat = self.spo2_expert(spo2 * mask[:, 2:3])        # [B,128]

        # ── Step 2: Unimodal 헤드 → 확신도 ──
        ecg_uni  = self.ecg_head(ecg_feat)                        # [B,5]
        imu_uni  = self.imu_head(imu_feat)
        spo2_uni = self.spo2_head(spo2_feat)

        # conf_m = max(softmax(uni_m)) — detach: expert로 gradient 안 흘림
        conf = torch.stack([
            F.softmax(ecg_uni,  dim=-1).max(dim=-1).values,
            F.softmax(imu_uni,  dim=-1).max(dim=-1).values,
            F.softmax(spo2_uni, dim=-1).max(dim=-1).values,
        ], dim=1).detach()                                        # [B,3]

        # ── Step 3: 게이팅 가중치 ──
        if self.gate_mode == "conf_routed":
            gate_raw = conf / self.temperature                    # [B,3]
        else:
            aux_for_gate = ecg_aux
            if self.reliability_mode == "none":
                aux_for_gate = ecg_aux.clone()
                aux_for_gate[:, AUX_RELIABILITY] = 0.0
            gate_in  = torch.cat([aux_for_gate, mask, conf], dim=-1)  # [B,16]
            gate_in  = self.gate_in_norm(gate_in)
            gate_raw = self.gate_net(gate_in)                        # [B,3]

        # 결측 모달리티 hard masking
        neg_inf = torch.full_like(gate_raw, float("-inf"))
        gate_masked = torch.where(mask > 0.5, gate_raw, neg_inf)
        all_masked = (mask.sum(dim=-1, keepdim=True) == 0)
        gate_masked = torch.where(all_masked.expand_as(gate_masked),
                                  gate_raw, gate_masked)
        gate_w = F.softmax(gate_masked, dim=-1)                    # [B,3]

        # ── Step 4: Fusion ──
        uni_stack = torch.stack([ecg_uni, imu_uni, spo2_uni], dim=1)  # [B,3,5]

        if self.fusion_level == "logit":
            probs = F.softmax(uni_stack, dim=-1)                   # [B,3,5]
            p_mix = (gate_w.unsqueeze(-1) * probs).sum(dim=1)      # [B,5]
            logits = torch.log(p_mix.clamp(min=1e-8))             # log-prob
        else:
            fused = (gate_w[:, 0:1] * ecg_feat
                     + gate_w[:, 1:2] * imu_feat
                     + gate_w[:, 2:3] * spo2_feat)                 # [B,128]
            logits = self.fusion_mlp(fused)                        # [B,5]

        return {
            "logits":            logits,
            "gate_weights":      gate_w,
            "unimodal_logits":   uni_stack,
            "conf_per_modality": conf,
        }

    def loss(self, batch: Dict[str, Tensor],
             out: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """메인 CE + aux_loss_weight × 평균 unimodal CE."""
        if out is None:
            out = self.forward(batch)

        labels = batch["label"]
        if self.fusion_level == "logit":
            main_loss = F.nll_loss(out["logits"], labels)         # logits=log-prob
        else:
            main_loss = F.cross_entropy(out["logits"], labels)

        if self.aux_loss_weight > 0 and "unimodal_logits" in out:
            uni = out["unimodal_logits"]                          # [B,3,5]
            mask = batch["mask"]
            aux, n_valid = 0.0, 0
            for m_idx in range(3):
                m_mask = mask[:, m_idx]
                if m_mask.sum() == 0:
                    continue
                aux = aux + F.cross_entropy(uni[m_mask > 0.5, m_idx, :],
                                            labels[m_mask > 0.5])
                n_valid += 1
            if n_valid > 0:
                main_loss = main_loss + self.aux_loss_weight * (aux / n_valid)

        return main_loss


# ═════════════════════════════════════════════════════════════════════════════
# 3) CrossModalAttentionFusion — Transformer 기반 교차모달 융합
# ═════════════════════════════════════════════════════════════════════════════
#   3개 토큰 [ECG, IMU, SpO2] → TransformerEncoder(norm_first) → mean-pool → 5-class
#   ECG 토큰 = [emb_bn(16) + ecg_aux(10)] = 26 → d_model (P1 점수 전부 토큰에 포함)
#   attention_weights → XAI: "이 판정에서 어떤 모달리티가 어디 주목했나"
ECG_BN_DIM  = 16                       # 임베딩 병목 차원 (과적합 방지)
ECG_TOK_DIM = ECG_BN_DIM + ECG_AUX_DIM  # ECG 토큰 입력 = 26
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 2


class CrossModalAttentionFusion(nn.Module):
    """Transformer 기반 교차모달 융합 — GatedFusionModel과 동일 batch 인터페이스."""

    def __init__(
        self,
        d_model:         int   = D_MODEL,
        n_heads:         int   = N_HEADS,
        n_layers:        int   = N_LAYERS,
        dropout:         float = 0.3,
        aux_loss_weight: float = 0.3,
        emb_bottleneck:  int   = ECG_BN_DIM,
        num_classes:     int   = NUM_CLASSES,
    ):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        self.d_model = d_model
        self.num_classes = num_classes

        # ── ECG 임베딩 병목 (768 → emb_bottleneck) ──
        self.ecg_bn = nn.Sequential(
            nn.Linear(EMB_DIM, emb_bottleneck),
            nn.Dropout(0.5),   # 강한 dropout으로 과적합 방지
        )

        # ── 모달리티별 토큰 투영 → d_model ──
        ecg_tok_in = emb_bottleneck + ECG_AUX_DIM
        self.ecg_proj  = nn.Sequential(nn.Linear(ecg_tok_in, d_model), nn.LayerNorm(d_model))
        self.imu_proj  = nn.Sequential(nn.Linear(IMU_DIM,    d_model), nn.LayerNorm(d_model))
        self.spo2_proj = nn.Sequential(nn.Linear(SPO2_DIM,   d_model), nn.LayerNorm(d_model))

        # ── Cross-Modal Transformer Encoder ──
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,   # 토큰 3개라 작게
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,               # Pre-LN: 학습 안정성
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── 분류 헤드 ──
        self.cls_head = _mlp(d_model, (d_model // 2,), num_classes, dropout=dropout)
        # 보조(unimodal): 어텐션 후 각 토큰 → 5-class (XAI + 학습 정규화)
        self.ecg_uni_head  = nn.Linear(d_model, num_classes)
        self.imu_uni_head  = nn.Linear(d_model, num_classes)
        self.spo2_uni_head = nn.Linear(d_model, num_classes)

        self._last_attn: Optional[Tensor] = None  # 마지막 레이어 어텐션 캡처(XAI)

    def forward(self, batch: Dict[str, Tensor],
                return_aux: bool = False) -> Dict[str, Tensor]:
        """Returns dict: logits[B,5], unimodal_logits[B,3,5], attention_weights[B,3,3],
        gate_weights[B,3], conf_per_modality[B,3]."""
        ecg_emb = batch["ecg_emb"]   # [B,768]
        ecg_aux = batch["ecg_aux"]   # [B,10]
        imu     = batch["imu"]       # [B,12]
        spo2    = batch["spo2"]      # [B,8]
        mask    = batch["mask"]      # [B,3]

        B = ecg_emb.size(0)

        # ── Step 1: 토큰 구성 ──
        ecg_bn  = self.ecg_bn(ecg_emb)                       # [B,16]
        ecg_tok = torch.cat([ecg_bn, ecg_aux], dim=-1)        # [B,26]
        ecg_tok = self.ecg_proj(ecg_tok)                      # [B,d]
        imu_tok  = self.imu_proj(imu)                         # [B,d]
        spo2_tok = self.spo2_proj(spo2)                       # [B,d]

        # 결측 모달리티 → 토큰 zero-out
        ecg_tok  = ecg_tok  * mask[:, 0:1]
        imu_tok  = imu_tok  * mask[:, 1:2]
        spo2_tok = spo2_tok * mask[:, 2:3]

        tokens = torch.stack([ecg_tok, imu_tok, spo2_tok], dim=1)  # [B,3,d]

        # ── Step 2: Cross-Modal Attention ──
        # 결측 토큰은 key/value에서 무시. mask:1=있음→pad_mask:True=무시
        pad_mask = (mask < 0.5)                               # [B,3]
        all_masked = pad_mask.all(dim=-1, keepdim=True)       # 전결측 방지
        if all_masked.any():
            pad_mask = pad_mask & ~all_masked.expand_as(pad_mask)

        ctx = self.transformer(tokens, src_key_padding_mask=pad_mask)  # [B,3,d]

        # 마지막 레이어 어텐션 가중치 수동 계산 (Pre-LN 반영)
        with torch.no_grad():
            last_layer = self.transformer.layers[-1]
            normed = last_layer.norm1(tokens)
            _, attn_w = last_layer.self_attn(
                normed, normed, normed,
                key_padding_mask=pad_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            self._last_attn = attn_w.detach()                 # [B,3,3]

        # ── Step 3: 분류 (mean-pool, 결측 토큰 제외) ──
        valid_mask = (~pad_mask).float().unsqueeze(-1)        # [B,3,1]
        pooled = (ctx * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        logits = self.cls_head(pooled)                        # [B,5]

        ecg_uni  = self.ecg_uni_head(ctx[:, 0, :])           # [B,5]
        imu_uni  = self.imu_uni_head(ctx[:, 1, :])
        spo2_uni = self.spo2_uni_head(ctx[:, 2, :])
        unimodal_logits = torch.stack([ecg_uni, imu_uni, spo2_uni], dim=1)  # [B,3,5]

        # ── Step 4: 분석용 부가 출력 ──
        conf = torch.stack([
            F.softmax(ecg_uni,  dim=-1).max(dim=-1).values,
            F.softmax(imu_uni,  dim=-1).max(dim=-1).values,
            F.softmax(spo2_uni, dim=-1).max(dim=-1).values,
        ], dim=1).detach()                                    # [B,3]

        attn = self._last_attn if self._last_attn is not None \
               else torch.ones(B, 3, 3, device=logits.device) / 3
        gate_w = attn.sum(dim=1)                              # 각 토큰이 받은 어텐션 합
        gate_w = gate_w / gate_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return {
            "logits":            logits,
            "unimodal_logits":   unimodal_logits,
            "attention_weights": attn,        # [B,3,3]  (XAI)
            "gate_weights":      gate_w,       # [B,3]    (분석·시각화)
            "conf_per_modality": conf,         # [B,3]
        }

    def loss(self, batch: Dict[str, Tensor], out: Dict[str, Tensor]) -> Tensor:
        """메인 CE + 보조 unimodal CE."""
        label = batch["label"]
        main_loss = F.cross_entropy(out["logits"], label)
        if self.aux_loss_weight > 0:
            uni = out["unimodal_logits"]   # [B,3,5]
            aux = sum(F.cross_entropy(uni[:, m, :], label) for m in range(uni.size(1))) / uni.size(1)
            return main_loss + self.aux_loss_weight * aux
        return main_loss
