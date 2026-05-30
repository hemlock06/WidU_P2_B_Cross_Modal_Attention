# P2-B Cross-Modal Attention — 멀티모달 응급 융합 (Project 2-B)

## 개요

멀티모달 응급탐지의 학습형 융합 중 cross-modal attention 접근이다. P2-A(late fusion)에서 분기한
별도 트랙으로, ECG(768 임베딩)·IMU·SpO2를 토큰으로 보고 모달 간 multi-head attention으로 융합한다.
late fusion이 채널별로 독립 처리한 뒤 말단에서 결합하는 데 반해, 본 접근은 **교차모달 맥락을 학습으로
포착**한다.

다만 조건부 독립으로 조립한 proxy 데이터에서는 attention이 baseline(concat/gating)과 동급일 것으로
예상되며, 고유 우위는 **설명가능성(어텐션 가중치 해석)과 결측 강건성**, 그리고 시간 정렬된 실 데이터
에서의 정확도다. 배경은 P2-A(`WidU_P2_A_Late_Fusion`)의 설계·한계 분석을 참조.

---

## 접근 (P2-A vs P2-B)

| | P2-A | P2-B (본 레포) |
|---|---|---|
| 모델 | late fusion | cross-modal attention |
| 결합 | 채널별 독립 처리 → 말단 결합 | 모달 간 multi-head attention |
| 역할 | 베이스라인(≈ concat) | 신규 기여 |

핵심 어블레이션: **"cross-modal attention이 late fusion을 이기는가"** — 동일 데이터·평가로 측정한다.

---

## 계획 구조

```
fusion/
├── model.py          융합 구조 3종: ConcatMLP(≈A) / GatedFusion / CrossModalAttention
├── dataset.py        멀티모달 조립 (P1 임베딩·probs + IMU·SpO2 피처 + 라벨) — proxy/실데이터 공통
├── metrics.py        2D 평가: {confounder 오경보율} × {결측 강건성}
├── train.py          재사용 학습 루프
├── run_dryrun.py     proxy 데이터 → fusion_synth_v1.pt (예행, 비배포)
├── train_on_field.py 실 데이터 + fresh init → fusion_field_v1.pt
└── artifacts/        가중치·리포트 (gitignore)
```

---

## 상태

레포 셋업만 완료했고 빌드는 미착수다. 선행 조건(P1 ECG 검출기와 ECG 임베딩 768차원 노출)이 갖춰져
있어 예행(proxy) 스캐폴딩에 진입할 수 있다.

---

## 관련 레포

- **P1** (`WidU_P1_ECG-emergency-detection`): ECG-FM + LoRA 심장 검출기 — ECG 임베딩 공급.
- **P2-A** (`WidU_P2_A_Late_Fusion`): late fusion 접근과 구조적 한계 — 본 트랙의 출발점.
