# P2-B Cross-Modal Attention — 멀티모달 응급 융합 (Project 2-B)

## 개요

멀티모달 응급탐지의 학습형 융합 중 cross-modal attention 접근이다. ECG(768 임베딩)·IMU·SpO2를
토큰으로 보고 모달 간 multi-head attention으로 융합한다. late fusion이 채널별로 독립 처리한 뒤
말단에서 결합하는 데 반해, 본 접근은 **교차모달 맥락을 학습으로 포착**한다.

핵심 입장은 정직한 측정이다. 조건부 독립으로 조립한 proxy 데이터에는 "동시각 ECG↔IMU↔SpO2"의
시간 대응이 부재해 cross-modal attention이 게이팅으로 퇴화할 수 있다. 따라서 attention의 우월을
선험적으로 가정하지 않고, 다음 **2D 평가**로 세 융합 구조(concat / gated / attention)를 비교한다.
정확도(clean macro-F1)는 동급으로 예상하며, 변별과 고유 기여는 2D 평가와 attention XAI에서 나온다.

---

## 핵심 기여 — 2D 평가 + attention XAI

late fusion 베이스라인이 단일 정확도로 닫히는 데 비해, 본 트랙은 두 축으로 응급 융합을 평가한다.

- **축1 — confounder 오경보율 (낮을수록 좋음).** 한 모달리티가 강하게 응급을 가리켜도 맥락은
  비응급인 사례에서의 오발화율. cross-modal attention의 존재 이유(맥락 거부권)를 직접 시험한다.
- **축2 — 결측 강건성 (높을수록 좋음).** 모달리티를 하나씩 결측시켰을 때 clean 대비 성능 유지.

### confounder는 실데이터에 앵커한다 (비순환)

합성 confounder는 "심은 규칙을 모델이 배웠나"의 동어반복이 된다. 따라서 각 confounder의 *정의
모달리티*는 실 신호로 두고, 맥락만 정직한 정상 앵커로 채운다.

| confounder | 정의 모달리티 (실) | 오발화 대상 | 비순환 근거 |
|---|---|---|---|
| 만성 비정상 ECG | CPSC Ectopic (비정상이나 비응급) | 심혈관(2) | proxy 조립에 미사용 → 모델 미관측 |
| 무호흡 | ucddb 무호흡 desaturation | 저산소(4) | 실 desat은 보상성 빈맥 거의 없음(HR 비상승) |
| 모션 아티팩트 | NSTDB 모션 노이즈 ECG | 심혈관(2) | reliability 단독으로 분리 (P1 추론 필요) |

---

## 구조

```
fusion/
├── schema.py          샘플 스키마 (768 임베딩 + 10 ecg_aux + 12 IMU + 8 SpO2, 5분류)
├── priors.py          클래스 조건부 사전분포 + 실데이터 캘리브레이션 샘플러
├── features.py        실 raw 윈도우 → IMU/SpO2 핸드크래프트 피처
├── dataset.py         조립기(실 P1 임베딩) + Dataset/loader + proxy 빌드
├── model.py           ConcatMLP / GatedFusionModel / CrossModalAttentionFusion
├── metrics.py         2D 평가: {confounder 오경보율} × {결측 강건성}
├── confounders.py     실데이터 앵커 confounder 셋 (만성 ECG·무호흡·모션) — 비순환
├── xai.py             attention 가중치 추출·요약
├── train.py           재사용 학습 루프 (concat/gated/cross_attn 공통)
├── run_dryrun.py      proxy 드라이런 → 2D 리포트 + fusion_synth_v1.pt (예행·버림)
├── train_on_field.py  실 데이터 + fresh init → fusion_field_v1.pt (미착수)
└── artifacts/         가중치·리포트 (gitignore)
```

5분류 taxonomy: 0 정상(안정) · 1 정상(운동) · 2 심혈관 · 3 충격(낙상) · 4 저산소.

---

## 실행

```bash
# proxy 드라이런: 3 모델 학습 + 2D 비교 + attention XAI → fusion/artifacts/
python -m fusion.run_dryrun --epochs 80

# 프록시를 P1 임베딩 캐시에서 자체 조립 (자체완결)
python -m fusion.run_dryrun --epochs 80 --build-data

# 단일 모델 학습
python -m fusion.train --model cross_attn --epochs 80
```

ECG 임베딩은 P1(ECG 검출기)의 mean-pool 768차원을 사용한다(재인코딩하지 않음). 대용량 외부
데이터(P1 임베딩 캐시·IMU 캘리브레이션·raw 신호)는 절대경로로 참조하며 레포에 복사하지 않는다.

---

## 상태

proxy 드라이런 파이프라인이 작동한다 — 3 융합 구조 학습, 2D 평가, attention XAI, 산출물 저장까지
끝까지 돈다. `fusion_synth_v1.pt`는 예행용·비배포(버림)이며 필드 모델이 아니다.

드라이런의 목적은 배관 검증·바닥값·구조 검증이다. 정확도는 세 구조가 동급으로 나타나는 것이
예상이며, 이는 응급 효능의 증명이 아니라 "조건부 독립 proxy에서는 attention의 joint 우위가
부재"라는 구조적 사실의 실측이다. 실제 정확도 변별과 attention의 맥락 거부권은 시간 정렬된 실
데이터(`train_on_field.py`)에서 fresh init으로 검증한다.

---

## 관련 레포

- **P1**: ECG-FM + LoRA 심장 검출기 — ECG 임베딩(768) 공급.
- **P2-A** (late fusion): concat/gated 베이스라인과 조건부 독립 합성의 구조적 한계 — 본 트랙의 출발점(아카이브).
