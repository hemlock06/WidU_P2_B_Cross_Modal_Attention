"""실데이터 학습 — fresh init → fusion_field_v1.pt (필드 단계, 미착수).

드라이런(proxy)과의 차이:
  - 입력: 실 시간정렬 멀티모달 (ECG+IMU+SpO2 동일인·동시각). proxy의 조건부 독립 한계 없음.
  - 가중치: proxy 산출(fusion_synth_v1.pt)을 *버리고* fresh init — proxy는 배관 검증용이라
            실데이터로 재학습한다(전이 안 함). 동일 model.py·train.py 루프 재사용.
  - 평가: confounder-FP × 결측강건성 2D는 동일 metrics.py. 단 confounder가 합성 앵커가 아니라
          실 정렬쌍 자체에서 측정 → cross-modal attention의 joint 우위가 처음으로 검증 가능.

선행 조건 (필드 데이터 확보 후):
  1) 실 멀티모달 코호트 수집·동기화·윈도우잉 → fusion/dataset.py 스키마로 정규화
  2) features.py 로 IMU/SpO2 피처, P1 채널로 ECG 임베딩(768) 추출
  3) train.py 루프 재사용 (--model cross_attn), fresh init
  4) metrics.py 2D + xai.py 어텐션 해석

이 파일은 인터페이스 고정용 스텁이다. 데이터 원천 확보 전에는 실행 불가.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "train_on_field: 실 시간정렬 멀티모달 코호트 확보 후 구현. "
        "현재는 proxy 드라이런(run_dryrun.py)까지가 범위."
    )


if __name__ == "__main__":
    main()
