"""ptt_ppg 정상 3분류(sit/walk/run) 피험자단위 분할 정본 생성·공유.

학습 융합(P2-B)과 IMU 임계룰 baseline이 동일 분할을 써야 apples-to-apples. 윈도우 섞기
금지(누수·낙관편향) → 피험자단위 22명 train/test 분리. 결정적(seed 고정).
출력: interim/ptt_subject_split.json (양 트랙 공통 참조).
"""
import json
import os
from pathlib import Path

import numpy as np

OUT  = Path(os.environ.get("P2_DATA_DIR", "data")) / "interim" / "ptt_subject_split.json"
SEED = 42
N_TRAIN = 15   # 15 train / 7 test (≈68/32, 피험자단위)

subjects = np.array([f"s{i}" for i in range(1, 23)])
rng = np.random.default_rng(SEED)
perm = rng.permutation(subjects)
train = sorted(perm[:N_TRAIN].tolist(), key=lambda s: int(s[1:]))
test  = sorted(perm[N_TRAIN:].tolist(), key=lambda s: int(s[1:]))

split = {
    "seed": SEED,
    "task": "ptt_ppg sit/walk/run 3-class (0=sit,1=walk,2=run)",
    "n_subjects": 22, "n_train": len(train), "n_test": len(test),
    "train": train, "test": test,
    "note": "피험자단위 분할(윈도우 섞기 금지). P2-B 학습융합 + IMU 임계룰 baseline 공통 사용.",
}
OUT.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"저장: {OUT}")
print(f"train({len(train)}): {train}")
print(f"test ({len(test)}): {test}")
