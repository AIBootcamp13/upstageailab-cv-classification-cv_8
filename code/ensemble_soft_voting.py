# ensemble_soft_voting.py
import os
import glob
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta, timezone

# 아웃풋 디렉토리 지정
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "output"))

# fold별 예측 결과 csv 파일들 자동 탐색 (가장 최근 것들)
fold_csvs = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*seperate_doc_v2*.csv')))
assert len(fold_csvs) > 1, "앙상블할 fold csv 파일이 2개 이상 필요합니다."
print("[INFO] 앙상블 대상 파일들:")
for f in fold_csvs:
    print(" -", os.path.basename(f))

# 모든 파일 읽어서 리스트로 저장
fold_dfs = [pd.read_csv(f) for f in fold_csvs]

# ID 동일한지 검증
for df in fold_dfs[1:]:
    assert (fold_dfs[0]['ID'] == df['ID']).all(), "ID 순서가 다릅니다."

# soft voting 앙상블
print("\n[INFO] soft voting 진행 중...")
all_preds = [df['target'].values for df in fold_dfs]
ensemble_preds = []
for i in range(len(all_preds[0])):
    votes = [preds[i] for preds in all_preds]
    voted = Counter(votes).most_common(1)[0][0]  # 최빈값
    ensemble_preds.append(voted)

# 최종 결과 저장
final_df = fold_dfs[0].copy()
final_df['target'] = ensemble_preds

# 한국 시간 기준 파일명 생성
KST = timezone(timedelta(hours=9))
timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
ensemble_filename = os.path.join(OUTPUT_DIR, f"{timestamp}_efficient_seperate_doc_v2_ensemble.csv")
final_df.to_csv(ensemble_filename, index=False)
print(f"\n✅ 앙상블 결과 저장 완료: {ensemble_filename}")
