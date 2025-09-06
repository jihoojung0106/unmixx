# import glob
# import subprocess
# import os

# # 체크포인트들이 있는 디렉토리
# ckpt_dir = "Experiments/checkpoint/fin_all_rev5/20250827_01"
# # .ckpt 파일들 모두 찾기
# exclude_keys = ["last"]

# ckpts = glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt"))
# # 제외 키워드가 들어있는 파일은 걸러냄
# ckpts = [ckpt for ckpt in ckpts if not any(key in ckpt for key in exclude_keys)]
# ckpts.sort(key=lambda x: int(x.split("epoch=")[1].split(".ckpt")[0]), reverse=True)

# print(f"총 {len(ckpts)}개 checkpoint 발견")

# # 하나씩 실행
# for ckpt in ckpts:
#     print(f"Running with checkpoint: {ckpt}")
#     subprocess.run(["python", "mytest_meldey.py", "--pretrained", ckpt])
import os, glob, subprocess

# 체크포인트들이 있는 디렉토리
ckpt_dir = "Experiments/checkpoint/fin_all_rev5/20250827_01"
# 제외할 키워드
exclude_keys = ["last"]

# .ckpt 파일들 모두 찾기
ckpts = glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt"))
# 제외 키워드 필터
ckpts = [ckpt for ckpt in ckpts if not any(key in ckpt for key in exclude_keys)]

# 숫자 추출 → int 변환
def get_epoch_num(path):
    return int(path.split("epoch=")[1].split(".ckpt")[0])

# epoch < 220 조건 추가
ckpts = [ckpt for ckpt in ckpts if get_epoch_num(ckpt) < 238]

# 정렬 (큰 숫자 → 작은 숫자)
ckpts.sort(key=get_epoch_num, reverse=True)

print(f"총 {len(ckpts)}개 checkpoint 발견 (epoch < 220)")

# 하나씩 실행
for ckpt in ckpts:
    print(f"Running with checkpoint: {ckpt}")
    subprocess.run(["python", "mytest_meldey.py", "--pretrained", ckpt])
