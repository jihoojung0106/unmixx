import json
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# === 설정 ===
LIST_PATH  = "all_filelist.txt"     # 경로 목록이 적힌 텍스트
OUT_JSON   = "all_filelist.json"    # 저장 파일명
TIME_UNIT  = "sec"                  # "sample" 또는 "sec"

# === 함수 ===
def get_length(path: Path, unit: str = "sec") -> float:
    info = sf.info(str(path))
    return info.frames if unit == "sample" else info.frames / info.samplerate

# === 메인 ===
def main():
    # (1) 파일 목록 읽기
    with open(LIST_PATH, "r") as f:
        files = [Path(line.strip()) for line in f if line.strip()]

    if not files:
        print("⚠️  all_filelist.txt 가 비어 있습니다.")
        return

    # (2) 길이 계산
    results = []
    total_len = 0.0
    for path in tqdm(files, desc="길이 측정 중"):
        try:
            length = get_length(path, TIME_UNIT)
            results.append({"path": str(path), "length_sec": length})
            total_len += length
        except Exception as e:
            tqdm.write(f"❌ {path} - {e}")  # 손상된 파일 등

    # (3) JSON 저장 (첫 번째 스크립트와 동일 형식)
    with open(OUT_JSON, "w") as jf:
        json.dump({
            "total_length_sec": total_len,
            "total_length_hour": total_len / 3600,
            "n_files": len(results),
            "files": results
        }, jf, ensure_ascii=False, indent=2)

    print(f"✅ {len(results)}개 파일의 총 길이: {total_len:.2f}초 ≈ {total_len/3600:.2f}시간")
    print(f"→ {OUT_JSON} 저장 완료")

if __name__ == "__main__":
    main()