"""대회 원본 4개 파일을 이 저장소의 필요한 위치에 복사하고 SHA-256을 확인한다.

대회 제공 자료는 공개 저장소에 올리지 않았다 (제출 안내: 변형하지 않은 제공 자료는 파일명과 해시로 식별).
사용법:  python setup_raw_data.py <train.csv·test.csv·metaData.csv·sample_submission.csv 가 있는 폴더>
줄바꿈이 CRLF로 바뀐 사본이면 LF로 되돌린 뒤 다시 확인한다 (내용은 같음).
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA = {
    "train.csv": "1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc",
    "test.csv": "d257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa",
    "metaData.csv": "9870dc6c9cbe7e6fc810bbe589dd9e06f309bb173a268a1bd63478f430785380",
    "sample_submission.csv": "8c5e10d4201460ddff2d78bf10a229ec8b604e86ae2f18bcd96969093981c93d",
}
DEST = {
    "train.csv": ["SINCON_2026_요모조모_kdbwls/data/raw", "SINCON_최종제출_nayun/data/raw", "SINCON_2026_요모조모_Dohyeong/code/dh_models/sincon-2026"],
    "test.csv": ["SINCON_2026_요모조모_kdbwls/data/raw", "SINCON_최종제출_nayun/data/raw", "SINCON_2026_요모조모_Dohyeong/code/dh_models/sincon-2026"],
    "metaData.csv": ["SINCON_2026_요모조모_kdbwls/data/raw", "SINCON_최종제출_nayun/data/raw", "SINCON_2026_요모조모_Dohyeong/code/dh_models/sincon-2026"],
    "sample_submission.csv": ["SINCON_2026_요모조모_kdbwls/data/raw", "SINCON_최종제출_nayun/data/raw", "SINCON_2026_요모조모_Dohyeong/code/dh_models/sincon-2026"],
}

src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if src is None or not src.is_dir():
    raise SystemExit(__doc__)
ok = True
for name, want in SHA.items():
    data = (src / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != want:
        data = data.replace(b"\r\n", b"\n")
    good = hashlib.sha256(data).hexdigest() == want
    ok &= good
    for d in DEST[name]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        (ROOT / d / name).write_bytes(data)
    print(f"{name}: SHA-256 {'일치' if good else '불일치 (원본이 다를 수 있음)'} -> {', '.join(DEST[name])}")
print("준비 완료" if ok else "확인 필요: 해시가 다른 파일이 있습니다")
