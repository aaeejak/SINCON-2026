"""최종 A (submission_fixcancel_geombawi, Public 0.144388) 재현 스크립트.

사용 (패키지 루트에서):
    python code/run_all.py              # 저장된 개별 모델 예측에서 시작 (약 2분)
    python code/run_all.py --retrain    # v7·역별 LightGBM까지 다시 학습해 저장본과 비교 (약 10분 추가)

준비: 대회 원본 train.csv, test.csv, metaData.csv, sample_submission.csv 를 data/raw/ 에 둔다.
작업은 work/ 폴더에서 하고, 단계마다 저장본(artifacts/, predictions/)과 log1p 차이를 출력한다.
"""
import argparse, hashlib, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work"
RAW_SHA = {  # 대회 원본 SHA-256
    "train.csv": "1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc",
    "test.csv": "d257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa",
    "metaData.csv": "9870dc6c9cbe7e6fc810bbe589dd9e06f309bb173a268a1bd63478f430785380",
    "sample_submission.csv": "8c5e10d4201460ddff2d78bf10a229ec8b604e86ae2f18bcd96969093981c93d",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(*args):
    print("$ python", " ".join(map(str, args)), flush=True)
    t = time.time()
    subprocess.run([sys.executable, *map(str, args)], cwd=W, check=True)
    print(f"  ({time.time() - t:.0f}초)", flush=True)


def setup():
    if W.exists():
        shutil.rmtree(W)
    (W / "data" / "raw").mkdir(parents=True)
    (W / "data" / "derived").mkdir(parents=True)
    (W / "data" / "external").mkdir(parents=True)
    (W / "send").mkdir()
    for f, h in RAW_SHA.items():
        src = ROOT / "data" / "raw" / f
        assert src.exists(), f"data/raw/{f} 가 필요합니다"
        assert sha(src) == h, f"{f} SHA-256 불일치"
        shutil.copy(src, W / "data" / f)          # 파이프라인용 data/
        shutil.copy(src, W / "data" / "raw" / f)  # 검바위 코드용 data/raw/
    shutil.copytree(ROOT / "data" / "external", W / "external")
    shutil.copy(ROOT / "data" / "holidays_kr_2025_2026.csv", W)
    shutil.copy(ROOT / "data" / "external_geombawi" / "calendar_features.csv", W / "data" / "derived")
    shutil.copy(ROOT / "data" / "external_geombawi" / "events_master.csv", W / "data" / "external")
    for f in ["sincon_pipeline_v7.py", "sincon_pipeline_v10.py"]:
        shutil.copy(ROOT / "code" / f, W)
    shutil.copytree(ROOT / "code" / "scripts", W / "scripts")
    shutil.copytree(ROOT / "code" / "geombawi_nayun", W / "code")
    for f in ["submission_OPT4_final_geo.csv", "submission_OPT4_final_geo_okf.csv"]:  # cancel_adjust 입력
        shutil.copy(ROOT / "artifacts" / "blend_chain" / f, W)
    print("원본 SHA-256 확인, 작업 폴더 준비 완료", flush=True)


TE = None


def L(path):
    return np.log1p(TE.merge(pd.read_csv(path), on="example_id").boardings.values)


def compare(new, ref, tol=1e-9):
    d = float(np.abs(L(new) - L(ref)).max())
    ok = d < tol
    print(f"  비교 {Path(new).name} vs {Path(ref).relative_to(ROOT)}: 최대 log 차이 {d:.2e} {'OK' if ok else '다름'}", flush=True)
    return ok


def check_format(path):
    s = pd.read_csv(path)
    te = pd.read_csv(W / "data" / "test.csv")  # 제공 sample_submission은 옛 117,576행 판이라 test.csv의 ID로 확인
    assert list(s.columns) == ["example_id", "boardings"] and len(s) == 157194 == len(te)
    assert s.example_id.is_unique and set(s.example_id) == set(te.example_id)
    assert s.boardings.notna().all() and np.isfinite(s.boardings).all() and (s.boardings >= 0).all()
    print(f"  형식 OK: {Path(path).name} (157,194행, ID 일치, 결측·음수·무한대 없음)", flush=True)


def main():
    global TE
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__, flush=True)
    setup()
    TE = pd.read_csv(W / "data" / "test.csv")[["example_id"]]
    A, B = ROOT / "artifacts" / "models", ROOT / "artifacts" / "blend_chain"
    ok = True

    if a.retrain:
        print("\n[1] 개별 모델 다시 학습 (kdbwls 모델)", flush=True)
        run("sincon_pipeline_v7.py", "--mode", "submit", "--out", "v7.csv")
        ok &= compare(W / "v7.csv", A / "v7.csv", tol=1e-6)
        run("scripts/run_per_station_final.py", "ps15")
        ok &= compare(W / "send" / "model_ps15.csv", A / "ps15.csv", tol=1e-6)
        # 수준 맞춤: OPT4 원본 섞기와 log 평균을 같게 (Public 미사용)
        lp, lo = L(W / "send" / "model_ps15.csv"), L(B / "OPT4_raw.csv")
        s = TE.copy(); s["boardings"] = np.expm1(lp + (lo.mean() - lp.mean())); s.to_csv(W / "ps15_lvl.csv", index=False)
        ok &= compare(W / "ps15_lvl.csv", A / "ps15_lvl.csv", tol=1e-6)

    print("\n[2] 섞기 원본 OPT5_raw: 저장본 사용 (Public 점수 C 행렬로 계산한 log 가중 평균, README 5절)", flush=True)
    shutil.copy(B / "OPT5_raw.csv", W / "OPT5_raw.csv")

    print("\n[3] 후처리 full_pp (축제 → 명절 0.5 → 공연장·인접역+OKF → 검단 0.5 → SSG 종료)", flush=True)
    run("scripts/full_pp.py", "OPT5_raw.csv", "OPT5_pp.csv")
    ok &= compare(W / "OPT5_pp.csv", B / "OPT5_pp.csv")

    print("\n[4] 팀 섞기 0.5/0.5 (log1p 공간)", flush=True)
    s = TE.copy(); s["boardings"] = np.expm1(0.5 * L(W / "OPT5_pp.csv") + 0.5 * L(B / "team_blend_opt4_75_v6pp_25.csv"))
    s.to_csv(W / "OPT5pp_team50.csv", index=False)
    ok &= compare(W / "OPT5pp_team50.csv", B / "OPT5pp_team50.csv")

    print("\n[5] SSG·OKF 보정 세기 복원: 저장본 사용 (README 5절)", flush=True)
    shutil.copy(B / "OPT5pp_team50_fix.csv", W / "OPT5pp_team50_fix.csv")

    print("\n[6] 취소 행사 반영 (OKF 9/26~27, GHOSTIVAL 10/24~25)", flush=True)
    run("scripts/cancel_adjust.py", "OPT5pp_team50_fix.csv", "OPT5pp_team50_fix_cancel.csv", "1.0")
    ok &= compare(W / "OPT5pp_team50_fix_cancel.csv", B / "OPT5pp_team50_fix_cancel.csv")

    print("\n[7] 검바위(3208) 시험 기간 보정 (nayun 코드)", flush=True)
    run("code/exam_adjust.py", "OPT5pp_team50_fix_cancel.csv", "step_exam.csv", "0.6")
    run("code/geombawi_adjust.py", "step_exam.csv", "final_A.csv", "hours", "h1516")
    ok &= compare(W / "final_A.csv", ROOT / "predictions" / "final_A_fixcancel_geombawi.csv", tol=1e-12)
    check_format(W / "final_A.csv")

    print("\n재현 성공" if ok else "\n차이 있음 — 위 비교 확인", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
