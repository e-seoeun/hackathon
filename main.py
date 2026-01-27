# main.py
'''
hackathon
ㄴdownloader
    process0_downloader.py
    process0_tle_downloader.py
    spacetrack_auth.txt (ID 및 Password 입력)
ㄴrepository
    ㄴdecay_data
        decay_YYYYMMDD.csv
        .
        .
        .
    ㄴtip_data
        tip_YYYYMMDD.csv
        .
        .
        .
    ㄴtle_data
        YYYYMMDD.tle (날짜별 tle데이터가 미리 다운받아 있어야 합니다.)
        .
        .
        .
main.py
process1_track.py
process2_calculate.py
process3_plot.py



* process0_downloader
spacetrack에서 원하는 기간동안의 decay_data message와 tip_data message를 다운받아 날짜별로 저장합니다.
spacetrack ID와 Password를 spacetrack_auth에 저장하는 과정이 필요합니다.

* process0_tle_downloader
spacetrack에서 현재 기준의 tle카탈로그를 저장합니다.
spacetrack ID와 Password를 spacetrack_auth에 저장하는 과정이 필요합니다.

* process1 
입력한 날짜의 TLE를 불러와 고도 2,000km이하일 경우 Epoch기준 subpoint에서의 궤적(총 10초, 1초 간격)을 생성합니다. 
기준날짜(입력한 날짜)의 전 날의 TLE도 함께 불러온 후 기준날짜와 동일한 Norad ID의 TLE epoch가 90분 이상의 차이가 날 경우 동일 시간동안 전파합니다. 
두 날짜의 TLE에서 생성한 전파 궤적과 궤도요소 정보를 YYYYMMDD_result_raw.csv파일로 중간저장합니다.

* process2 
YYYYMMDD_result_raw.csv를 불러온 후 기준날짜와 전날짜 기준 Apparent Angular Velocity, Motion Angle을 계산하고 
수직방향 및 수평방향 거리인 D1, D2를 계산하여 최종 결과파일인 YYYYMMDD_result.csv파일을 저장합니다.

* process3
실행 후 원하는 Norad ID를 input으로 입력하면 YYYYMMDD_result_raw.csv 및 YYYYMMDD_result.csv기반 그래프를 생성합니다.

main process1-2과정을 자동으로 진행합니다.
'''

from process1_tracks import main as run_tle
import subprocess
import sys
from datetime import datetime, timedelta

today = input("INPUT YYYYMMDD : ").strip()

dt = datetime.strptime(today, "%Y%m%d")
prev = (dt - timedelta(days=1)).strftime("%Y%m%d")

if __name__ == "__main__":
    # 1) TLE → epoch track CSV
    sys.argv = [
        "main.py",
        "--today", f"repository/tle_data/{today}.tle",
        "--prev",  f"repository/tle_data/{prev}.tle",
        "--out",   f"{today}_result_raw.csv",
    ]
    run_tle()

    # 2) dv/dm/d1/d2 계산
    subprocess.run([
        sys.executable,
        "process2_calculate.py",
        f"{today}_result_raw.csv",
        f"{today}_result.csv",
    ], check=True)
