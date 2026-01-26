* process0
spacetrack에서 원하는 기간동안의 decay message와 tip message를 다운받아 날짜별로 저장합니다.
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

* main
process1-2과정을 자동으로 진행합니다.
