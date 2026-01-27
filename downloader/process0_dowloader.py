'''
spacetrack_auth.txt파일에 spacetrack의 ID와 Password를 입력한 후 실행됩니다.
start day부터 end day의 전 날까지의 기간동안 spacettrack에 저장되어있는 decay_data message와 tip_data message를 다운받아 날짜별로 저장합니다.
SSL관련 내용은 GPT가 알려준 방법입니다.
'''
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import requests
import urllib3
import time



# SSL / Proxy CA settings
# 회사컴퓨터에서 접근이 안되어서, GPT가 추천한 아래 방법으로 접속하였습니다.
CA_BUNDLE = None  # 예: r"C:\Users\seoeunlee\certs\kari_proxy_ca.pem"
VERIFY_SSL = False  # 테스트: False, 장기 운용: True + CA_BUNDLE 권장

if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Space-Track endpoints
BASE = "https://www.space-track.org"
LOGIN_URL = f"{BASE}/ajaxauth/login"
QUERY_BASE = f"{BASE}/basicspacedata/query"


@dataclass(frozen=True)
class Auth:
    identity: str
    password: str


def load_auth_txt(path: str) -> Auth:
    kv: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    if "identity" not in kv or "password" not in kv:
        raise ValueError(f"Auth file must contain identity=... and password=... : {path}")
    return Auth(identity=kv["identity"], password=kv["password"])


def verify_arg():
    return CA_BUNDLE if (VERIFY_SSL and CA_BUNDLE) else VERIFY_SSL


def login(session: requests.Session, auth: Auth, timeout_s: int = 30) -> None:
    r = session.post(
        LOGIN_URL,
        data={"identity": auth.identity, "password": auth.password},
        timeout=timeout_s,
        verify=verify_arg(),
        allow_redirects=True,
    )
    r.raise_for_status()

    if not session.cookies:
        raise RuntimeError("Login seems to have failed: no cookies were set.")

    t = (r.text or "").lower()
    if "login" in t and "password" in t and "identity" in t:
        raise RuntimeError("Login page returned; login likely failed.")


def get_json(session: requests.Session, url: str, timeout_s: int = 60) -> List[Dict]:
    r = session.get(url, timeout=timeout_s, verify=verify_arg(), allow_redirects=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/json" not in ct:
        raise RuntimeError(f"Non-JSON response. CT={ct} URL={r.url} BODY={r.text[:300]}")
    data = r.json()
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def detect_date_field(sample_row: Dict, candidates: List[str]) -> str:
    keys = set(sample_row.keys())
    for c in candidates:
        if c in keys:
            return c

    # 마지막 안전장치(?)
    for k, v in sample_row.items():
        if isinstance(v, str) and len(v) >= 10 and v[4:5] == "-" and v[7:8] == "-":
            return k

    raise RuntimeError(f"Could not detect a date field. keys={sorted(sample_row.keys())[:60]}")


def build_range_url(cls: str, field: str, start_utc: datetime, end_utc: datetime) -> str:
    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end_utc.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{QUERY_BASE}/class/{cls}/{field}/{start_str}--{end_str}/format/json"


def write_csv(rows: List[Dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    if not rows:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fetch_tip_for_day(session: requests.Session, day_utc: date) -> List[Dict]:
    sample = get_json(session, f"{QUERY_BASE}/class/tip_data/limit/1/format/json")
    if not sample:
        return []
    date_field = detect_date_field(
        sample[0],
        candidates=["MESSAGE_EPOCH", "MSG_EPOCH", "EPOCH", "INSERT_DATE", "CREATED", "CREATE_DATE"],
    )

    start = datetime(day_utc.year, day_utc.month, day_utc.day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    rows = get_json(session, build_range_url("tip_data", date_field, start, end))
    for r in rows:
        r["query_day_utc"] = day_utc.isoformat()
        r["query_date_field"] = date_field
        r["message_type"] = "tip_data"
    return rows


def fetch_decay_for_day(session: requests.Session, day_utc: date) -> List[Dict]:
    sample = get_json(session, f"{QUERY_BASE}/class/decay_data/limit/1/format/json")
    if not sample:
        return []
    # decay_data 샘플 쿼리 문서에 DECAY_EPOCH가 등장하므로 1순위로 둠
    date_field = detect_date_field(
        sample[0],
        candidates=["DECAY_EPOCH", "EPOCH", "MESSAGE_EPOCH", "INSERT_DATE", "CREATED", "CREATE_DATE"],
    )

    start = datetime(day_utc.year, day_utc.month, day_utc.day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    rows = get_json(session, build_range_url("decay_data", date_field, start, end))
    for r in rows:
        r["query_day_utc"] = day_utc.isoformat()
        r["query_date_field"] = date_field
        r["message_type"] = "decay_data"
    return rows


def daterange(d0: date, d1_exclusive: date):
    cur = d0
    while cur < d1_exclusive:
        yield cur
        cur += timedelta(days=1)


def main():
    auth = load_auth_txt("spacetrack_auth.txt")
    tip_dir = os.path.join("..", "repository", "tip_data")
    decay_dir = os.path.join("..", "repository", "decay_data")
    os.makedirs(tip_dir, exist_ok=True)
    os.makedirs(decay_dir, exist_ok=True)

    # -------- 1년치 범위 설정 --------
    #start day부터 end day -1 일 까지의 tip_data, decay자료를 다운받습니다.
    start_day = date(2025, 1, 1)
    end_day_exclusive = date(2025, 1, 4)

    # 요청 사이 딜레이(서버 부담/차단 방지)
    request_delay_s = 0.6

    with requests.Session() as session:
        login(session, auth)

        for day in daterange(start_day, end_day_exclusive):
            ymd = day.strftime("%Y%m%d")
            tip_csv = os.path.join(tip_dir, f"tip_{ymd}.csv")
            decay_csv = os.path.join(decay_dir, f"decay_{ymd}.csv")

            # 이미 둘 다 있으면 스킵(중단 후 재실행 대비)
            if os.path.exists(tip_csv) and os.path.exists(decay_csv):
                print(f"[SKIP] {day.isoformat()} already downloaded")
                continue

            # TIP
            if not os.path.exists(tip_csv):
                tip_rows = fetch_tip_for_day(session, day)
                write_csv(tip_rows, tip_csv)
                print(f"[OK] {day.isoformat()} TIP={len(tip_rows)} -> {tip_csv}")
                time.sleep(request_delay_s)

            # DECAY
            if not os.path.exists(decay_csv):
                decay_rows = fetch_decay_for_day(session, day)
                write_csv(decay_rows, decay_csv)
                print(f"[OK] {day.isoformat()} DECAY={len(decay_rows)} -> {decay_csv}")
                time.sleep(request_delay_s)

if __name__ == "__main__":
    main()