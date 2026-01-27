#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

from pathlib import Path
import httpx
from spacetrack import SpaceTrackClient


@dataclass(frozen=True)
class Auth:
    identity: str
    password: str


def load_auth_txt(path: str) -> Auth:
    kv: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    if "identity" not in kv or "password" not in kv:
        raise ValueError("auth file must contain identity=... and password=...")
    return Auth(identity=kv["identity"], password=kv["password"])


def fix_missing_leading_zeros(line: str) -> str:
    line = re.sub(r"^1\s{5}", "1 0000", line)
    line = re.sub(r"^2\s{5}", "2 0000", line)
    line = re.sub(r"^1\s{4}", "1 000", line)
    line = re.sub(r"^2\s{4}", "2 000", line)
    line = re.sub(r"^1\s{3}", "1 00", line)
    line = re.sub(r"^2\s{3}", "2 00", line)
    line = re.sub(r"^1\s{2}", "1 0", line)
    line = re.sub(r"^2\s{2}", "2 0", line)
    return line


def parse_norad_from_line1(l1: str) -> Optional[int]:
    m = re.match(r"^1\s+(\d+)", l1)
    if not m:
        return None
    return int(m.group(1))


def main():
    AUTH_FILE = "spacetrack_auth.txt"

    # 저장 폴더 (없으면 생성)
    out_dir = Path("..", "repository","tle_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # now(UTC) 기준 파일명
    now_utc = datetime.now(timezone.utc)
    #out_file = out_dir / f"{now_utc.strftime('%Y%m%d_%H%M')}.tle" #다운 받은 시간까지 정확하게 파일명에 남기고 싶다면 이걸 키기
    out_file = out_dir / f"{now_utc.strftime('%Y%m%d')}.tle"

    auth = load_auth_txt(AUTH_FILE)

    # SSL 검증 끄기(임시). 가능하면 사내 CA로 verify 설정 추천.
    client = httpx.Client(verify=False)
    st = SpaceTrackClient(identity=auth.identity, password=auth.password, httpx_client=client)

    data = st.gp(
        iter_lines=True,
        epoch=">now-30",  # 최근 30일 epoch 레코드
        format="3le",
        # 필요하면(권장): decay_date="null-val" 로 on-orbit만
        # decay_date="null-val",
    )

    best: Dict[int, Tuple[str, str, str]] = {}
    buf = []

    for raw in data:
        s = (raw or "").strip("\r\n")
        if not s:
            continue
        buf.append(s)
        if len(buf) < 3:
            continue

        name, l1, l2 = buf[0], buf[1], buf[2]
        buf.clear()

        l1 = fix_missing_leading_zeros(l1)
        l2 = fix_missing_leading_zeros(l2)

        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue

        norad = parse_norad_from_line1(l1)
        if norad is None:
            continue

        # 위성당 1세트만 저장
        if norad in best:
            continue

        if name.startswith("0 "):
            name = name[2:]

        best[norad] = (name, l1, l2)

    with open(out_file, "w", encoding="utf-8") as f:
        for norad in sorted(best.keys()):
            name, l1, l2 = best[norad]
            f.write(name + "\n")
            f.write(l1 + "\n")
            f.write(l2 + "\n")

    print("unique sats:", len(best))
    print("saved:", str(out_file))


if __name__ == "__main__":
    main()
