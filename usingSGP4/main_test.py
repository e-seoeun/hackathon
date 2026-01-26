from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

from usingSGP4.tle_epoch_subsat_topocentric_track import orbit_from_tle_lines, track_epoch_pm5s_pair

MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137
DAY_S = 86400.0


def _tle_epoch_to_year_doy(l1: str) -> Tuple[int, float]:
    yy = int(l1[18:20])
    doy = float(l1[20:32])
    return yy, doy


def tle_year_doy_to_mjd(yy: int, doy: float) -> float:
    year = 1900 + yy if yy >= 57 else 2000 + yy

    def jd_from_ymd(y: int, m: int, d: int) -> float:
        a = (14 - m) // 12
        yy2 = y + 4800 - a
        mm2 = m + 12 * a - 3
        jdn = d + ((153 * mm2 + 2) // 5) + 365 * yy2 + (yy2 // 4) - (yy2 // 100) + (yy2 // 400) - 32045
        return float(jdn) - 0.5

    jd_jan1 = jd_from_ymd(year, 1, 1)
    mjd_jan1 = jd_jan1 - 2400000.5
    return mjd_jan1 + (doy - 1.0)


def epoch_diff_minutes(l1_a: str, l1_b: str) -> float:
    yy_a, doy_a = _tle_epoch_to_year_doy(l1_a)
    yy_b, doy_b = _tle_epoch_to_year_doy(l1_b)
    mjd_a = tle_year_doy_to_mjd(yy_a, doy_a)
    mjd_b = tle_year_doy_to_mjd(yy_b, doy_b)
    return abs(mjd_b - mjd_a) * 1440.0


def mean_motion_rev_day(l2: str) -> float:
    return float(l2[52:63])


def approx_alt_km_from_mean_motion(n_rev_day: float) -> float:
    n_rad_s = n_rev_day * 2.0 * math.pi / DAY_S
    a_km = (MU_EARTH_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    return a_km - R_EARTH_KM


@dataclass
class TLE3:
    name: str
    l1: str
    l2: str

    @property
    def satno(self) -> int:
        return int(self.l1[2:7])


def iter_tle3(path: str) -> Iterator[TLE3]:
    name_buf: str = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue

        if ln.startswith("0 "):
            name_buf = ln[2:].strip()
            i += 1
            continue

        if ln.startswith("1 ") and i + 1 < n and lines[i + 1].lstrip().startswith("2 "):
            l1 = lines[i].strip()
            l2 = lines[i + 1].strip()
            name = name_buf if name_buf else f"SAT-{int(l1[2:7])}"
            yield TLE3(name=name, l1=l1, l2=l2)
            name_buf = ""
            i += 2
            continue

        i += 1


def build_prev_map(prev_path: str) -> Dict[int, TLE3]:
    m: Dict[int, TLE3] = {}
    for tle in iter_tle3(prev_path):
        m[tle.satno] = tle
    return m


def write_csv(rows: List[dict], out_csv: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def enrich_rows(rows: List[dict], satno: int, tag: str, epoch_diff_min: float, alt_km_est: float) -> List[dict]:
    out = []
    for r in rows:
        rr = dict(r)
        rr["satno"] = satno
        rr["tag"] = tag            # "today" / "prev"
        rr["epoch_diff_min"] = round(float(epoch_diff_min), 3)
        rr["alt_km_est"] = round(float(alt_km_est), 3)
        out.append(rr)
    return out


def process_two_days(
    today_path: str,
    prev_path: str,
    out_dir: str = "tracks_out",
    alt_max_km: float = 2000.0,
    epoch_min_diff_min: float = 90.0,
    half_window_s: float = 5.0,
    step_s: float = 0.1,
    max_today_samples: int = 5,   # ✅ 오늘 기준 최대 5개만 시험
) -> None:
    prev_map = build_prev_map(prev_path)
    os.makedirs(out_dir, exist_ok=True)

    n_seen_leo_today = 0
    n_written = 0

    for tle_today in iter_tle3(today_path):
        n_rev_day = mean_motion_rev_day(tle_today.l2)
        alt_km_est = approx_alt_km_from_mean_motion(n_rev_day)
        if alt_km_est > alt_max_km:
            continue

        # ✅ "오늘 LEO 후보" 기준으로 5개만 테스트
        n_seen_leo_today += 1
        if n_seen_leo_today > max_today_samples:
            break

        satno = tle_today.satno
        tle_prev = prev_map.get(satno)
        if tle_prev is None:
            print(f"[SKIP] satno={satno}: prev day not found")
            continue

        diff_min = epoch_diff_minutes(tle_prev.l1, tle_today.l1)
        if diff_min < epoch_min_diff_min:
            print(f"[SKIP] satno={satno}: epoch diff {diff_min:.2f} min < {epoch_min_diff_min}")
            continue

        print(f"[RUN] satno={satno} alt~{alt_km_est:.1f} km  epoch_diff={diff_min:.2f} min")

        orb_today = orbit_from_tle_lines(tle_today.l1, tle_today.l2)
        orb_prev  = orbit_from_tle_lines(tle_prev.l1, tle_prev.l2)

        rows_today, rows_prev = track_epoch_pm5s_pair(
            orb_today=orb_today,
            orb_prev=orb_prev,
            half_window_s=5.0,
            step_s=0.1,
            print_console=False,
        )

        rows_today = enrich_rows(rows_today, satno, "today", diff_min, alt_km_est)
        rows_prev  = enrich_rows(rows_prev,  satno, "prev",  diff_min, alt_km_est)

        out_today = os.path.join(out_dir, f"{satno}_today.csv")
        out_prev  = os.path.join(out_dir, f"{satno}_prev.csv")

        write_csv(rows_today, out_today)
        write_csv(rows_prev, out_prev)

        print(f"  wrote: {out_today}")
        print(f"  wrote: {out_prev}")
        n_written += 1

    print(f"[DONE] tested_today_leo={n_seen_leo_today}, written_pairs={n_written}")


def main():
    today_path = os.path.join("../repository", "tle_data", "20250517.tle")
    prev_path  = os.path.join("../repository", "tle_data", "20250516.tle")

    process_two_days(
        today_path=today_path,
        prev_path=prev_path,
        out_dir="../tracks_out",
        alt_max_km=2000.0,
        epoch_min_diff_min=90.0,
        half_window_s=5.0,
        step_s=0.1,
        max_today_samples=5,
    )


if __name__ == "__main__":
    main()
