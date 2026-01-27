# skyfield_tle_epoch_subsat_track_csv.py
# -----------------------------------------------------------------------------
# 목적
# - "today TLE" 기준으로: 각 위성(NORAD ID)의 TLE epoch 시각(t0)에서
#   "epoch 직하점(subsatellite point)의 해발 0m 관측지"를 가정하고
#   t0 ± 5초 구간을 1초 간격으로 (Topocentric) RA/Dec + Az/El을 계산
# - "prev-day TLE 목록"에서 같은 NORAD ID가 있으면:
#   prev TLE epoch와 today TLE epoch 차이가 90분(=5400s) 이상일 때만
#   동일한 시각(t0±5s)에 대해 prev TLE로도 전파하여 비교값 계산
# - CSV 출력:
#   norad_id, object_name, epoch_time_utc,
#   site_lat_deg, site_lon_deg
#   t_offset_s,
#   today_ra_deg, today_dec_deg, today_az_deg, today_el_deg,
#   prev_ra_deg,  prev_dec_deg,  prev_az_deg,  prev_el_deg,
#   prev_epoch_time_utc, epoch_diff_sec
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import math

from typing import Dict, List, Optional, Tuple
from skyfield.api import EarthSatellite, load, wgs84


# --------------------------------------------
# Helpers
# --------------------------------------------
def _safe_float(x: float) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _time_iso_utc(t) -> str:
    # Skyfield Time -> ISO-like UTC string
    # (utc_iso() produces "YYYY-MM-DDTHH:MM:SSZ")
    try:
        return t.utc_iso()
    except Exception:
        # fallback
        y, mo, d, hh, mm, ss = t.utc
        # ss may be float
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{int(hh):02d}:{int(mm):02d}:{ss:06.3f}Z"


def parse_tle_file(path: str, ts) -> Dict[int, Dict]:
    """
    return dict keyed by NORAD ID:
      {
        satno: {
          "name": str,
          "l1": str,
          "l2": str,
          "sat": EarthSatellite,
          "epoch": Time
        },
        ...
      }
    """
    out: Dict[int, Dict] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("1 ") and (i + 1) < len(lines) and lines[i + 1].lstrip().startswith("2 "):
            name = ""  # no name line
            l1 = lines[i].strip()
            l2 = lines[i + 1].strip()
            i += 2
        else:
            # name + line1 + line2
            if (i + 2) >= len(lines):
                break
            name = lines[i].strip("0 ")
            l1 = lines[i + 1].strip()
            l2 = lines[i + 2].strip()
            i += 3

        # basic validation
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue

        # NORAD ID columns 3-7 in TLE line1 (0-index [2:7])
        try:
            satno = int(l1[2:7])
        except Exception:
            continue

        try:
            sat = EarthSatellite(l1, l2, name=name, ts=ts)
            epoch = sat.epoch
        except Exception:
            continue

        out[satno] = {"name": name, "l1": l1, "l2": l2, "sat": sat, "epoch": epoch}

    return out


def find_prev_candidate(prev_map: Dict[int, Dict], satno: int) -> Optional[Dict]:
    """
    prev TLE 파일에 satno가 있으면 그 항목 반환.
    (여러 개를 지원하려면 이 구조를 리스트로 바꾸면 됨.)
    """
    return prev_map.get(satno)


def subsat_site_from_sat_at_epoch(sat: EarthSatellite):
    """
    today TLE의 epoch 시각에서 위성 직하점(subpoint) -> 관측지(해발 0m)
    returns: (site, lat_deg, lon_deg)
    """
    t0 = sat.epoch
    sp = sat.at(t0).subpoint()
    lat_deg = sp.latitude.degrees
    lon_deg = sp.longitude.degrees  # East-positive, [-180,180] range typically
    # 해발 0m로 고정
    site = wgs84.latlon(latitude_degrees=lat_deg, longitude_degrees=lon_deg, elevation_m=0.0)
    #print(site)
    return site, lat_deg, lon_deg


def topocentric_angles(sat: EarthSatellite, t, site):
    """
    returns (ra_deg, dec_deg, az_deg, el_deg)
    - RA/Dec: topocentric apparent RA/Dec
    - Az/El: topocentric azimuth/elevation
    """
    top = (sat - site).at(t)

    # RA/Dec
    ra, dec, _ = top.radec()  # by default apparent
    ra_deg = ra.hours * 15.0
    dec_deg = dec.degrees

    # Az/El
    az, alt, _ = top.altaz()
    az_deg = az.degrees
    el_deg = alt.degrees
    return ra_deg, dec_deg, az_deg, el_deg

def mean_altitude_km_from_tle(sat: EarthSatellite) -> float:
    m = sat.model

    n_rad_per_min = getattr(m, "no_kozai", None)
    if n_rad_per_min is None:
        n_rad_per_min = getattr(m, "no", float("nan"))  # fallback

    n_rad_per_min = float(n_rad_per_min)
    if (not math.isfinite(n_rad_per_min)) or n_rad_per_min <= 0:
        return float("nan")

    mu_km3_s2 = 398600.4418
    re_km = 6378.137
    n_rad_per_s = n_rad_per_min / 60.0
    a_km = (mu_km3_s2 / (n_rad_per_s ** 2)) ** (1.0 / 3.0)
    return a_km - re_km


def is_leo_by_mean_altitude(sat: EarthSatellite, max_alt_km=2000.0) -> bool:
    h_km = mean_altitude_km_from_tle(sat)
    if not math.isfinite(h_km):
        return False
    return h_km <= max_alt_km



def tle_elements_from_sat(sat: EarthSatellite) -> Tuple[float, float, float, float, float, float]:

    m = sat.model  # sgp4 model object

    bstar = float(getattr(m, "bstar", float("nan")))

    inc_deg  = float(getattr(m, "inclo", float("nan"))) * 180.0 / math.pi
    raan_deg = float(getattr(m, "nodeo", float("nan"))) * 180.0 / math.pi
    argp_deg = float(getattr(m, "argpo", float("nan"))) * 180.0 / math.pi
    ma_deg   = float(getattr(m, "mo",    float("nan"))) * 180.0 / math.pi

    # mean motion: sgp4는 보통 rad/min 단위(no_kozai)를 가짐
    n_rad_per_min = float(getattr(m, "no_kozai", float("nan")))
    mm_rev_per_day = n_rad_per_min * (1440.0 / (2.0 * math.pi))  # rev/day

    return bstar, inc_deg, raan_deg, argp_deg, ma_deg, mm_rev_per_day



# --------------------------------------------
# Main
# --------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", required=True, help="today TLE file path")
    ap.add_argument("--prev", required=True, help="prev-day TLE file path")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--window_s", type=int, default=5, help="half window seconds (default=5)")
    ap.add_argument("--step_s", type=int, default=1, help="step seconds (default=1)")
    ap.add_argument("--min_epoch_diff_s", type=int, default=5400, help="min epoch diff to compute prev-track (default=5400 = 90min)")
    args = ap.parse_args()

    ts = load.timescale()

    today_map = parse_tle_file(args.today, ts)
    prev_map = parse_tle_file(args.prev, ts)

    fieldnames = [
        "norad_id",
        "epoch_time_utc",
        "site_lat_deg",
        "site_lon_deg",
        "t_offset_s",
        "today_ra_deg",
        "today_dec_deg",
        "today_az_deg",
        "today_el_deg",
        "prev_ra_deg",
        "prev_dec_deg",
        "prev_az_deg",
        "prev_el_deg",
        "prev_epoch_time_utc",
        "epoch_diff_sec",
        "Bstar",
        "Inc_deg",
        "RAAN_deg",
        "AP_deg",
        "MA_deg",
        "MM_rev/day",
        "MeanAlt_km",
        "object_name",
    ]

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for satno, item_today in today_map.items():
            sat_today: EarthSatellite = item_today["sat"]
            bstar, inc_deg, raan_deg, ap_deg, ma_deg, mm_rpd = tle_elements_from_sat(sat_today)

            mean_alt_km = mean_altitude_km_from_tle(sat_today)
            if not is_leo_by_mean_altitude(sat_today, max_alt_km=2000.0):
                continue

            name = item_today["name"] or ""
            t0 = sat_today.epoch

            # 관측지 = today epoch에서의 직하점, 해발 0m
            site, site_lat_deg, site_lon_deg = subsat_site_from_sat_at_epoch(sat_today)

            # prev 후보
            prev_item = find_prev_candidate(prev_map, satno)
            sat_prev: Optional[EarthSatellite] = None
            prev_epoch_iso = ""
            epoch_diff_sec = float("nan")
            use_prev = False

            if prev_item is not None:
                sat_prev = prev_item["sat"]
                tprev = sat_prev.epoch
                epoch_diff_sec = abs((t0.tt - tprev.tt) * 86400.0)
                prev_epoch_iso = _time_iso_utc(tprev)
                use_prev = (epoch_diff_sec >= args.min_epoch_diff_s)

            # t0 ± window, step
            for dt in range(-args.window_s, args.window_s + 1, args.step_s):
                t = ts.tt_jd(t0.tt + dt / 86400.0)

                # today angles
                try:
                    ra_t, dec_t, az_t, el_t = topocentric_angles(sat_today, t, site)
                except Exception:
                    ra_t = dec_t = az_t = el_t = float("nan")

                # prev angles (only if epoch diff >= threshold and sat exists)
                if use_prev and sat_prev is not None:
                    try:
                        ra_p, dec_p, az_p, el_p = topocentric_angles(sat_prev, t, site)
                    except Exception:
                        ra_p = dec_p = az_p = el_p = float("nan")
                else:
                    ra_p = dec_p = az_p = el_p = float("nan")

                w.writerow(
                    {
                        "norad_id": satno,
                        "epoch_time_utc": _time_iso_utc(t0),
                        "site_lat_deg": f"{site_lat_deg:.4f}",
                        "site_lon_deg": f"{site_lon_deg:.4f}",
                        "t_offset_s": dt,
                        "today_ra_deg": f"{_safe_float(ra_t):.4f}",
                        "today_dec_deg": f"{_safe_float(dec_t):.4f}",
                        "today_az_deg": f"{_safe_float(az_t):.4f}",
                        "today_el_deg": f"{_safe_float(el_t):.4f}",
                        "prev_ra_deg": f"{_safe_float(ra_p):.4f}",
                        "prev_dec_deg": f"{_safe_float(dec_p):.4f}",
                        "prev_az_deg": f"{_safe_float(az_p):.4f}",
                        "prev_el_deg": f"{_safe_float(el_p):.4f}",
                        "prev_epoch_time_utc": prev_epoch_iso,
                        "epoch_diff_sec": f"{_safe_float(epoch_diff_sec):.3f}",
                        #TLE에서 얻을 수 있는 정보들 추가함!!
                        "Bstar": f"{_safe_float(bstar):.6e}",
                        "Inc_deg": f"{_safe_float(inc_deg):.4f}",
                        "RAAN_deg": f"{_safe_float(raan_deg):.4f}",
                        "AP_deg": f"{_safe_float(ap_deg):.4f}",
                        "MA_deg": f"{_safe_float(ma_deg):.4f}",
                        "MM_rev/day": f"{_safe_float(mm_rpd):.8f}",
                        "MeanAlt_km": f"{_safe_float(mean_alt_km):.4f}",
                        "object_name": name,
                    }
                )
