# tle_epoch_subsat_radec.py
# -----------------------------------------------------------------------------
# 목적
# - TLE 한 개(OrbitT)를 입력으로 받아,
#   "TLE epoch 시각의 직하점(subsatellite point)에 해발 0m 관측자"를 가정하고
#   epoch ± 5초(총 10초 윈도우) 동안의 Topocentric RA/Dec 궤적을 계산/출력한다.
#
# 출력
# - CSV: out_csv (기본: epoch_track_10s.csv)
#   컬럼: satno, t_offset_s, jd, mjd, ra_deg, dec_deg, obs_lat_deg, obs_lon_deg
#
# 요구사항
# - 너의 포팅 sgp4.py 안에 아래가 존재해야 함:
#   - class Sgdp4Model with:
#       - init(OrbitT) -> imode (int)
#       - satpos_xyz(jd: float, want_vel: bool=False) -> (pos_xyz, vel_xyz, mode)
#       - attribute SGDP4_jd0 (TLE epoch JD)
#   - OrbitT dataclass
#   - SGDP4_DEEP_NORM / SGDP4_DEEP_RESN / SGDP4_DEEP_SYNC
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 너의 포팅 모듈 (파일명이 sgp4.py 라고 했던 기준)
from sgp4 import (
    Sgdp4Model,
    OrbitT,
    SGDP4_DEEP_NORM,
    SGDP4_DEEP_RESN,
    SGDP4_DEEP_SYNC,
)

# ----------------------------
# Constants
# ----------------------------
D2R = math.pi / 180.0
R2D = 180.0 / math.pi
XKMPER = 6378.135
FLAT = 1.0 / 298.257

MJD0_JD_OFFSET = 2400000.5


# ----------------------------
# Small vector / site types
# ----------------------------
@dataclass
class Site:
    lat_deg: float
    lon_deg: float
    alt_km: float


@dataclass
class Vec3:
    x: float
    y: float
    z: float


# ----------------------------
# Utils
# ----------------------------
def modulo(x: float, y: float) -> float:
    x = math.fmod(x, y)
    if x < 0.0:
        x += y
    return x


def mjd_to_jd(mjd: float) -> float:
    return mjd + MJD0_JD_OFFSET


def jd_to_mjd(jd: float) -> float:
    return jd - MJD0_JD_OFFSET


# ----------------------------
# GMST (match your C-style formula)
# ----------------------------
def gmst_deg(mjd: float) -> float:
    t = (mjd - 51544.5) / 36525.0
    g = (
        280.46061837
        + 360.98564736629 * (mjd - 51544.5)
        + t * t * (0.000387933 - t / 38710000.0)
    )
    return modulo(g, 360.0)


def dgmst_deg_per_day(mjd: float) -> float:
    t = (mjd - 51544.5) / 36525.0
    return 360.98564736629 + t * (0.000387933 - t / 38710000.0)


# ----------------------------
# Observer position (ECI-ish) as in your candidate_search.py
# ----------------------------
def obspos_xyz(mjd: float, site: Site) -> Tuple[Vec3, Vec3]:
    s = math.sin(site.lat_deg * D2R)
    ff = math.sqrt(1.0 - FLAT * (2.0 - FLAT) * s * s)
    gc = 1.0 / ff + site.alt_km / XKMPER
    gs = (1.0 - FLAT) * (1.0 - FLAT) / ff + site.alt_km / XKMPER

    theta = gmst_deg(mjd) + site.lon_deg
    dtheta = dgmst_deg_per_day(mjd) * D2R / 86400.0  # rad/s

    coslat = math.cos(site.lat_deg * D2R)
    sinlat = math.sin(site.lat_deg * D2R)
    costh = math.cos(theta * D2R)
    sinth = math.sin(theta * D2R)

    pos = Vec3(
        x=gc * coslat * costh * XKMPER,
        y=gc * coslat * sinth * XKMPER,
        z=gs * sinlat * XKMPER,
    )
    vel = Vec3(
        x=-gc * coslat * sinth * XKMPER * dtheta,
        y=gc * coslat * costh * XKMPER * dtheta,
        z=0.0,
    )
    return pos, vel


# ----------------------------
# Precession angles (match your C-style block)
#   NOTE: candidate_search.py used precession_angles(mjd, 51544.5) which is odd
#         but we keep the same call signature/behavior for consistency.
# ----------------------------
def precession_angles(mjd0: float, mjd: float) -> Tuple[float, float, float]:
    t0 = (mjd0 - 51544.5) / 36525.0
    t = (mjd - mjd0) / 36525.0

    zeta = (2306.2181 + 1.39656 * t0 - 0.000139 * t0 * t0) * t
    zeta += (0.30188 - 0.000344 * t0) * t * t + 0.017998 * t * t * t
    zeta *= D2R / 3600.0

    z = (2306.2181 + 1.39656 * t0 - 0.000139 * t0 * t0) * t
    z += (1.09468 + 0.000066 * t0) * t * t + 0.018203 * t * t * t
    z *= D2R / 3600.0

    theta = (2004.3109 - 0.85330 * t0 - 0.000217 * t0 * t0) * t
    theta += -(0.42665 + 0.000217 * t0) * t * t - 0.041833 * t * t * t
    theta *= D2R / 3600.0

    return zeta, z, theta


def _radec_from_topo_and_prec(
    dx: float, dy: float, dz: float,
    zeta: float, z: float, theta: float
) -> Tuple[float, float]:
    """
    Topocentric vector (dx,dy,dz) -> RA/Dec (deg)
    and then apply the same precession block ordering you used.
    """
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    if rng <= 0.0:
        return float("nan"), float("nan")

    ra_rad = modulo(math.atan2(dy, dx), 2.0 * math.pi)
    de_rad = math.asin(dz / rng)

    a = math.cos(de_rad) * math.sin(ra_rad + zeta)
    b = math.cos(theta) * math.cos(de_rad) * math.cos(ra_rad + zeta) - math.sin(theta) * math.sin(de_rad)
    c = math.sin(theta) * math.cos(de_rad) * math.cos(ra_rad + zeta) + math.cos(theta) * math.sin(de_rad)

    ra_deg = modulo((math.atan2(a, b) + z) * R2D, 360.0)
    de_deg = math.asin(c) * R2D
    return ra_deg, de_deg


# ----------------------------
# Subsatellite point at epoch (simple spherical Earth)
# ----------------------------
def subsatellite_point_from_xyz(pos: Vec3) -> Tuple[float, float]:
    """
    Satellite geocentric position (km) -> (lat_deg, lon_deg)
    - Simple spherical formula (geocentric latitude)
    - Longitude returned in [0, 360)
    """
    r = math.sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z)
    if r <= 0.0:
        return float("nan"), float("nan")

    lat = math.asin(pos.z / r) * R2D
    lon = math.atan2(pos.y, pos.x) * R2D
    lon = modulo(lon, 360.0)
    return lat, lon


# ----------------------------
# Main: epoch ±5s RA/Dec track (total 10s window)
# ----------------------------
def radec_track_epoch_pm5s_subsat(
    orb: OrbitT,
    half_window_s: float = 5.0,
    step_s: float = 1.0,
) -> List[Dict]:
    """
    Returns list of dict rows for CSV writing.
    """
    model = Sgdp4Model()
    imode = model.init(orb)

    if imode in (SGDP4_DEEP_NORM, SGDP4_DEEP_RESN, SGDP4_DEEP_SYNC):
        raise RuntimeError("Deep-space 모드(TLE period >= ~225min)는 현재 트랙 계산에서 제외")

    jd0 = float(model.SGDP4_jd0)
    mjd0 = jd_to_mjd(jd0)

    # epoch에서 위성 위치
    pos0, _vel0, _mode0 = model.satpos_xyz(jd0, want_vel=False)
    satpos0 = Vec3(pos0.x, pos0.y, pos0.z)

    # epoch 직하점 관측자(고도 0 km)
    obs_lat_deg, obs_lon_deg = subsatellite_point_from_xyz(satpos0)
    if not (math.isfinite(obs_lat_deg) and math.isfinite(obs_lon_deg)):
        raise RuntimeError("Failed to compute subsatellite point at epoch")

    site = Site(lat_deg=obs_lat_deg, lon_deg=obs_lon_deg, alt_km=0.0)

    rows: List[Dict] = []
    t = -half_window_s
    # total window is 2*half_window_s seconds; with step=1 -> 11 points (-5..+5)
    while t <= half_window_s + 1e-12:
        mjd = mjd0 + t / 86400.0
        jd = mjd_to_jd(mjd)

        pos, _vel, _mode = model.satpos_xyz(jd, want_vel=False)
        satpos = Vec3(pos.x, pos.y, pos.z)

        obspos, _obsvel = obspos_xyz(mjd, site)

        dx = satpos.x - obspos.x
        dy = satpos.y - obspos.y
        dz = satpos.z - obspos.z

        # precession: keep your existing convention (mjd, 51544.5)
        zeta, z, theta = precession_angles(mjd, 51544.5)

        ra_deg, dec_deg = _radec_from_topo_and_prec(dx, dy, dz, zeta, z, theta)

        az_deg, el_deg = azel_from_topocentric(dx, dy, dz, site) # substation 기준으로 잘 출력되나.. test용으로 넣어보았습ㄴ디ㅏ.
        print(
            f"[t={t:+5.1f}s] "
            f"RA={ra_deg:8.3f} deg  "
            f"Dec={dec_deg:7.3f} deg  "
            f"Az={az_deg:7.3f} deg  "
            f"El={el_deg:7.3f} deg"
        )

        rows.append(
            dict(
                satno=orb.satno,
                t_offset_s=float(t),
                jd=float(jd),
                mjd=float(mjd),
                ra_deg=float(ra_deg),
                dec_deg=float(dec_deg),
                obs_lat_deg=float(obs_lat_deg),
                obs_lon_deg=float(obs_lon_deg),
            )
        )

        t += step_s

    return rows



def _tle_epoch_to_year_doy(l1: str) -> Tuple[int, float]:
    yy = int(l1[18:20])
    doy = float(l1[20:32])
    return yy, doy


def _tle_bstar_to_float(l1: str) -> float:
    raw = l1[53:61]
    raw = raw.replace(" ", "0")
    mant = raw[0:6]
    exp = raw[6:8]
    try:
        m = float(mant) * 1e-5
        e = int(exp)
        return m * (10.0 ** e)
    except Exception:
        s = l1[53:61].strip()
        if not s:
            return 0.0
        if s[0] in "+-":
            mant2 = s[0] + "0." + s[1:6]
            exp2 = s[6:]
        else:
            mant2 = "0." + s[0:5]
            exp2 = s[5:]
        return float(mant2) * (10.0 ** int(exp2))


def orbit_from_tle_lines(l1: str, l2: str) -> OrbitT:
    satno = int(l1[2:7])
    yy, doy = _tle_epoch_to_year_doy(l1)
    bstar = _tle_bstar_to_float(l1)

    inc_deg = float(l2[8:16])
    raan_deg = float(l2[17:25])
    ecc = float("0." + l2[26:33].strip())
    argp_deg = float(l2[34:42])
    m_deg = float(l2[43:51])
    n_rev_day = float(l2[52:63])

    return OrbitT(
        satno=satno,
        ep_year=yy,  # init()에서 19xx/20xx 보정
        ep_day=doy,
        ecc=ecc,
        rev=n_rev_day,
        eqinc=inc_deg * D2R,
        ascn=raan_deg * D2R,
        argp=argp_deg * D2R,
        mnan=m_deg * D2R,
        bstar=bstar,
    )

def write_csv(rows: List[Dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if not rows:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            f.write("")
        return

    # stable header order
    fieldnames = [
        "satno",
        "t_offset_s",
        "jd",
        "mjd",
        "ra_deg",
        "dec_deg",
        "obs_lat_deg",
        "obs_lon_deg",
    ]
    # plus any extra keys (if you add later)
    extra = []
    seen = set(fieldnames)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                extra.append(k)
                seen.add(k)

    fieldnames = fieldnames + extra

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)



def azel_from_topocentric(dx: float, dy: float, dz: float, site: Site) -> Tuple[float, float]:
    """
    Topocentric vector (ECI-like, observer-centered) -> Azimuth, Elevation [deg]

    Azimuth:
      - North = 0 deg
      - East  = 90 deg
      - Range [0, 360)

    Elevation:
      - Horizon = 0 deg
      - Zenith  = 90 deg
    """
    lat = site.lat_deg * D2R

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)

    # ENU 변환
    east  = -math.sin(site.lon_deg * D2R) * dx + math.cos(site.lon_deg * D2R) * dy
    north = (
        -sin_lat * math.cos(site.lon_deg * D2R) * dx
        -sin_lat * math.sin(site.lon_deg * D2R) * dy
        +cos_lat * dz
    )
    up = (
        cos_lat * math.cos(site.lon_deg * D2R) * dx
        +cos_lat * math.sin(site.lon_deg * D2R) * dy
        +sin_lat * dz
    )

    horiz_dist = math.sqrt(east * east + north * north)

    az = math.degrees(math.atan2(east, north))
    if az < 0.0:
        az += 360.0

    el = math.degrees(math.atan2(up, horiz_dist))

    return az, el



# -----------------------------------------------------------------------------
# 사용 예시 (직접 실행할 때만 동작)
# -----------------------------------------------------------------------------
def main():

    l1 = "1 25544U 98067A   25137.12345678  .00012345  00000-0  10270-3 0  9991"
    l2 = "2 25544  51.6400 123.4567 0005000  45.0000 315.0000 15.50000000123456"
    orb = orbit_from_tle_lines(l1, l2)   # 너의 기존 함수

    rows = radec_track_epoch_pm5s_subsat(orb, half_window_s=5.0, step_s=0.1)
    write_csv(rows, "epoch_track_10s.csv")
    print("[OK] wrote epoch_track_10s.csv")


if __name__ == "__main__":
    main()
