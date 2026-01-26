# tle_epoch_subsat_radec.py
# -----------------------------------------------------------------------------
# 목적
# - TLE 한 개(OrbitT)를 입력으로 받아,
#   "TLE epoch 시각의 직하점(subsatellite point)에 해발 0m 관측자"를 가정하고
#   epoch ± 5초(총 10초 윈도우) 동안의 Topocentric RA/Dec + Az/El 궤적을 계산한다.
#
# 핵심 원칙(중요)
# - "직하점"은 ECEF(지구고정)에서 정의해야 함.
#   => 위성 ECI 위치를 GMST로 ECEF로 회전 -> ECEF에서 지오데식(lat/lon) 계산
# - Az/El은 ENU(지표 고정) 좌표계에서 계산해야 함.
#   => 위성/관측자 모두 ECEF에서 difference -> ENU 변환 -> Az/El
# - RA/Dec(Topocentric)는 관측자 기준 벡터를 ECI에서 만든 뒤 (또는 ECI로 변환한 뒤)
#   RA=atan2(y,x), Dec=asin(z/r)로 계산 (필요하면 precession 적용 가능)
#
# 출력
# - 콘솔: t_offset_s, RA/Dec, Az/El
# - CSV 저장은 옵션(원하면 write_csv 호출)
#
# 요구사항
# - 포팅 sgp4_test.py 안에 아래가 존재해야 함:
#   - class Sgdp4Model with:
#       - init(OrbitT) -> imode (int)
#       - satpos_xyz(jd: float, want_vel: bool=False) -> (pos_xyz, vel_xyz, mode)
#       - attribute SGDP4_jd0 (TLE epoch JD)
#   - OrbitT dataclass
#   - SGDP4_DEEP_NORM / SGDP4_DEEP_RESN / SGDP4_DEEP_SYNC
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sgp4_test import (
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

MJD0_JD_OFFSET = 2400000.5

# WGS84
WGS84_A = 6378.137  # km
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# ----------------------------
# Types
# ----------------------------
@dataclass
class Site:
    lat_deg: float
    lon_deg: float  # East-positive, degrees, wrapped to [0,360)
    alt_km: float


@dataclass
class Vec3:
    x: float
    y: float
    z: float


# ----------------------------
# Time helpers
# ----------------------------
def jd_to_mjd(jd: float) -> float:
    return jd - MJD0_JD_OFFSET


def mjd_to_jd(mjd: float) -> float:
    return mjd + MJD0_JD_OFFSET


def modulo(x: float, y: float) -> float:
    x = math.fmod(x, y)
    if x < 0.0:
        x += y
    return x


# ----------------------------
# GMST (same style as your code)
# ----------------------------
def gmst_deg(mjd: float) -> float:
    t = (mjd - 51544.5) / 36525.0
    g = (
        280.46061837
        + 360.98564736629 * (mjd - 51544.5)
        + t * t * (0.000387933 - t / 38710000.0)
    )
    return modulo(g, 360.0)


# ----------------------------
# ECI <-> ECEF rotation (GMST about +Z)
#   Conventions vary; this pair is self-consistent.
# ----------------------------
def eci_to_ecef(r_eci: Vec3, mjd: float) -> Vec3:
    th = gmst_deg(mjd) * D2R
    c = math.cos(th)
    s = math.sin(th)
    # r_ecef = R3(+th) r_eci
    return Vec3(
        x= c * r_eci.x + s * r_eci.y,
        y=-s * r_eci.x + c * r_eci.y,
        z= r_eci.z,
    )


def ecef_to_eci(r_ecef: Vec3, mjd: float) -> Vec3:
    th = gmst_deg(mjd) * D2R
    c = math.cos(th)
    s = math.sin(th)
    # inverse rotation: R3(-th)
    return Vec3(
        x= c * r_ecef.x - s * r_ecef.y,
        y= s * r_ecef.x + c * r_ecef.y,
        z= r_ecef.z,
    )


# ----------------------------
# ECEF <-> Geodetic (WGS84)
# ----------------------------
def ecef_to_geodetic_latlon(r_ecef: Vec3) -> Tuple[float, float]:
    """
    ECEF (km) -> geodetic lat/lon (deg), WGS84
    lon wrapped to [0,360)
    """
    x, y, z = r_ecef.x, r_ecef.y, r_ecef.z
    lon = math.atan2(y, x)

    p = math.sqrt(x * x + y * y)
    if p < 1e-12:
        lat = math.copysign(math.pi / 2.0, z)
        return lat * R2D, modulo(lon * R2D, 360.0)

    # iterative solve for geodetic latitude
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sinlat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sinlat * sinlat)
        lat_new = math.atan2(z + WGS84_E2 * N * sinlat, p)
        if abs(lat_new - lat) < 1e-14:
            lat = lat_new
            break
        lat = lat_new

    return lat * R2D, modulo(lon * R2D, 360.0)


def geodetic_to_ecef(site: Site) -> Vec3:
    """
    Geodetic lat/lon/alt -> ECEF (km), WGS84
    """
    lat = site.lat_deg * D2R
    lon = site.lon_deg * D2R
    sinlat = math.sin(lat)
    coslat = math.cos(lat)
    sinlon = math.sin(lon)
    coslon = math.cos(lon)

    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sinlat * sinlat)
    x = (N + site.alt_km) * coslat * coslon
    y = (N + site.alt_km) * coslat * sinlon
    z = (N * (1.0 - WGS84_E2) + site.alt_km) * sinlat
    return Vec3(x, y, z)


# ----------------------------
# RA/Dec from topocentric vector in ECI
# ----------------------------
def radec_from_topocentric_eci(dr_eci: Vec3) -> Tuple[float, float]:
    """
    dr_eci: observer->satellite vector in ECI (km)
    Returns: (ra_deg in [0,360), dec_deg)
    """
    x, y, z = dr_eci.x, dr_eci.y, dr_eci.z
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0:
        return float("nan"), float("nan")

    ra = modulo(math.atan2(y, x) * R2D, 360.0)
    dec = math.asin(z / r) * R2D
    return ra, dec


# ----------------------------
# Az/El from topocentric vector in ECEF via ENU
# ----------------------------
def azel_from_topocentric_ecef(dr_ecef: Vec3, site: Site) -> Tuple[float, float]:
    """
    dr_ecef: observer->satellite vector in ECEF (km)
    site: observer geodetic lat/lon
    Returns:
      az_deg: North=0, East=90, [0,360)
      el_deg: horizon=0, zenith=90
    """
    lat = site.lat_deg * D2R
    lon = site.lon_deg * D2R
    sinlat = math.sin(lat)
    coslat = math.cos(lat)
    sinlon = math.sin(lon)
    coslon = math.cos(lon)

    # ECEF -> ENU
    # [e]   [-sinlon,        coslon,       0][dx]
    # [n] = [-sinlat*coslon, -sinlat*sinlon, coslat][dy]
    # [u]   [ coslat*coslon,  coslat*sinlon, sinlat][dz]
    dx, dy, dz = dr_ecef.x, dr_ecef.y, dr_ecef.z
    east  = -sinlon * dx + coslon * dy
    north = -sinlat * coslon * dx - sinlat * sinlon * dy + coslat * dz
    up    =  coslat * coslon * dx + coslat * sinlon * dy + sinlat * dz

    horiz = math.sqrt(east * east + north * north)
    az = math.degrees(math.atan2(east, north))
    az = az + 360.0 if az < 0.0 else az
    el = math.degrees(math.atan2(up, horiz))
    return az, el


# ----------------------------
# Subsatellite point at epoch (ECEF geodetic)
# ----------------------------
def subsatellite_site_from_ecef(sat_ecef: Vec3) -> Site:
    """
    sat_ecef: satellite position in ECEF (km)
    return: subsatellite Site at sea level (alt=0)
    """
    lat_deg, lon_deg = ecef_to_geodetic_latlon(sat_ecef)
    return Site(lat_deg=lat_deg, lon_deg=lon_deg, alt_km=0.0)



# ----------------------------
# Main: epoch ±5s track
# ----------------------------
def track_epoch_pm5s_pair(
    orb_today: OrbitT,
    orb_prev: OrbitT,
    half_window_s: float = 5.0,
    step_s: float = 0.1,
    print_console: bool = True,
) -> Tuple[List[Dict], List[Dict]]:

    # ----------------------------
    # 1. Init models
    # ----------------------------
    model_today = Sgdp4Model()
    imode_today = model_today.init(orb_today)

    model_prev = Sgdp4Model()
    imode_prev = model_prev.init(orb_prev)

    for imode in (imode_today, imode_prev):
        if imode in (SGDP4_DEEP_NORM, SGDP4_DEEP_RESN, SGDP4_DEEP_SYNC):
            raise RuntimeError("Deep-space 모드 제외")

    # ----------------------------
    # 2. Epoch check (VERY IMPORTANT)
    # ----------------------------
    jd0_today = float(model_today.SGDP4_jd0)
    jd0_prev  = float(model_prev.SGDP4_jd0)

    print("\n[DEBUG] Epoch check")
    print(f"  jd0_today = {jd0_today:.9f}")
    print(f"  jd0_prev  = {jd0_prev:.9f}")
    print(f"  diff_sec  = {(jd0_today - jd0_prev) * 86400.0:.3f} s")

    # 기준 시간축 = today epoch
    mjd0 = jd_to_mjd(jd0_today)

    # ----------------------------
    # 3. Satellite position at respective epochs
    # ----------------------------
    pos0_today, _, _ = model_today.satpos_xyz(jd0_today, want_vel=False)
    pos0_prev,  _, _ = model_prev.satpos_xyz(jd0_today,  want_vel=False)
    print('postoday', 'poseprev', pos0_today, pos0_prev)

    sat0_today_eci = Vec3(pos0_today.x, pos0_today.y, pos0_today.z)
    sat0_prev_eci  = Vec3(pos0_prev.x,  pos0_prev.y,  pos0_prev.z)

    d0 = math.sqrt(
        (sat0_today_eci.x - sat0_prev_eci.x)**2 +
        (sat0_today_eci.y - sat0_prev_eci.y)**2 +
        (sat0_today_eci.z - sat0_prev_eci.z)**2
    )

    print("\n[DEBUG] Satellite position at own epochs")
    print(f"  |r_today(epoch_today) - r_prev(epoch_prev)| = {d0:.3f} km")

    # ----------------------------
    # 4. Observer site (TODAY subsatellite, fixed)
    # ----------------------------

    # 1) TLE에서 들어온 경사각 확인 (rad/deg)
    print("[DEBUG] inc check")
    print("  eqinc(rad) =", orb_today.eqinc)
    print("  eqinc(deg) =", orb_today.eqinc * 180.0 / math.pi)

    # 2) epoch 위성 ECI 벡터 확인
    print("[DEBUG] sat0_today_eci")
    print(f"  x={sat0_today_eci.x:.3f} y={sat0_today_eci.y:.3f} z={sat0_today_eci.z:.3f}")

    # 3) 그 벡터로부터 '지오데식 말고' 단순 지구중심 위도(geocentric lat) 확인
    rho = math.sqrt(sat0_today_eci.x ** 2 + sat0_today_eci.y ** 2)
    lat_gc = math.degrees(math.atan2(sat0_today_eci.z, rho))
    print(f"[DEBUG] geocentric lat from ECI = {lat_gc:.6f} deg")



    # Satellite at epoch (ECI)
    pos0_today, _, _ = model_today.satpos_xyz(jd0_today, want_vel=False)
    sat0_today_eci = Vec3(pos0_today.x, pos0_today.y, pos0_today.z)

    # ECI -> ECEF at epoch time
    sat0_today_ecef = eci_to_ecef(sat0_today_eci, mjd0)

    # Subsatellite site (NOW correct)
    site = subsatellite_site_from_ecef(sat0_today_ecef)

    obs_ecef_fixed = geodetic_to_ecef(site)

    print("[DEBUG] subsat lat/lon from ECEF")
    print(f"  lat = {site.lat_deg:.6f} deg")
    print(f"  lon = {site.lon_deg:.6f} deg")

    rows_today: List[Dict] = []
    rows_prev: List[Dict] = []

    # ----------------------------
    # 5. Time loop
    # ----------------------------
    t = -half_window_s
    while t <= half_window_s + 1e-12:
        mjd = mjd0 + t / 86400.0
        jd  = mjd_to_jd(mjd)

        # propagate both at SAME jd
        pos_today, _, _ = model_today.satpos_xyz(jd, want_vel=False)
        pos_prev,  _, _ = model_prev.satpos_xyz(jd, want_vel=False)

        sat_today_eci = Vec3(pos_today.x, pos_today.y, pos_today.z)
        sat_prev_eci  = Vec3(pos_prev.x,  pos_prev.y,  pos_prev.z)

        # --- ECI separation check (KEY)
        d_eci = math.sqrt(
            (sat_today_eci.x - sat_prev_eci.x)**2 +
            (sat_today_eci.y - sat_prev_eci.y)**2 +
            (sat_today_eci.z - sat_prev_eci.z)**2
        )


        # --- observer ECI
        obs_eci = ecef_to_eci(obs_ecef_fixed, mjd)

        # --- topocentric vectors
        dr_today_eci = Vec3(
            sat_today_eci.x - obs_eci.x,
            sat_today_eci.y - obs_eci.y,
            sat_today_eci.z - obs_eci.z,
        )
        dr_prev_eci = Vec3(
            sat_prev_eci.x - obs_eci.x,
            sat_prev_eci.y - obs_eci.y,
            sat_prev_eci.z - obs_eci.z,
        )

        # angular results
        ra_t, dec_t = radec_from_topocentric_eci(dr_today_eci)
        ra_p, dec_p = radec_from_topocentric_eci(dr_prev_eci)

        # --- for Az/El: need satellite ECEF and topocentric ECEF vector
        sat_today_ecef = eci_to_ecef(sat_today_eci, mjd)
        sat_prev_ecef  = eci_to_ecef(sat_prev_eci,  mjd)

        dr_today_ecef = Vec3(
            sat_today_ecef.x - obs_ecef_fixed.x,
            sat_today_ecef.y - obs_ecef_fixed.y,
            sat_today_ecef.z - obs_ecef_fixed.z,
        )
        dr_prev_ecef = Vec3(
            sat_prev_ecef.x - obs_ecef_fixed.x,
            sat_prev_ecef.y - obs_ecef_fixed.y,
            sat_prev_ecef.z - obs_ecef_fixed.z,
        )

        az_t, el_t = azel_from_topocentric_ecef(dr_today_ecef, site)
        az_p, el_p = azel_from_topocentric_ecef(dr_prev_ecef,  site)

        print(
            f"[t={t:+6.2f}s] "
            f"TODAY RA={ra_t:9.4f} Dec={dec_t:9.4f} Az={az_t:9.4f} El={el_t:9.4f} | "
            f"PREV  RA={ra_p:9.4f} Dec={dec_p:9.4f} Az={az_p:9.4f} El={el_p:9.4f}"
        )

        t += step_s



    return rows_today, rows_prev



# ----------------------------
# Optional CSV writer
# ----------------------------
def write_csv(rows: List[Dict], out_csv: str) -> None:
    if not rows:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# Minimal TLE -> OrbitT parser (kept from your style)
# ----------------------------
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
        ep_year=yy,  # init()에서 19xx/20xx 보정된다고 가정
        ep_day=doy,
        ecc=ecc,
        rev=n_rev_day,
        eqinc=inc_deg * D2R,
        ascn=raan_deg * D2R,
        argp=argp_deg * D2R,
        mnan=m_deg * D2R,
        bstar=bstar,
    )
