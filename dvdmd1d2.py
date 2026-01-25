import csv
import math
from collections import defaultdict
from typing import Dict, Tuple

IN_CSV = "20250517_result.csv"
OUT_CSV = "20250517_result_dv_dm_d1d2.csv"

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

def to_float(x: str) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "unset"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

def is_finite(x: float) -> bool:
    return (x is not None) and math.isfinite(x)

def wrap360(deg: float) -> float:
    deg = deg % 360.0
    if deg < 0:
        deg += 360.0
    return deg

def shortest_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    # a - b in (-180, +180]
    d = (a_deg - b_deg) % 360.0
    if d > 180.0:
        d -= 360.0
    return d

def angsep_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = ra1_deg * D2R
    dec1 = dec1_deg * D2R
    ra2 = ra2_deg * D2R
    dec2 = dec2_deg * D2R
    dra = ra2 - ra1
    cosd = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(dra)
    cosd = max(-1.0, min(1.0, cosd))
    return math.acos(cosd) * R2D


def bearing_dm_deg(
    ra_m1_deg: float, dec_m1_deg: float,
    ra_0_deg: float,  dec_0_deg: float,
    ra_p1_deg: float, dec_p1_deg: float,
    dt_total_s: float = 2.0
) -> float:
    """
    epoch(0s) 기준 진행방향 DM
    - 입력: (-1s, 0s, +1s) RA/Dec
    - 출력: DM [deg], 0=N, 90=E, [0,360)
    """

    # 중앙차분 (각속도 벡터)
    dra = shortest_angle_diff_deg(ra_p1_deg, ra_m1_deg)
    ddec = dec_p1_deg - dec_m1_deg

    # 국소접평면에서의 속도 성분 (deg/s)
    v_east  = (dra * math.cos(dec_0_deg * D2R)) / dt_total_s
    v_north = ddec / dt_total_s

    # 진행방향
    dm = math.degrees(math.atan2(v_east, v_north))
    return wrap360(dm)


def dv_dm_from_m1_p1(ra_m1, dec_m1, ra_0, deg_0, ra_p1, dec_p1):
    # 2초 동안의 각거리 / 2 = deg/s
    dv = angsep_deg(ra_m1, dec_m1, ra_p1, dec_p1) / 2.0
    dm = bearing_dm_deg(ra_m1, dec_m1, ra_0, deg_0, ra_p1, dec_p1)
    return dv, dm


def xy_gnomonic_deg(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """
    gnomonic(standard) 좌표 (x,y).
    입력/출력 단위: deg (내부는 rad)
    """
    ra  = ra_deg  * D2R
    dec = dec_deg * D2R
    ra0 = ra0_deg * D2R
    de0 = dec0_deg * D2R

    # wrap-aware ΔRA in (-pi, pi]
    dra = math.atan2(math.sin(ra - ra0), math.cos(ra - ra0))

    denom = math.sin(de0)*math.sin(dec) + math.cos(de0)*math.cos(dec)*math.cos(dra)
    eps = 1e-12
    if abs(denom) < eps:
        denom = math.copysign(eps, denom if denom != 0 else 1.0)

    x = (math.cos(dec) * math.sin(dra)) / denom
    y = (math.sin(de0)*math.cos(dec)*math.cos(dra) - math.cos(de0)*math.sin(dec)) / denom
    return x * R2D, y * R2D


def d1_d2_from_tracks_gnomonic_deg(
    tra_m1, tdec_m1, tra_0, tdec_0, tra_p1, tdec_p1,
    pra_m1, pdec_m1, pra_0, pdec_0, pra_p1, pdec_p1,
):
    """
    analyze_and_write_results2의 D1/D2 개념을 그대로 사용.
    - today 3점과 prev 3점을 today (ra0,dec0) 기준 gnomonic xy로 투영
    - 각 트랙의 중심 m, 방향 u를 만들고
    - d = m_prev - m_today 를 today 트랙의 (수직 n, 평행 u)로 분해
    반환: (D1_deg, D2_deg, cos_theta)
    """
    ra0, dec0 = tra_0, tdec_0

    # today track xy
    xt1, yt1 = xy_gnomonic_deg(tra_m1, tdec_m1, tra_p1, tdec_p1)
    xt2, yt2 = xy_gnomonic_deg(tra_p1, tdec_p1, ra0, dec0)

    # prev track xy
    xp1, yp1 = xy_gnomonic_deg(pra_m1, pdec_m1, ra0, dec0)
    xp2, yp2 = xy_gnomonic_deg(pra_p1, pdec_p1, ra0, dec0)

    # 각 트랙의 중심점
    mt_x = 0.5*(xt1 + xt2); mt_y = 0.5*(yt1 + yt2)
    mp_x = 0.5*(xp1 + xp2); mp_y = 0.5*(yp1 + yp2)

    #방향벡터
    vt_x = (xt2 - xt1); vt_y = (yt2 - yt1)
    vp_x = (xp2 - xp1); vp_y = (yp2 - yp1)

    nt = math.hypot(vt_x, vt_y)
    npv = math.hypot(vp_x, vp_y)
    if nt < 1e-12 or npv < 1e-12:
        return float("nan"), float("nan"), float("nan")

    ut_x, ut_y = vt_x/nt, vt_y/nt
    up_x, up_y = vp_x/npv, vp_y/npv

    # today 트랙 법선
    n_x, n_y = -ut_y, ut_x

    # 중심점 차이벡털를 분해
    d_x, d_y = (mp_x - mt_x), (mp_y - mt_y)

    D1 = d_x*n_x + d_y*n_y      # 수직 성분
    D2 = d_x*ut_x + d_y*ut_y    # 평행 성분
    cos_theta = abs(ut_x*up_x + ut_y*up_y) # 방향 유사도 알아볼 때 사용.

    return D1, D2



# key: (norad_id, epoch_time_utc, site_lat, site_lon, site_alt_m)
Key = Tuple[str, str, str, str, str]

def main():
    groups: Dict[Key, Dict[int, Dict[str, str]]] = defaultdict(dict)

    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            norad_id = (row.get("norad_id") or "").strip()
            epoch = (row.get("epoch_time_utc") or "").strip()
            site_lat = (row.get("site_lat_deg") or "").strip()
            site_lon = (row.get("site_lon_deg") or "").strip()
            site_alt = (row.get("site_alt_m") or "").strip()

            try:
                t_offset = int(float(row.get("t_offset_s")))
            except Exception:
                continue
            if t_offset not in (-1, 0, +1):
                continue

            key: Key = (norad_id, epoch, site_lat, site_lon, site_alt)
            groups[key][t_offset] = row


    out_fields = [
        "norad_id",
        "sat_name",
        "epoch_time_utc",
        "site_lat_deg",
        "site_lon_deg",
        "site_alt_m",
        "today_dv",
        "prev_dv",
        "dv_diff_percent",
        "today_dm",
        "prev_dm",
        "dm_diff",
        "d1",  # vertical (Dec-direction)
        "d2",  # horizontal (RA-direction, scaled by cos(Dec0))
        "prev_epoch_time_utc",
        "epoch_diff_sec",
    ]

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()

        for key, samp in groups.items():
            if (-1 not in samp) or (0 not in samp) or (+1 not in samp):
                continue

            row_m1 = samp[-1]
            row_0  = samp[0]
            row_p1 = samp[+1]

            norad_id, epoch, site_lat, site_lon, site_alt = key
            sat_name = (row_0.get("sat_name") or row_m1.get("sat_name") or row_p1.get("sat_name") or "").strip()

            # today points
            tra_m1 = to_float(row_m1.get("today_ra_deg")); tdec_m1 = to_float(row_m1.get("today_dec_deg"))
            tra_0  = to_float(row_0.get("today_ra_deg"));  tdec_0  = to_float(row_0.get("today_dec_deg"))
            tra_p1 = to_float(row_p1.get("today_ra_deg")); tdec_p1 = to_float(row_p1.get("today_dec_deg"))

            # prev points (없으면 스킵)
            pra_m1 = to_float(row_m1.get("prev_ra_deg")); pdec_m1 = to_float(row_m1.get("prev_dec_deg"))
            pra_0  = to_float(row_0.get("prev_ra_deg"));  pdec_0  = to_float(row_0.get("prev_dec_deg"))
            pra_p1 = to_float(row_p1.get("prev_ra_deg")); pdec_p1 = to_float(row_p1.get("prev_dec_deg"))

            if not all(is_finite(v) for v in (pra_m1, pdec_m1, pra_0, pdec_0, pra_p1, pdec_p1)):
                continue
            if not all(is_finite(v) for v in (tra_m1, tdec_m1, tra_0, tdec_0, tra_p1, tdec_p1)):
                continue

            today_dv, today_dm = dv_dm_from_m1_p1(tra_m1, tdec_m1,tra_0, tdec_0, tra_p1, tdec_p1)
            prev_dv,  prev_dm  = dv_dm_from_m1_p1(pra_m1, pdec_m1,pra_0, pdec_0, pra_p1, pdec_p1)

            dv_diff_pct = float("nan")
            if abs(prev_dv) > 0.0:
                dv_diff_pct = ((today_dv - prev_dv) / today_dv) * 100.0

            dm_diff = shortest_angle_diff_deg(today_dm, prev_dm)

            # D1/D2 at epoch (0 sec) only
            d1_deg, d2_deg = d1_d2_from_tracks_gnomonic_deg(
                tra_m1, tdec_m1, tra_0, tdec_0, tra_p1, tdec_p1,
                pra_m1, pdec_m1, pra_0, pdec_0, pra_p1, pdec_p1,
            )

            prev_epoch_iso = (row_0.get("prev_epoch_time_utc") or "").strip()
            epoch_diff_sec = to_float(row_0.get("epoch_diff_sec"))

            w.writerow({
                "norad_id": norad_id,
                "sat_name": sat_name,
                "epoch_time_utc": epoch,
                "site_lat_deg": site_lat,
                "site_lon_deg": site_lon,
                "site_alt_m": site_alt,
                "today_dv": f"{today_dv:.4f}",
                "prev_dv": f"{prev_dv:.4f}",
                "dv_diff_percent": f"{dv_diff_pct:.4f}",
                "today_dm": f"{today_dm:.4f}",
                "prev_dm": f"{prev_dm:.4f}",
                "dm_diff": f"{dm_diff:.4f}",
                "d1": f"{d1_deg:4f}",
                "d2": f"{d2_deg:.4f}",
                "prev_epoch_time_utc": prev_epoch_iso,
                "epoch_diff_sec": f"{epoch_diff_sec:.3f}" if is_finite(epoch_diff_sec) else "nan",
            })

    print(f"wrote: {OUT_CSV}")

if __name__ == "__main__":
    main()
