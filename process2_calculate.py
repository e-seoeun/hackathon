#!/usr/bin/env python3
import csv
import math
from collections import defaultdict
from typing import Dict, Tuple
import sys

IN_CSV = sys.argv[1]
OUT_CSV = sys.argv[2]

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# key: (norad_id, epoch_time_utc, site_lat, site_lon)
Key = Tuple[str, str, str, str]


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


def xy_gnomonic_deg(ra_deg: float, dec_deg: float, ra0_deg: float, dec0_deg: float) -> Tuple[float, float]:
    """
    gnomonic(standard) 좌표 (x,y)
    입력/출력 단위: deg (내부는 rad)
    """
    ra = ra_deg * D2R
    dec = dec_deg * D2R
    ra0 = ra0_deg * D2R
    de0 = dec0_deg * D2R

    # wrap-aware ΔRA in (-pi, pi]
    dra = math.atan2(math.sin(ra - ra0), math.cos(ra - ra0))

    denom = math.sin(de0) * math.sin(dec) + math.cos(de0) * math.cos(dec) * math.cos(dra)
    eps = 1e-12
    if abs(denom) < eps:
        denom = math.copysign(eps, denom if denom != 0 else 1.0)

    x = (math.cos(dec) * math.sin(dra)) / denom
    y = (math.sin(de0) * math.cos(dec) * math.cos(dra) - math.cos(de0) * math.sin(dec)) / denom
    return x * R2D, y * R2D


def track_dir_angle_gnomonic_deg(
    ra_m1: float, dec_m1: float,
    ra_p1: float, dec_p1: float,
    ra0: float, dec0: float,
) -> Tuple[float, float, float]:
    """
    같은 투영 기준점(ra0,dec0)에서
    트랙 방향벡터 v=(dx,dy)와 방향각 theta 반환.
    theta는 +x축 기준(북점 무관), [0,360)
    """
    x1, y1 = xy_gnomonic_deg(ra_m1, dec_m1, ra0, dec0)
    x2, y2 = xy_gnomonic_deg(ra_p1, dec_p1, ra0, dec0)
    dx, dy = (x2 - x1), (y2 - y1) #벡터 계산하기
    if math.hypot(dx, dy) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    theta = wrap360(math.degrees(math.atan2(dy, dx)))
    return theta, dx, dy


def vector_angle_diff_signed_deg(ax: float, ay: float, bx: float, by: float) -> float:
    """
    두 벡터 a,b 사이 signed 각도차 (deg), (-180,180]
    북점/좌표계 기준 없이, 같은 평면에서의 상대각.
    """
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")

    ax, ay = ax / na, ay / na
    bx, by = bx / nb, by / nb

    dot = max(-1.0, min(1.0, ax * bx + ay * by))
    cross = ax * by - ay * bx
    return math.degrees(math.atan2(cross, dot))


def d1_d2_from_tracks_gnomonic_deg(
    tra_m1: float, tdec_m1: float, tra_0: float, tdec_0: float, tra_p1: float, tdec_p1: float,
    pra_m1: float, pdec_m1: float, pra_0: float, pdec_0: float, pra_p1: float, pdec_p1: float,
) -> Tuple[float, float]:
    """
    - today/prev를 today (ra0,dec0) 기준 gnomonic xy로 투영
    - 각 트랙 중심점 차이를 today 트랙의 (수직 n, 평행 u)로 분해
    반환: (D1_deg, D2_deg)
    """
    ra0, dec0 = tra_0, tdec_0

    # today track xy
    xt1, yt1 = xy_gnomonic_deg(tra_m1, tdec_m1, ra0, dec0)
    xt2, yt2 = xy_gnomonic_deg(tra_p1, tdec_p1, ra0, dec0)

    # prev track xy
    xp1, yp1 = xy_gnomonic_deg(pra_m1, pdec_m1, ra0, dec0)
    xp2, yp2 = xy_gnomonic_deg(pra_p1, pdec_p1, ra0, dec0)

    # 중심점
    mt_x = 0.5 * (xt1 + xt2)
    mt_y = 0.5 * (yt1 + yt2)
    mp_x = 0.5 * (xp1 + xp2)
    mp_y = 0.5 * (yp1 + yp2)

    # today 방향벡터
    vt_x = (xt2 - xt1)
    vt_y = (yt2 - yt1)
    nt = math.hypot(vt_x, vt_y)
    if nt < 1e-12:
        return float("nan"), float("nan")

    ut_x, ut_y = vt_x / nt, vt_y / nt
    n_x, n_y = -ut_y, ut_x  # today 법선

    d_x, d_y = (mp_x - mt_x), (mp_y - mt_y)

    D1 = d_x * n_x + d_y * n_y              # 수직
    D2 = -(d_x * ut_x + d_y * ut_y)         # 평행(부호는 기존 정의 유지)

    return D1, D2


def main() -> None:
    groups: Dict[Key, Dict[int, Dict[str, str]]] = defaultdict(dict)

    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            norad_id = (row.get("norad_id") or "").strip()
            epoch = (row.get("epoch_time_utc") or "").strip()
            site_lat = (row.get("site_lat_deg") or "").strip()
            site_lon = (row.get("site_lon_deg") or "").strip()

            try:
                t_offset = int(float(row.get("t_offset_s")))
            except Exception:
                continue
            if t_offset not in (-1, 0, +1):
                continue

            key: Key = (norad_id, epoch, site_lat, site_lon)
            groups[key][t_offset] = row

    out_fields = [
        "norad_id",
        "epoch_time_utc",
        "site_lat_deg",
        "site_lon_deg",
        "today_AV",
        "prev_AV",
        "DV_percent",
        "today_MA",   # gnomonic xy에서의 방향각(+x 기준)
        "prev_MA",
        "DA",         # gnomonic xy에서의 두 트랙 방향 차(signed)
        "D1",
        "D2",
        "prev_epoch_time_utc",
        "epoch_diff_sec",
        # 추가: TLE 파라미터
        "Bstar",
        "Inc_deg",
        "RAAN_deg",
        "AP_deg",
        "MA_deg",
        "MM_rev/day",
        "MeanAlt_km",
        "object_name",
    ]

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()

        for key, samp in groups.items():
            if (-1 not in samp) or (0 not in samp) or (+1 not in samp):
                continue

            row_m1 = samp[-1]
            row_0 = samp[0]
            row_p1 = samp[+1]

            norad_id, epoch, site_lat, site_lon = key
            object_name = (row_0.get("object_name") or row_m1.get("object_name") or row_p1.get("object_name") or "").strip()

            # today points
            tra_m1 = to_float(row_m1.get("today_ra_deg"));  tdec_m1 = to_float(row_m1.get("today_dec_deg"))
            tra_0  = to_float(row_0.get("today_ra_deg"));   tdec_0  = to_float(row_0.get("today_dec_deg"))
            tra_p1 = to_float(row_p1.get("today_ra_deg"));  tdec_p1 = to_float(row_p1.get("today_dec_deg"))

            # prev points
            pra_m1 = to_float(row_m1.get("prev_ra_deg"));   pdec_m1 = to_float(row_m1.get("prev_dec_deg"))
            pra_0  = to_float(row_0.get("prev_ra_deg"));    pdec_0  = to_float(row_0.get("prev_dec_deg"))
            pra_p1 = to_float(row_p1.get("prev_ra_deg"));   pdec_p1 = to_float(row_p1.get("prev_dec_deg"))

            if not all(is_finite(v) for v in (tra_m1, tdec_m1, tra_0, tdec_0, tra_p1, tdec_p1)):
                continue
            if not all(is_finite(v) for v in (pra_m1, pdec_m1, pra_0, pdec_0, pra_p1, pdec_p1)):
                continue

            # AV (deg/s): 2초 각거리 / 2
            today_av = angsep_deg(tra_m1, tdec_m1, tra_p1, tdec_p1) / 2.0
            prev_av  = angsep_deg(pra_m1, pdec_m1, pra_p1, pdec_p1) / 2.0

            dv_percent = float("nan")
            if abs(today_av) > 0.0:
                dv_percent = ((today_av - prev_av) / today_av) * 100.0

            # MA/DA: gnomonic xy에서 상대적인 방향각/각도차
            # 투영 기준점은 today의 (tra_0, tdec_0)로 통일
            today_ma, tvx, tvy = track_dir_angle_gnomonic_deg(
                tra_m1, tdec_m1, tra_p1, tdec_p1, tra_0, tdec_0
            )
            prev_ma, pvx, pvy = track_dir_angle_gnomonic_deg(
                pra_m1, pdec_m1, pra_p1, pdec_p1, tra_0, tdec_0
            )

            da = vector_angle_diff_signed_deg(tvx, tvy, pvx, pvy)
            #기준점 없이 바ㅑㅇ향 일치도만 상대적으로 구하자

            # D1/D2
            d1_deg, d2_deg = d1_d2_from_tracks_gnomonic_deg(
                tra_m1, tdec_m1, tra_0, tdec_0, tra_p1, tdec_p1,
                pra_m1, pdec_m1, pra_0, pdec_0, pra_p1, pdec_p1,
            )

            prev_epoch_iso = (row_0.get("prev_epoch_time_utc") or "").strip()
            epoch_diff_sec = to_float(row_0.get("epoch_diff_sec"))

            # TLE 정보 (row_0 기준)
            bstar = (row_0.get("Bstar") or "").strip()
            inc = (row_0.get("Inc_deg") or "").strip()
            raan = (row_0.get("RAAN_deg") or "").strip()
            ap = (row_0.get("AP_deg") or "").strip()
            ma = (row_0.get("MA_deg") or "").strip()
            mm = (row_0.get("MM_rev/day") or "").strip()
            malt = (row_0.get("MeanAlt_km") or "").strip()

            w.writerow({
                "norad_id": norad_id,
                "epoch_time_utc": epoch,
                "site_lat_deg": site_lat,
                "site_lon_deg": site_lon,
                "today_AV": f"{today_av:.6f}",
                "prev_AV": f"{prev_av:.6f}",
                "DV_percent": f"{dv_percent:.6f}" if is_finite(dv_percent) else "nan",
                "today_MA": f"{today_ma:.6f}" if is_finite(today_ma) else "nan",
                "prev_MA": f"{prev_ma:.6f}" if is_finite(prev_ma) else "nan",
                "DA": f"{da:.6f}" if is_finite(da) else "nan",
                "D1": f"{d1_deg:.6f}" if is_finite(d1_deg) else "nan",
                "D2": f"{d2_deg:.6f}" if is_finite(d2_deg) else "nan",
                "prev_epoch_time_utc": prev_epoch_iso,
                "epoch_diff_sec": f"{epoch_diff_sec:.3f}" if is_finite(epoch_diff_sec) else "nan",
                "Bstar": bstar,
                "Inc_deg": inc,
                "RAAN_deg": raan,
                "AP_deg": ap,
                "MA_deg": ma,
                "MM_rev/day": mm,
                "MeanAlt_km": malt,
                "object_name": object_name,
            })

    print(f"wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()