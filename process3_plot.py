# -*- coding: utf-8 -*-
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS

IN_TRACK_CSV = "20250517_result_raw.csv"            
IN_DVDM_CSV  = "20250517_result.csv"   
D2R = math.pi / 180.0

def to_float(x):
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "unset"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

def is_finite(x):
    return (x is not None) and math.isfinite(x)

def wrap180_deg(d):
    return (d + 180.0) % 360.0 - 180.0

def unwrap_about_center(ra_deg_arr, ra_center):
    ra_deg_arr = np.asarray(ra_deg_arr, float)
    return ra_center + (wrap180_deg(ra_deg_arr - ra_center))

def choose_center(ra_arr, dec_arr):
    ra0 = float(ra_arr[0])
    ra_rel = wrap180_deg(np.asarray(ra_arr) - ra0)
    ra_center = (ra0 + float(np.median(ra_rel))) % 360.0
    dec_center = float(np.median(dec_arr))
    return ra_center, dec_center

def load_tracks_for_norad(path, norad_id, window_s=5):
    """
    returns dict:
      {
        "epoch_time_utc": str,
        "site_lat_deg": float,
        "site_lon_deg": float,
        "site_alt_m": float,
        "t_offset": [..],
        "today_ra": [..], "today_dec":[..],
        "prev_ra":  [..], "prev_dec":[..],
      }
    첫 매칭된 epoch 그룹을 사용. (여러 그룹이면 가장 먼저 읽힌 것)
    """
    target = str(norad_id).strip()

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("norad_id") or "").strip() != target:
                continue
            try:
                dt = int(float(row.get("t_offset_s")))
            except Exception:
                continue
            if dt < -window_s or dt > window_s:
                continue
            rows.append(row)

    if not rows:
        return None

    # epoch/site 키로 그룹핑: 보통 한 norad_id에 한 그룹이지만, 안전하게 처리
    # 여기서는 "첫 그룹"만 사용
    def key_of(row):
        return (
            (row.get("epoch_time_utc") or "").strip(),
            (row.get("site_lat_deg") or "").strip(),
            (row.get("site_lon_deg") or "").strip(),
            (row.get("site_alt_m") or "").strip(),
            (row.get("sat_name") or "").strip(),
            (row.get("epoch_diff_sec") or "").strip(),
        )

    first_key = key_of(rows[0])
    rows = [rw for rw in rows if key_of(rw) == first_key]

    # t_offset 기준 정렬
    rows.sort(key=lambda rw: int(float(rw.get("t_offset_s"))))

    out = {
        "epoch_time_utc": first_key[0],
        "site_lat_deg": to_float(first_key[1]),
        "site_lon_deg": to_float(first_key[2]),
        "site_alt_m": to_float(first_key[3]),
        "sat_name": first_key[4],
        "epoch_diff_sec": round(to_float(first_key[5])/3600, 2),
        "t_offset": [],
        "today_ra": [], "today_dec": [],
        "prev_ra":  [], "prev_dec":  [],
    }

    for rw in rows:
        out["t_offset"].append(int(float(rw.get("t_offset_s"))))
        out["today_ra"].append(to_float(rw.get("today_ra_deg")))
        out["today_dec"].append(to_float(rw.get("today_dec_deg")))
        out["prev_ra"].append(to_float(rw.get("prev_ra_deg")))
        out["prev_dec"].append(to_float(rw.get("prev_dec_deg")))


    # NaN 제거(동시에)
    def _mask(ra, dec):
        ra = np.asarray(ra, float); dec = np.asarray(dec, float)
        m = np.isfinite(ra) & np.isfinite(dec)
        return ra[m], dec[m]

    out["today_ra"], out["today_dec"] = _mask(out["today_ra"], out["today_dec"])
    out["prev_ra"],  out["prev_dec"]  = _mask(out["prev_ra"],  out["prev_dec"])

    return out

def load_dvdm_for_norad(path, norad_id):
    """
    returns dict with dv/dm/d1/d2 etc. if exists.
    같은 norad_id가 여러 줄이면 첫 줄 사용.
    """
    target = str(norad_id).strip()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("norad_id") or "").strip() != target:
                continue
            return {
                "today_dv": to_float(row.get("today_dv")),
                "prev_dv":  to_float(row.get("prev_dv")),
                "today_dm": to_float(row.get("today_dm")),
                "prev_dm":  to_float(row.get("prev_dm")),
                "dm_diff":  to_float(row.get("dm_diff")),
                "d1":       to_float(row.get("d1")),
                "d2":       to_float(row.get("d2")),
                "epoch_time_utc": (row.get("epoch_time_utc") or "").strip(),
                "epoch_diff_sec": to_float(row.get("epoch_diff_sec")),
            }
    return None



from matplotlib.markers import MarkerStyle

def mark_oriented_triangle_world(ax, ra_deg_arr, dec_deg_arr, transform,
                                 size=10, color="k", ahead_px=1, mode="local", z=10):
    """
    ra/dec 배열의 진행방향을 보고, 마지막 점에 삼각형(화살표처럼) 표시.
    - transform: ax.get_transform("world") 넣어야 함
    - mode="local": 마지막 구간 방향(권장)
    """
    if ra_deg_arr is None or dec_deg_arr is None:
        return
    ra_deg_arr = np.asarray(ra_deg_arr, float)
    dec_deg_arr = np.asarray(dec_deg_arr, float)

    m = np.isfinite(ra_deg_arr) & np.isfinite(dec_deg_arr)
    if not np.any(m):
        return
    ra_deg_arr = ra_deg_arr[m]
    dec_deg_arr = dec_deg_arr[m]
    if len(ra_deg_arr) < 2:
        return

    # 마지막 유효한 두 점 선택
    j = len(ra_deg_arr) - 1
    i = j - 1
    while i >= 0:
        if (ra_deg_arr[i] != ra_deg_arr[j]) or (dec_deg_arr[i] != dec_deg_arr[j]):
            break
        i -= 1
    if i < 0:
        return

    # world -> display 로 변환해서 각도 계산
    P = transform.transform(np.array([[ra_deg_arr[i], dec_deg_arr[i]],
                                      [ra_deg_arr[j], dec_deg_arr[j]]], float))
    p0, p1 = P[0], P[1]
    v = p1 - p0
    nrm = float(np.hypot(v[0], v[1]))
    if nrm <= 0:
        return
    u = v / nrm
    angle_deg = float(np.degrees(np.arctan2(u[1], u[0])))

    # 약간 앞으로 뺀 지점에 마커 찍기
    p_end = p1 + u * float(ahead_px)
    ra_end, dec_end = transform.inverted().transform(p_end)

    mk = MarkerStyle((3, 0, angle_deg + 30))  # 삼각형 방향 맞추기
    ax.plot([ra_end], [dec_end],
            marker=mk, markersize=float(size),
            markerfacecolor=color, markeredgewidth=0.0, markeredgecolor="none",
            linestyle="None", transform=transform, zorder=z, alpha=0.8,
            clip_on=True)



def create_wcs_axes(ra0_deg, dec0_deg, fov_ra_deg=10.0, fov_dec_deg=10.0, npix=1200, projection="TAN"):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [npix/2 + 0.5, npix/2 + 0.5]
    wcs.wcs.crval = [ra0_deg, dec0_deg]
    wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    wcs.wcs.cdelt = np.array([fov_ra_deg/npix, fov_dec_deg/npix], dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_xlim(0.5, npix + 0.5)
    ax.set_ylim(0.5, npix + 0.5)

    ax.coords.grid(True, color="0.5", linestyle="dotted", alpha=0.8)

    # 축 라벨 크기
    ax.coords[0].set_axislabel("RA (J2000)", fontsize=14)
    ax.coords[1].set_axislabel("Dec (J2000)", fontsize=14)

    # 포맷만 지정 (fontsize 넣지 말기)
    ax.coords[0].set_major_formatter("hh:mm")
    ax.coords[1].set_major_formatter("dd")

    # 눈금 글씨 크기
    ax.coords[0].set_ticklabel(size=14)
    ax.coords[1].set_ticklabel(size=14)

    return fig, ax, wcs

def main():
    norad_id = input("NORAD ID 입력: ").strip()

    tr = load_tracks_for_norad(IN_TRACK_CSV, norad_id, window_s=5)
    if tr is None:
        print(f"[ERROR] {IN_TRACK_CSV}에서 norad_id={norad_id} 데이터를 못 찾음")
        return

    dv = load_dvdm_for_norad(IN_DVDM_CSV, norad_id)
    if dv is None:
        print(f"[WARN] {IN_DVDM_CSV}에서 norad_id={norad_id} dv/dm/d1/d2를 못 찾음")

    # center: today 기준(없으면 prev 기준)
    if len(tr["today_ra"]) >= 1:
        ra_center, dec_center = choose_center(tr["today_ra"], tr["today_dec"])
    else:
        ra_center, dec_center = choose_center(tr["prev_ra"], tr["prev_dec"])

    # unwrap(경계 넘어가는 경우 대비)
    today_ra_u = unwrap_about_center(tr["today_ra"], ra_center) if len(tr["today_ra"]) else np.array([])
    prev_ra_u  = unwrap_about_center(tr["prev_ra"],  ra_center) if len(tr["prev_ra"])  else np.array([])

    # FOV: ra±5, dec±5
    fov_ra_deg  = 30.0
    fov_dec_deg = 30.0

    fig, ax, wcs = create_wcs_axes(ra_center, dec_center, fov_ra_deg=fov_ra_deg, fov_dec_deg=fov_dec_deg,
                                   npix=1200, projection="TAN")
    world = ax.get_transform("world")

    # plot: today / prev
    if len(today_ra_u):
        ax.plot(today_ra_u, tr["today_dec"], "-", linewidth=2.5, alpha=0.8,
                transform=world, zorder=5, label="today", color="red")
        '''ax.plot(today_ra_u, tr["today_dec"], "o", markersize=4, alpha=0.8,
                transform=world, zorder=6)'''
        mark_oriented_triangle_world(ax, today_ra_u, tr["today_dec"], world,
                                     size=12, color="red", ahead_px=2, z=20)


    if len(prev_ra_u):
        ax.plot(prev_ra_u, tr["prev_dec"], "-", linewidth=2.5, alpha=0.8,
                transform=world, zorder=4, label="prev", color="royalblue")
        '''ax.plot(prev_ra_u, tr["prev_dec"], "o", markersize=4, alpha=0.8,
                transform=world, zorder=4)'''
        mark_oriented_triangle_world(ax, prev_ra_u, tr["prev_dec"], world,
                                     size=12, color="royalblue", ahead_px=2, z=19)

    # title
    epoch = tr['epoch_time_utc'].replace("T", " ").replace("Z", "")
    object_name = tr['sat_name']
    diff_time = tr['epoch_diff_sec']
    ax.set_title(f"TLE date 2025-05-17 \n NORAD ID {norad_id} | Epoch {epoch}", fontsize=18)

    # bottom text (dv/dm/d1/d2)
    lines = []
    if object_name:
        ax.text(
            0.02, 0.19,  # ← 살짝 위로
            object_name,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=18,  # ← 크게
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            zorder=101
        )
    if dv is not None:
        if is_finite(dv.get("today_dv")) and is_finite(dv.get("prev_dv")):
            dvper = (dv["today_dv"] - dv["prev_dv"]) / dv["today_dv"] * 100
            lines.append(f"AV   {dv['today_dv']:.4f} / {dv['prev_dv']:.4f} (DV {dv['today_dv']-dv['prev_dv']:.4f}°/s, {dvper:+.1f}%)")
        elif is_finite(dv.get("today_dv")):
            lines.append(f"AV   {dv['today_dv']:.4f}  (deg/s)")


        if is_finite(dv.get("today_dm")) and is_finite(dv.get("prev_dm")):
            if is_finite(dv.get("dm_diff")):
                lines.append(f"MA   {dv['today_dm']:.1f} / {dv['prev_dm']:.1f}  (DA {dv['dm_diff']:+.1f}°)")
            else:
                lines.append(f"MA   {dv['today_dm']:.1f} / {dv['prev_dm']:.1f}")
        elif is_finite(dv.get("today_dm")):
            lines.append(f"Da   {dv['today_dm']:.1f}")

        if is_finite(dv.get("d1")):
            lines.append(f"D1   {dv['d1']:.2f}°")
        if is_finite(dv.get("d2")):
            lines.append(f"D2   {dv['d2']:+.2f}°")

        if diff_time:
            lines.append(f"DT   {diff_time} h")

    if lines:
        ax.text(
            0.02, 0.02, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            zorder=100
        )

    # legend
    ax.legend(loc="upper right", framealpha=0.0, edgecolor="none", fontsize=14)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
