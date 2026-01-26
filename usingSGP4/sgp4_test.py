# sgp4_test.py  (formerly sgdp4_port.py)
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Public constants (other modules import these)
# -----------------------------------------------------------------------------
XMNPDA = 1440.0
SGDP4_DEEP_NORM = 4
SGDP4_DEEP_RESN = 5
SGDP4_DEEP_SYNC = 6

SGDP4_ERROR     = -1
SGDP4_NOT_INIT  = 0
SGDP4_ZERO_ECC  = 1
SGDP4_NEAR_SIMP = 2
SGDP4_NEAR_NORM = 3
SGDP4_DEEP_NORM = 4
SGDP4_DEEP_RESN = 5
SGDP4_DEEP_SYNC = 6

# -----------------------------------------------------------------------------
# Public types (other modules import OrbitT)
# -----------------------------------------------------------------------------
@dataclass
class OrbitT:
    satno: int
    ep_year: int
    ep_day: float
    ecc: float
    rev: float
    eqinc: float
    ascn: float
    argp: float
    mnan: float
    bstar: float

@dataclass
class XYZ:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Kep:
    smjaxs: float = 0.0
    radius: float = 0.0
    theta: float = 0.0
    eqinc: float = 0.0
    ascn: float = 0.0
    argp: float = 0.0
    ecc: float = 0.0
    rdotk: float = 0.0
    rfdotk: float = 0.0

# -----------------------------------------------------------------------------
# Numeric constants (match your sgdp4.c port)
# -----------------------------------------------------------------------------
PI = math.pi
TWOPI = 2.0 * math.pi
TOTHRD = 2.0 / 3.0

# IMPORTANT: JD1900 must match your original C reference.
# If your C used a different JD1900, change it here.
JD1900 = 2415020.5

Q0 = 120.0
S0 = 78.0
XJ2 = 1.082616e-3
XJ3 = -2.53881e-6
XJ4 = -1.65597e-6
XKMPER = 6378.135
AE = 1.0

XKE = 7.43669161331734132e-2
CK2 = 0.5 * XJ2 * AE * AE
CK4 = -0.375 * XJ4 * AE**4
QOMS2T = 1.880279159015270643865e-9
KS = AE * (1.0 + S0 / XKMPER)
a3ovk2 = (-XJ3 / CK2) * (AE**3)

ECC_ZERO = 0.0
ECC_ALL = 1.0e-4
ECC_EPS = 1.0e-6
ECC_LIMIT_LOW = -1.0e-3
ECC_LIMIT_HIGH = 1.0 - ECC_EPS
EPS_COSIO = 1.5e-12
NR_EPS = 1.0e-12

# deep.c 외부 플래그 (현재 deep 미구현이면 그대로 두고 deep 모드에서 예외)
Set_LS_zero = 0

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _sincos(x: float) -> Tuple[float, float]:
    return math.sin(x), math.cos(x)

def _pow4(x: float) -> float:
    x2 = x * x
    return x2 * x2

def _cube(x: float) -> float:
    return x * x * x

def _sign(a: float, b: float) -> float:
    return abs(a) if b >= 0.0 else -abs(a)

def _mod2pi(x: float) -> float:
    x = math.fmod(x, TWOPI)
    if x < 0.0:
        x += TWOPI
    return x

# -----------------------------------------------------------------------------
# Model (NO GLOBALS): every coefficient/state lives in this instance
# -----------------------------------------------------------------------------
class Sgdp4Model:
    def __init__(self) -> None:
        # public
        self.SGDP4_jd0: float = 0.0

        # mode/state
        self.imode: int = SGDP4_NOT_INIT
        self.Isat: int = 0

        # copied elements (sgdp4.c globals -> self.*)
        self.xno = 0.0
        self.xmo = 0.0
        self.eo = 0.0
        self.xincl = 0.0
        self.omegao = 0.0
        self.xnodeo = 0.0
        self.bstar = 0.0

        # precomputed globals
        self.sinIO = 0.0
        self.cosIO = 0.0
        self.sinXMO = 0.0
        self.cosXMO = 0.0

        self.c1 = self.c2 = self.c3 = self.c4 = self.c5 = 0.0
        self.d2 = self.d3 = self.d4 = 0.0
        self.omgcof = self.xmcof = self.xlcof = self.aycof = 0.0
        self.t2cof = self.t3cof = self.t4cof = self.t5cof = 0.0
        self.xnodcf = self.delmo = 0.0

        self.x7thm1 = self.x3thm1 = self.x1mth2 = 0.0

        self.aodp = 0.0
        self.eta = 0.0
        self.omgdot = 0.0
        self.xnodot = 0.0
        self.xnodp = 0.0
        self.xmdot = 0.0

        self.perigee = 0.0
        self.apogee = 0.0
        self.period = 0.0

    # ----------------------------
    # Deep-space stubs (keep same behavior)
    # ----------------------------
    def _dpinit(self, *args, **kwargs) -> int:
        raise NotImplementedError("Deep-space dpinit not implemented (port deep.c)")

    def _dpsec(self, *args, **kwargs) -> None:
        raise NotImplementedError("Deep-space dpsec not implemented (port deep.c)")

    def _dpper(self, *args, **kwargs) -> None:
        raise NotImplementedError("Deep-space dpper not implemented (port deep.c)")

    # ----------------------------
    # Public API
    # ----------------------------
    def init(self, orb: OrbitT) -> int:
        self.imode = self._init_sgdp4(orb)
        return self.imode

    def satpos_xyz(self, jd: float, want_vel: bool = False):
        # keep your wrapper signature: (pos, vel, mode)
        mode, pos, vel = self._satpos_xyz(jd, want_vel=want_vel)

        print('pos', pos)
        return pos, vel, mode

    # ----------------------------
    # Core ports (instance methods)
    # ----------------------------
    def _init_sgdp4(self, orb: OrbitT) -> int:
        # Convert year to Gregorian with century
        iyear = int(orb.ep_year)
        if iyear < 1957:
            iyear += (2000 if iyear < 57 else 1900)

        if iyear < 1901 or iyear > 2099:
            raise RuntimeError(f"init_sgdp4: Satellite ep_year error {iyear}")

        self.Isat = int(orb.satno)

        # days since 1900 reference (match your C)
        iday = ((iyear - 1901) * 1461) // 4 + 364 + 1
        self.SGDP4_jd0 = JD1900 + iday + (float(orb.ep_day) - 1.0)

        epoch = (iyear - 1900) * 1.0e3 + float(orb.ep_day)  # YYDDD.DDDD style

        # Copy elements
        self.eo = float(orb.ecc)
        self.xno = float(orb.rev) * TWOPI / XMNPDA  # rad/min
        self.xincl = float(orb.eqinc)
        self.xnodeo = float(orb.ascn)
        self.omegao = float(orb.argp)
        self.xmo = float(orb.mnan)
        self.bstar = float(orb.bstar)

        # checks
        if self.eo < 0.0 or self.eo > ECC_LIMIT_HIGH:
            raise RuntimeError(f"init_sgdp4: Eccentricity out of range {self.Isat} ({self.eo})")

        if self.xno < 0.035 * TWOPI / XMNPDA or self.xno > 18.0 * TWOPI / XMNPDA:
            raise RuntimeError(f"init_sgdp4: Mean motion out of range {self.Isat} ({self.xno})")

        if self.xincl < 0.0 or self.xincl > PI:
            raise RuntimeError(f"init_sgdp4: Inclination out of range {self.Isat} ({self.xincl})")

        # mode preselect
        if self.eo < ECC_ZERO:
            imode = SGDP4_ZERO_ECC
        else:
            imode = SGDP4_NOT_INIT

        # Recover original mean motion (xnodp) and semimajor axis (aodp)
        self.sinIO, self.cosIO = _sincos(self.xincl)

        theta2 = self.cosIO * self.cosIO
        theta4 = theta2 * theta2
        self.x3thm1 = 3.0 * theta2 - 1.0
        self.x1mth2 = 1.0 - theta2
        self.x7thm1 = 7.0 * theta2 - 1.0

        a1 = (XKE / self.xno) ** TOTHRD
        betao2 = 1.0 - self.eo * self.eo
        betao = math.sqrt(betao2)

        temp0 = (1.5 * CK2) * self.x3thm1 / (betao * betao2)
        del1 = temp0 / (a1 * a1)
        a0 = a1 * (1.0 - del1 * (1.0 / 3.0 + del1 * (1.0 + del1 * 134.0 / 81.0)))
        del0 = temp0 / (a0 * a0)

        self.xnodp = self.xno / (1.0 + del0)
        self.aodp = a0 / (1.0 - del0)

        self.perigee = (self.aodp * (1.0 - self.eo) - AE) * XKMPER
        self.apogee  = (self.aodp * (1.0 + self.eo) - AE) * XKMPER
        self.period  = (TWOPI * 1440.0 / XMNPDA) / self.xnodp

        if imode == SGDP4_ZERO_ECC:
            self.imode = imode
            return imode

        # model selection
        if self.period >= 225.0 and Set_LS_zero < 2:
            imode = SGDP4_DEEP_NORM
        elif self.perigee < 220.0:
            imode = SGDP4_NEAR_SIMP
        else:
            imode = SGDP4_NEAR_NORM

        # For perigee below 156 km alter S and QOMS2T
        if self.perigee < 156.0:
            s4 = self.perigee - 78.0
            if s4 < 20.0:
                s4 = 20.0
            qoms24 = _pow4((120.0 - s4) * (AE / XKMPER))
            s4 = s4 / XKMPER + AE
        else:
            s4 = KS
            qoms24 = QOMS2T

        pinvsq = 1.0 / (self.aodp * self.aodp * betao2 * betao2)
        tsi = 1.0 / (self.aodp - s4)
        self.eta = self.aodp * self.eo * tsi
        etasq = self.eta * self.eta
        eeta = self.eo * self.eta
        psisq = abs(1.0 - etasq)
        coef = qoms24 * _pow4(tsi)
        coef1 = coef / (psisq ** 3.5)

        self.c2 = coef1 * self.xnodp * (
            self.aodp * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq)) +
            (0.75 * CK2) * tsi / psisq * self.x3thm1 *
            (8.0 + 3.0 * etasq * (8.0 + etasq))
        )
        self.c1 = self.bstar * self.c2

        self.c4 = 2.0 * self.xnodp * coef1 * self.aodp * betao2 * (
            self.eta * (2.0 + 0.5 * etasq) +
            self.eo * (0.5 + 2.0 * etasq) -
            (2.0 * CK2) * tsi / (self.aodp * psisq) * (
                -3.0 * self.x3thm1 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta)) +
                0.75 * self.x1mth2 * (2.0 * etasq - eeta * (1.0 + etasq)) * math.cos(2.0 * self.omegao)
            )
        )

        self.c5 = 0.0
        self.c3 = 0.0
        self.omgcof = 0.0

        if imode == SGDP4_NEAR_NORM:
            self.c5 = 2.0 * coef1 * self.aodp * betao2 * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
            if self.eo > ECC_ALL:
                self.c3 = coef * tsi * a3ovk2 * self.xnodp * AE * self.sinIO / self.eo
            self.omgcof = self.bstar * self.c3 * math.cos(self.omegao)

        temp1 = (3.0 * CK2) * pinvsq * self.xnodp
        temp2 = temp1 * CK2 * pinvsq
        temp3 = (1.25 * CK4) * pinvsq * pinvsq * self.xnodp

        self.xmdot = self.xnodp + (0.5 * temp1 * betao * self.x3thm1 + 0.0625 * temp2 * betao *
                                   (13.0 - 78.0 * theta2 + 137.0 * theta4))

        x1m5th = 1.0 - 5.0 * theta2
        self.omgdot = (-0.5 * temp1 * x1m5th + 0.0625 * temp2 *
                       (7.0 - 114.0 * theta2 + 395.0 * theta4) +
                       temp3 * (3.0 - 36.0 * theta2 + 49.0 * theta4))

        xhdot1 = -temp1 * self.cosIO
        self.xnodot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * theta2) +
                                2.0 * temp3 * (3.0 - 7.0 * theta2)) * self.cosIO

        self.xmcof = 0.0
        if self.eo > ECC_ALL:
            self.xmcof = (-TOTHRD * AE) * coef * self.bstar / eeta

        self.xnodcf = 3.5 * betao2 * xhdot1 * self.c1
        self.t2cof = 1.5 * self.c1

        temp0 = 1.0 + self.cosIO
        if abs(temp0) < EPS_COSIO:
            temp0 = _sign(EPS_COSIO, temp0)

        self.xlcof = 0.125 * a3ovk2 * self.sinIO * (3.0 + 5.0 * self.cosIO) / temp0
        self.aycof = 0.25 * a3ovk2 * self.sinIO

        self.sinXMO, self.cosXMO = _sincos(self.xmo)
        self.delmo = _cube(1.0 + self.eta * self.cosXMO)

        if imode == SGDP4_NEAR_NORM:
            c1sq = self.c1 * self.c1
            self.d2 = 4.0 * self.aodp * tsi * c1sq
            temp0 = self.d2 * tsi * self.c1 / 3.0
            self.d3 = (17.0 * self.aodp + s4) * temp0
            self.d4 = 0.5 * temp0 * self.aodp * tsi * (221.0 * self.aodp + 31.0 * s4) * self.c1
            self.t3cof = self.d2 + 2.0 * c1sq
            self.t4cof = 0.25 * (3.0 * self.d3 + self.c1 * (12.0 * self.d2 + 10.0 * c1sq))
            self.t5cof = 0.2 * (3.0 * self.d4 + 12.0 * self.c1 * self.d3 + 6.0 * self.d2 * self.d2 +
                                15.0 * c1sq * (2.0 * self.d2 + c1sq))
        elif imode == SGDP4_DEEP_NORM:
            imode = self._dpinit(epoch, self.omegao, self.xnodeo, self.xmo, self.eo, self.xincl,
                                 self.aodp, self.xmdot, self.omgdot, self.xnodot, self.xnodp)

        self.imode = imode
        return imode

    def _sgdp4(self, tsince_min: float, withvel: bool) -> Tuple[int, Kep]:
        K = Kep()

        em = self.eo
        xinc = self.xincl

        xmp = self.xmo + self.xmdot * tsince_min
        xnode = self.xnodeo + tsince_min * (self.xnodot + tsince_min * self.xnodcf)
        omega = self.omegao + self.omgdot * tsince_min

        if self.imode == SGDP4_ZERO_ECC:
            K.smjaxs = self.aodp * XKMPER / AE
            K.radius = K.smjaxs
            K.theta = math.fmod(PI + self.xnodp * tsince_min, TWOPI) - PI
            K.eqinc = self.xincl
            K.ascn = self.xnodeo
            K.argp = 0.0
            K.ecc = 0.0
            if withvel:
                K.rfdotk = self.aodp * self.xnodp * (XKMPER / AE * XMNPDA / 86400.0)
            return self.imode, K

        if self.imode == SGDP4_NEAR_SIMP:
            tempa = 1.0 - tsince_min * self.c1
            tempe = self.bstar * tsince_min * self.c4
            templ = tsince_min * tsince_min * self.t2cof
            a = self.aodp * tempa * tempa
            e = em - tempe
            xl = xmp + omega + xnode + self.xnodp * templ

        elif self.imode == SGDP4_NEAR_NORM:
            delm = self.xmcof * (_cube(1.0 + self.eta * math.cos(xmp)) - self.delmo)
            temp0 = tsince_min * self.omgcof + delm
            xmp += temp0
            omega -= temp0
            tempa = 1.0 - (tsince_min * (self.c1 + tsince_min * (self.d2 + tsince_min * (self.d3 + tsince_min * self.d4))))
            tempe = self.bstar * (self.c4 * tsince_min + self.c5 * (math.sin(xmp) - self.sinXMO))
            templ = tsince_min * tsince_min * (self.t2cof + tsince_min * (self.t3cof + tsince_min * (self.t4cof + tsince_min * self.t5cof)))
            a = self.aodp * tempa * tempa
            e = em - tempe
            xl = xmp + omega + xnode + self.xnodp * templ

        elif self.imode in (SGDP4_DEEP_NORM, SGDP4_DEEP_RESN, SGDP4_DEEP_SYNC):
            raise NotImplementedError("Deep-space propagation not implemented (dpsec/dpper needed)")

        else:
            return SGDP4_ERROR, K

        if a < 1.0:
            return SGDP4_ERROR, K
        if e < ECC_LIMIT_LOW:
            return SGDP4_ERROR, K
        if e < ECC_EPS:
            e = ECC_EPS
        elif e > ECC_LIMIT_HIGH:
            e = ECC_LIMIT_HIGH

        beta2 = 1.0 - e * e

        sinOMG, cosOMG = _sincos(omega)

        temp0 = 1.0 / (a * beta2)
        axn = e * cosOMG
        ayn = e * sinOMG + temp0 * self.aycof
        xlt = xl + temp0 * self.xlcof * axn

        elsq = axn * axn + ayn * ayn
        if elsq >= 1.0:
            return SGDP4_ERROR, K

        K.ecc = math.sqrt(elsq)

        capu = _mod2pi(xlt - xnode)
        epw = capu
        maxnr = K.ecc

        esinE = 0.0
        ecosE = 0.0
        sinEPW = 0.0
        cosEPW = 1.0

        for ii in range(10):
            sinEPW, cosEPW = _sincos(epw)

            ecosE = axn * cosEPW + ayn * sinEPW
            esinE = axn * sinEPW - ayn * cosEPW

            f = capu - epw + esinE
            if abs(f) < NR_EPS:
                break

            df = 1.0 - ecosE
            nr = f / df

            if ii == 0 and abs(nr) > 1.25 * maxnr:
                nr = _sign(maxnr, nr)
            else:
                nr = f / (df + 0.5 * esinE * nr)

            epw += nr

        temp0 = 1.0 - elsq
        betal = math.sqrt(temp0)
        pl = a * temp0
        r = a * (1.0 - ecosE)
        invR = 1.0 / r
        temp2 = a * invR
        temp3 = 1.0 / (1.0 + betal)
        cosu = temp2 * (cosEPW - axn + ayn * esinE * temp3)
        sinu = temp2 * (sinEPW - ayn - axn * esinE * temp3)
        u = math.atan2(sinu, cosu)

        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0
        temp0 = 1.0 / pl
        temp1 = CK2 * temp0
        temp2p = temp1 * temp0

        rk = r * (1.0 - 1.5 * temp2p * betal * self.x3thm1) + 0.5 * temp1 * self.x1mth2 * cos2u
        uk = u - 0.25 * temp2p * self.x7thm1 * sin2u
        xnodek = xnode + 1.5 * temp2p * self.cosIO * sin2u
        xinck = xinc + 1.5 * temp2p * self.cosIO * self.sinIO * cos2u

        if rk < 1.0:
            return SGDP4_ERROR, K

        K.radius = rk * XKMPER / AE
        K.theta = uk
        K.eqinc = xinck
        K.ascn = xnodek
        K.argp = omega
        K.smjaxs = a * XKMPER / AE

        if withvel:
            temp0s = math.sqrt(a)
            temp2v = XKE / (a * temp0s)
            K.rdotk = (XKE * temp0s * esinE * invR - temp2v * temp1 * self.x1mth2 * sin2u) * (XKMPER / AE * XMNPDA / 86400.0)
            K.rfdotk = (XKE * math.sqrt(pl) * invR + temp2v * temp1 * (self.x1mth2 * cos2u + 1.5 * self.x3thm1)) * (XKMPER / AE * XMNPDA / 86400.0)
        else:
            K.rdotk = 0.0
            K.rfdotk = 0.0

        return self.imode, K

    def _kep2xyz(self, K: Kep, want_vel: bool = True) -> Tuple[XYZ, Optional[XYZ]]:
        sinT, cosT = _sincos(K.theta)
        sinI, cosI = _sincos(K.eqinc)
        sinS, cosS = _sincos(K.ascn)

        xmx = -sinS * cosI
        xmy =  cosS * cosI

        ux =  xmx * sinT + cosS * cosT
        uy =  xmy * sinT + sinS * cosT
        uz =  sinI * sinT

        pos = XYZ(K.radius * ux, K.radius * uy, K.radius * uz)
        print('keppos', pos)

        if not want_vel:
            return pos, None

        vx =  xmx * cosT - cosS * sinT
        vy =  xmy * cosT - sinS * sinT
        vz =  sinI * cosT

        vel = XYZ(
            K.rdotk * ux + K.rfdotk * vx,
            K.rdotk * uy + K.rfdotk * vy,
            K.rdotk * uz + K.rfdotk * vz,
        )
        return pos, vel

    def _satpos_xyz(self, jd: float, want_vel: bool = True) -> Tuple[int, XYZ, Optional[XYZ]]:
        tsince = (jd - self.SGDP4_jd0) * XMNPDA
        mode, K = self._sgdp4(tsince, withvel=want_vel)
        pos, vel = self._kep2xyz(K, want_vel=want_vel)

        return mode, pos, vel
