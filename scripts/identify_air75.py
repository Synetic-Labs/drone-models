#!/usr/bin/env python3
r"""Rigorous system identification of the crazyflow ``air75_wisp`` params.

Fits the full first-principles parameter set (everything except mass, which is
measured) from real BetaFPV Air75 flight logs: full-rate (2 kHz) Betaflight
blackbox + Vicon ground truth.

Pipeline
--------
  1. load     decode every ``*/blackbox/*.bbl`` (orangebox) at loop rate;
              align Vicon pose from ``flight_synced.csv`` onto the blackbox
              clock. Cache decoded arrays to ``.sysid_cache/``.
  2. thrust   rpm2thrust from body-z specific force  (F = m*a_z, mass anchored)
  3. inertia  J (diagonal) + rpm2torque from the rotational Newton-Euler
              equation, with the arm length L anchored to the frame geometry
              (L = wheelbase/(2*sqrt2)); pure I/O data only fixes torque/J
              ratios, so L is the anchor that makes J absolute.
  4. rotor    static throttle->rpm map (rpm_idle/rpm_max -> thrust_min/max) and
              the first-order rotor_dyn_coef from the rpm step response.
  5. drag     body-frame linear drag from the translational residual (Vicon).
  6. refine   optional JAX windowed-rollout joint refinement of all params
              against the measured gyro / rpm / accelerometer trajectories.
  7. validate held-out flight: open-loop rollout RMSE (old vs staged vs refined)
  8. write    patch the [air75_wisp] block of this repo's
              drone_models/data/params.toml (and ensure the matching empty
              section in first_principles/params.toml that load_params needs).

Frames
------
Betaflight logs body rates / specific force in its board frame (x fwd / roll,
z up: accSmooth_z reads +1 g at rest). We keep that frame throughout; the per-
axis sign needed to make J positive-definite is resolved from the data and
reported. Motors are reindexed into crazyflow order (cf M1..M4 = bf [1,0,2,3])
so the standard X-quad mixing matrix applies.

Usage
-----
    .venv/bin/python scripts/identify_air75.py \
        --data-dir "/home/luke/Downloads/flight_data/FLIGHT DATA" [--write]
    # options: --no-refine  --plots  --val-session NAME  --mass-kg 0.0344
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, field

os.environ.setdefault("SCIPY_ARRAY_API", "1")  # before drone_models/scipy import

import numpy as np
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# constants / geometry
# ---------------------------------------------------------------------------
G = 9.80665
DEFAULT_MASS_KG = 0.0344        # measured all-up Air75 Wisp (with 1S LiPo)
ACC_1G = 2048.0                 # betaflight accSmooth raw units per g
MOTOR_POLES = 12               # -> rpm = eRPM_field * 100 / (poles/2)
ERPM_TO_RPM = 100.0 / (MOTOR_POLES / 2)
MOTOR_OUT_MIN, MOTOR_OUT_MAX = 158.0, 2047.0   # blackbox header motorOutput
BF_TO_CF = (1, 0, 2, 3)        # crazyflow motor k = betaflight motor BF_TO_CF[k]

WHEELBASE_M = 0.075            # Air75 frame spec (motor-to-motor diagonal)
L_ARM = WHEELBASE_M / (2.0 * math.sqrt(2.0))   # X-quad roll/pitch moment arm

# standard X-quad mixing matrix in crazyflow motor order (M1 FR, M2 RR, M3 RL,
# M4 FL); rows are [roll, pitch, yaw] sign patterns.
MIX = np.array([
    [-1.0, -1.0,  1.0,  1.0],   # roll
    [-1.0,  1.0,  1.0, -1.0],   # pitch
    [-1.0,  1.0, -1.0,  1.0],   # yaw
])

PROP_INERTIA = 2.5e-8          # kg m^2, light 40mm prop (gyroscopic term only)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".sysid_cache")


# ---------------------------------------------------------------------------
# blackbox decode + session loading
# ---------------------------------------------------------------------------
@dataclass
class Session:
    name: str
    t: np.ndarray            # s, blackbox clock, zeroed
    rpm: np.ndarray          # [N,4] crazyflow motor order, RPM
    motor_norm: np.ndarray   # [N,4] crazyflow order, normalised [0,1]
    gyro: np.ndarray         # [N,3] rad/s, raw (gyroUnfilt), betaflight body axes
    gyro_filt: np.ndarray    # [N,3] rad/s, FC-filtered (gyroADC)
    acc: np.ndarray          # [N,3] m/s^2 specific force, betaflight body axes
    vbat: np.ndarray         # [N] V
    throttle: np.ndarray     # [N] rcCommand[3]
    armed: np.ndarray        # [N] bool
    gyro_sat: np.ndarray     # [N] bool, gyro railed at +-2000 deg/s
    # vicon (NaN where unavailable), interpolated onto t:
    vic_pos: np.ndarray      # [N,3] m  (x,y,z up)
    vic_vel: np.ndarray      # [N,3] m/s
    vic_quat: np.ndarray     # [N,4] xyzw, body->world
    has_vicon: bool
    air: np.ndarray = field(default=None)  # [N] bool, valid airborne mask
    rpm_lag: int = 0         # samples rpm was shifted to align with acc/gyro


def _decode_bbl(bbl_path: str) -> dict:
    """Decode one .bbl to raw per-field arrays (cached as npz)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = bbl_path.replace(os.sep, "_").replace(" ", "_").strip("_")
    cache = os.path.join(CACHE_DIR, key + ".npz")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(bbl_path):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from orangebox import Parser
    p = Parser.load(bbl_path)
    names = list(p.field_names)
    ix = {n: names.index(n) for n in names}
    # tolerate a truncated/corrupt tail (some logs end mid-frame): keep every
    # frame decoded before the parser raises. The violent crash flights are the
    # most valuable and the most likely to be cut short.
    rows = []
    nf = len(names)
    try:
        for f in p.frames():
            if len(f.data) == nf:
                rows.append(f.data)
    except Exception as e:
        print(f"  [decode] {os.path.basename(bbl_path)}: kept {len(rows)} frames "
              f"before {type(e).__name__}")
    # some frames carry empty/None cells in non-core fields (mixed frame types);
    # coerce them to NaN so the whole log still loads
    A = np.array(rows, dtype=object)
    A[np.isin(A, np.array(["", "None", None], dtype=object))] = np.nan
    A = A.astype(float)
    out = {}
    for n in names:
        out[n.replace("[", "_").replace("]", "")] = A[:, ix[n]]
    np.savez_compressed(cache, **out)
    return out


def _vicon_aligner(session_dir: str):
    """Return (interp_fn, ok). interp_fn(loopIter) -> (pos[3], vel[3], quat[4]).

    flight_synced.csv carries both Abs_time (Vicon clock) and bb_loopIteration,
    giving a linear loopIteration->Abs_time map. We then interpolate the Vicon
    columns (sampled on Abs_time) against the blackbox loopIteration directly,
    which sidesteps any absolute-clock offset.
    """
    csv_path = os.path.join(session_dir, "flight_synced.csv")
    if not os.path.exists(csv_path):
        return None
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    if not rows or "bb_loopIteration" not in rows[0] or "b1_x" not in rows[0]:
        return None

    def col(k):
        out = np.array([r.get(k, "") for r in rows], dtype=object)
        return np.array([float(v) if v not in ("", "None", None) else np.nan
                         for v in out])

    li = col("bb_loopIteration")
    pos = np.c_[col("b1_x"), col("b1_y"), col("b1_z")]
    vel = np.c_[col("b1_vx"), col("b1_vy"), col("b1_vz")]
    quat = np.c_[col("b1_qx"), col("b1_qy"), col("b1_qz"), col("b1_qw")]
    good = np.isfinite(li) & np.isfinite(pos).all(1) & np.isfinite(quat).all(1)
    li, pos, vel, quat = li[good], pos[good], vel[good], quat[good]
    if len(li) < 10:
        return None
    order = np.argsort(li)
    li, pos, vel, quat = li[order], pos[order], vel[order], quat[order]

    def interp(loop_iter):
        inside = (loop_iter >= li[0]) & (loop_iter <= li[-1])
        def I(y):
            out = np.full((len(loop_iter), y.shape[1]), np.nan)
            for j in range(y.shape[1]):
                out[:, j] = np.interp(loop_iter, li, y[:, j])
            out[~inside] = np.nan
            return out
        return I(pos), I(vel), I(quat)

    return interp


def load_session(session_dir: str, mass_kg: float) -> Session | None:
    bbls = [p for p in glob.glob(os.path.join(session_dir, "**", "*.bbl"),
                                 recursive=True) if "btfl_all" not in p]
    if not bbls:
        return None
    bbl = sorted(bbls, key=os.path.getsize)[-1]   # largest = main flight
    d = _decode_bbl(bbl)
    if "time" not in d or "eRPM_0" not in d:
        return None
    t = (d["time"] - d["time"][0]) / 1e6
    erpm = np.c_[[d[f"eRPM_{i}"] for i in range(4)]].T
    motor = np.c_[[d[f"motor_{i}"] for i in range(4)]].T
    gyro = np.c_[[d[f"gyroUnfilt_{i}"] for i in range(3)]].T * (math.pi / 180.0)
    gyro_filt = np.c_[[d[f"gyroADC_{i}"] for i in range(3)]].T * (math.pi / 180.0)
    acc = np.c_[[d[f"accSmooth_{i}"] for i in range(3)]].T / ACC_1G * G
    vbat = d.get("vbatLatest", np.zeros_like(t))
    throttle = d.get("rcCommand_3", np.zeros_like(t))
    flags = d.get("flightModeFlags", np.ones_like(t))

    perm = list(BF_TO_CF)
    rpm = np.clip(erpm[:, perm] * ERPM_TO_RPM, 0.0, None)
    motor_norm = np.clip((motor[:, perm] - MOTOR_OUT_MIN)
                         / (MOTOR_OUT_MAX - MOTOR_OUT_MIN), 0.0, 1.0)
    # remove eRPM-telemetry latency: align measured rpm onto the loop clock
    rpm, motor_norm, rpm_lag = _align_rpm_lag(rpm, motor_norm, acc, flags > 0)
    armed = flags > 0
    # gyro rails at +-2000 deg/s; flag saturated samples so the fits skip them
    s_gyro = np.abs(np.c_[[d[f"gyroUnfilt_{i}"] for i in range(3)]].T)
    gyro_sat = (s_gyro >= 1990.0).any(1)

    # vicon alignment via loopIteration (never let a bad CSV drop the flight:
    # the rotational fits don't need Vicon, and the violent flights matter most)
    try:
        aligner = _vicon_aligner(session_dir)
    except Exception:
        aligner = None
    if aligner is not None and "loopIteration" in d:
        vp, vv, vq = aligner(d["loopIteration"])
        has_vicon = np.isfinite(vp).all(1).sum() > 100
    else:
        vp = vv = np.full((len(t), 3), np.nan)
        vq = np.full((len(t), 4), np.nan)
        has_vicon = False

    s = Session(name=os.path.basename(session_dir.rstrip("/")), t=t, rpm=rpm,
                motor_norm=motor_norm, gyro=gyro, gyro_filt=gyro_filt,
                acc=acc, vbat=vbat,
                throttle=throttle, armed=armed, gyro_sat=gyro_sat,
                vic_pos=vp, vic_vel=vv, vic_quat=vq, has_vicon=has_vicon,
                rpm_lag=rpm_lag)
    _mark_airborne(s)
    return s


def _mark_airborne(s: Session) -> None:
    """Airborne = armed, all rotors spinning, gyro not railed, and (if Vicon)
    clearly off the floor. No specific-force gate: that would throw away the
    inverted / high-rate samples in the violent flights, which carry the most
    inertia information."""
    finite = np.isfinite(s.rpm).all(1) & np.isfinite(s.gyro).all(1) & np.isfinite(s.acc).all(1)
    air = s.armed & finite & (s.rpm > 1000).all(1) & (~s.gyro_sat)
    if s.has_vicon:
        alt = s.vic_pos[:, 2]
        air = air & np.isfinite(alt) & (alt > 0.25)
    s.air = air


def discover_sessions(data_dir: str, mass_kg: float):
    dirs = set()
    for p in glob.glob(os.path.join(data_dir, "**", "*.bbl"), recursive=True):
        if "btfl_all" in p:
            continue
        d = os.path.dirname(p)
        if os.path.basename(d) == "blackbox":
            d = os.path.dirname(d)
        dirs.add(d)
    out = []
    for d in sorted(dirs):
        try:
            s = load_session(d, mass_kg)
        except Exception as e:
            print(f"  [skip] {os.path.basename(d)}: {e}")
            continue
        if s is None:
            continue
        n_air = int(s.air.sum())
        if n_air < 200:
            print(f"  [skip] {s.name}: only {n_air} airborne samples")
            continue
        out.append(s)
        print(f"  [use ] {s.name:18s} {n_air:6d} airborne  "
              f"vicon={'yes' if s.has_vicon else 'no '}  "
              f"{s.t[-1]:.0f}s @ {1/np.median(np.diff(s.t)):.0f}Hz  "
              f"rpm_lag={s.rpm_lag:+d}")
    return out


# ---------------------------------------------------------------------------
# numerical helpers
# ---------------------------------------------------------------------------
def _deriv(y: np.ndarray, t: np.ndarray, win: int = 21, poly: int = 3):
    """Savitzky-Golay derivative on a (nearly) uniform grid."""
    dt = float(np.median(np.diff(t)))
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < poly + 2:
        return np.gradient(y, t)
    if win % 2 == 0:
        win += 1
    return savgol_filter(y, win, poly, deriv=1, delta=dt)


def _smooth(y: np.ndarray, win: int = 21, poly: int = 3):
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < poly + 2:
        return y
    if win % 2 == 0:
        win += 1
    return savgol_filter(y, win, poly)


def _binmed(x, y, nb=14, mincount=20):
    """Robust binned-median samples of y(x) for slope fits dominated by noise."""
    edges = np.quantile(x, np.linspace(0, 1, nb))
    b = np.digitize(x, edges)
    xs, ys = [], []
    for k in range(1, nb):
        sel = b == k
        if sel.sum() >= mincount:
            xs.append(np.median(x[sel])); ys.append(np.median(y[sel]))
    return np.array(xs), np.array(ys)


def _air_runs(s: "Session", min_len: int = 40):
    """Contiguous airborne index runs. Smoothing/differentiating/window-
    integrating must stay inside a run: the `air` mask is punched full of gaps
    mid-flight (rotor dips below the spin gate, gyro rails during flips), and a
    Savitzky-Golay window or a reshape that straddles such a gap fabricates a
    derivative/integral across a time discontinuity -- precisely in the violent
    segments that carry the inertia information."""
    m = np.where(s.air)[0]
    if len(m) == 0:
        return []
    splits = np.where(np.diff(m) > 1)[0] + 1
    return [r for r in np.split(m, splits) if len(r) >= min_len]


def _sma0(x, y):
    """Through-origin standardised-major-axis (errors-in-variables) slope of
    y = s*x. Both x and y are noisy here (gyro endpoints AND rpm-derived torque),
    so an OLS slope is attenuated whichever way it is run; the SMA slope
    s = sign(<x,y>) * sqrt(<y,y>/<x,x>) is symmetric in x/y and dilution-free.
    Reduces to OLS only in the no-noise limit."""
    sxx = float(np.sum(x * x)); syy = float(np.sum(y * y))
    sxy = float(np.sum(x * y))
    if sxx <= 0 or syy <= 0:
        return 0.0
    return float(np.sign(sxy) * math.sqrt(syy / sxx))


def _align_rpm_lag(rpm, motor_norm, acc, armed, max_lag=15):
    """Estimate + remove the eRPM-telemetry latency relative to the loop-rate
    acc/gyro. Bidirectional-DSHOT eRPM is reported with a transport/filter delay,
    so measured rpm is misaligned with the inertial channels; that biases every
    transient-driven fit (torque/J, rotor dynamics). We find the integer sample
    shift maximising the correlation of the collective-thrust transient
    d/dt(sum rpm^2) against d/dt(acc_z), then roll rpm/motor onto the acc clock.
    Returns (rpm, motor_norm, lag)."""
    coarse = (armed & np.isfinite(rpm).all(1) & np.isfinite(acc[:, 2])
              & (rpm > 1000).all(1))
    if coarse.sum() < 3000:          # too little data to trust a lag estimate
        return rpm, motor_norm, 0
    dS = np.diff((rpm ** 2).sum(1)); da = np.diff(acc[:, 2])
    base = coarse[1:] & coarse[:-1]
    corrs = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b, mm = dS[:len(dS) - lag], da[lag:], base[:len(base) - lag] & base[lag:]
        else:
            a, b, mm = dS[-lag:], da[:len(da) + lag], base[-lag:] & base[:len(base) + lag]
        if mm.sum() < 1000:
            continue
        corrs[lag] = float(np.corrcoef(a[mm], b[mm])[0, 1])
    if not corrs:
        return rpm, motor_norm, 0
    best_lag = max(corrs, key=corrs.get)
    # distrust a peak pinned to the search boundary (correlation never turned
    # over -> not a real latency peak) or a weak/ambiguous peak
    if abs(best_lag) == max_lag or corrs[best_lag] < corrs.get(0, 0.0) + 0.01:
        best_lag = 0
    if best_lag != 0:
        rpm = np.roll(rpm, best_lag, axis=0)
        motor_norm = np.roll(motor_norm, best_lag, axis=0)
        if best_lag > 0:
            rpm[:best_lag] = np.nan; motor_norm[:best_lag] = np.nan
        else:
            rpm[best_lag:] = np.nan; motor_norm[best_lag:] = np.nan
    return rpm, motor_norm, best_lag


# ---------------------------------------------------------------------------
# stage 2: rpm2thrust
# ---------------------------------------------------------------------------
def fit_thrust(pool, mass_kg: float) -> dict:
    """F_tot = m * a_z(body) = a0_tot + a1*sum(rpm) + a2*sum(rpm^2).

    Reported as the per-motor curve rpm2thrust = [a0, a1, a2] with
    F_motor = a0 + a1*rpm + a2*rpm^2 (so a0_tot = 4*a0)."""
    S1, S2, F = [], [], []
    for s in pool:
        for run in _air_runs(s, min_len=8):
            # smooth body-z specific force within a contiguous run (prop
            # vibration is not thrust; smoothing across air-mask gaps would mix
            # non-adjacent samples)
            az = _smooth(s.acc[run, 2], win=51)
            S1.append(s.rpm[run].sum(1))
            S2.append((s.rpm[run] ** 2).sum(1))
            F.append(mass_kg * az)
    S1 = np.concatenate(S1); S2 = np.concatenate(S2); F = np.concatenate(F)

    # pure quadratic through origin (physical prior)
    a2 = float(np.linalg.lstsq(S2[:, None], F, rcond=None)[0][0])
    # uncentered R^2 (appropriate for a through-origin model; centered R^2 is
    # meaningless near hover where F is ~constant): 1 - SS_res / sum(F^2)
    r2_q = float(1 - np.sum((F - a2 * S2) ** 2) / np.sum(F ** 2))
    # full quadratic for diagnostics
    M = np.c_[np.ones_like(S1), S1, S2]
    full = np.linalg.lstsq(M, F, rcond=None)[0]
    resid = float(np.std(F - a2 * S2))
    return dict(rpm2thrust=[0.0, 0.0, a2], a2=a2, r2=r2_q, resid_N=resid,
                full_quad=full.tolist(), n=len(F),
                hover_total_N=float(np.median(F)))


# ---------------------------------------------------------------------------
# stage 3: inertia J (L anchored) + rpm2torque
# ---------------------------------------------------------------------------
def fit_inertia_torque(pool, c_T: float, L: float, prop_inertia: float,
                       thrust2torque: float | None = None,
                       J_override: tuple | None = None) -> dict:
    r"""Rotational Newton-Euler in integral (window) form, pooled across flights.

    Window-integrate the full Euler equation per contiguous airborne run:

        Jxx*dwx = integral(L*(F . mix_roll))  - (Jzz-Jyy)*integral(wy*wz)
        Jyy*dwy = integral(L*(F . mix_pitch)) - (Jxx-Jzz)*integral(wz*wx)
        Jzz*dwz = kappa*integral(F_yaw) + prop_inertia*d(L_prop) - (Jyy-Jxx)*integral(wx*wy)

    Improvements over the previous median-of-ratios fit:
      * contiguous-run windows (no integration across air-mask gaps);
      * the Coriolis/gyroscopic terms are kept and solved by fixed-point
        iteration (they are largest in exactly the high-excitation windows we
        rely on, so dropping them biased the diagonal J);
      * windows are selected by INPUT excitation (|integral(tau)|), not by the
        noisy rate response |dw| -- selecting on the response/denominator pulled
        J low;
      * the slope is the errors-in-variables SMA estimate ([_sma0]), dilution-
        free in both the gyro endpoints and the rpm-derived torque.

    Yaw is fundamentally one equation in two unknowns (only kappa/Jzz is
    observable from the yaw rate). Two modes:
      * default: anchor Jzz ~= Jxx+Jyy (planar perpendicular-axis) and read kappa
        off the yaw response. kappa then inherits the planar-anchor error.
      * thrust2torque given: fix kappa = thrust2torque*c_T from the prop's
        aerodynamic Q/T and instead *identify* Jzz from the yaw response, which
        breaks the coupling and gives a data-driven, anchor-free Jzz.
    The reported kappa/Jzz ratio is the quantity the yaw data actually constrains.
    """
    rpm_to_rad = 2 * math.pi / 60.0

    # ---- gyro sign resolution (per contiguous run) --------------------------
    flips = np.ones(3)
    for ax in (0, 1, 2):
        al, tq = [], []
        for s in pool:
            for run in _air_runs(s, 60):
                al.append(_deriv(s.gyro[run, ax], s.t[run], win=101))
                rp2 = s.rpm[run] ** 2
                tq.append((L * c_T * (rp2 @ MIX[ax])) if ax < 2 else (rp2 @ MIX[2]))
        if np.dot(np.concatenate(al), np.concatenate(tq)) < 0:
            flips[ax] = -1.0

    # ---- per-window quantities (endpoint diffs + integrals), pooled ---------
    win_s = 0.06
    dWx, dWy, dWz = [], [], []
    Itx, Ity, Iyaw = [], [], []        # integral of roll/pitch torque, yaw rpm^2
    Iyz, Izx, Ixy = [], [], []         # integral of rate products (Coriolis)
    dLprop = []                        # endpoint change of prop angular momentum
    for s in pool:
        for run in _air_runs(s, 60):
            t = s.t[run]; dt = float(np.median(np.diff(t)))
            W = max(int(win_s / dt), 2); n = len(t) // W
            if n < 1:
                continue
            cut = slice(0, n * W)
            wx = (flips[0] * _smooth(s.gyro[run, 0], 15))[cut].reshape(n, W)
            wy = (flips[1] * _smooth(s.gyro[run, 1], 15))[cut].reshape(n, W)
            wz = (flips[2] * _smooth(s.gyro[run, 2], 15))[cut].reshape(n, W)
            rp2 = s.rpm[run] ** 2
            # roll/pitch torque = L * thrust_differential = L * c_T * (rpm^2 . mix);
            # yaw regressor stays rpm^2 . mix (kappa carries the coefficient)
            tx = (L * c_T * (rp2 @ MIX[0]))[cut].reshape(n, W)
            ty = (L * c_T * (rp2 @ MIX[1]))[cut].reshape(n, W)
            sy = (rp2 @ MIX[2])[cut].reshape(n, W)
            lp = ((s.rpm[run] * rpm_to_rad) @ MIX[2])[cut].reshape(n, W)  # prop L_z
            dWx.append(wx[:, -1] - wx[:, 0]); dWy.append(wy[:, -1] - wy[:, 0])
            dWz.append(wz[:, -1] - wz[:, 0])
            Itx.append(tx.sum(1) * dt); Ity.append(ty.sum(1) * dt)
            Iyaw.append(sy.sum(1) * dt)
            Iyz.append((wy * wz).sum(1) * dt); Izx.append((wz * wx).sum(1) * dt)
            Ixy.append((wx * wy).sum(1) * dt)
            dLprop.append(lp[:, -1] - lp[:, 0])
    dWx = np.concatenate(dWx); dWy = np.concatenate(dWy); dWz = np.concatenate(dWz)
    Itx = np.concatenate(Itx); Ity = np.concatenate(Ity); Iyaw = np.concatenate(Iyaw)
    Iyz = np.concatenate(Iyz); Izx = np.concatenate(Izx); Ixy = np.concatenate(Ixy)
    dLprop = np.concatenate(dLprop)

    def sel(inp, pct=60):
        """keep windows with strong torque INPUT (not strong rate response)."""
        return np.abs(inp) >= np.percentile(np.abs(inp), pct)

    def _ols0(x, y):
        """through-origin OLS slope y=s*x (attenuated low by noise in x)."""
        sxx = float(np.sum(x * x))
        return float(np.sum(x * y) / sxx) if sxx else 0.0

    # ---- diagonal J: Coriolis-corrected SMA, fixed-point ---------------------
    # SMA (errors-in-variables) is the symmetric estimator -- the geometric mean
    # of the y-on-x and x-on-y OLS slopes -- so it is the principled point
    # estimate when both axes are noisy. We also report the OLS slope: at the low
    # torque->rate correlation this data gives, OLS (attenuated) and SMA bracket
    # the true J, and the width of that bracket (~1/corr) is the honest J
    # uncertainty. A tight bracket means J is well identified; a wide one (as here)
    # means the flight data alone cannot pin J -- anchor it with a bench/CAD value.
    sx, sy_, sz = sel(Itx), sel(Ity), sel(Iyaw)
    Jxx = abs(_sma0(dWx[sx], Itx[sx]))
    Jyy = abs(_sma0(dWy[sy_], Ity[sy_]))
    Jzz = Jxx + Jyy
    for _ in range(4):                                   # solve the coupling
        Jxx = abs(_sma0(dWx[sx], Itx[sx] - (Jzz - Jyy) * Iyz[sx]))
        Jyy = abs(_sma0(dWy[sy_], Ity[sy_] + (Jzz - Jxx) * Izx[sy_]))
        if thrust2torque is None:
            Jzz = Jxx + Jyy                              # planar anchor
    # attenuated lower bound (OLS) for the uncertainty bracket (always from data,
    # even when J is overridden -- the bracket is what feeds domain randomisation)
    Jxx_ols = abs(_ols0(dWx[sx], Itx[sx] - (Jzz - Jyy) * Iyz[sx]))
    Jyy_ols = abs(_ols0(dWy[sy_], Ity[sy_] + (Jzz - Jxx) * Izx[sy_]))
    Jxx_sma, Jyy_sma, Jzz_sma = Jxx, Jyy, Jzz       # data point estimate (high)

    # ---- yaw: kappa <-> Jzz (only the ratio is observable) ------------------
    prop = prop_inertia * dLprop
    if J_override is not None:
        # hardcoded (most-plausible / bench) J wins; kappa from prop Q/T if given,
        # else read off yaw at the hardcoded Jzz
        Jxx, Jyy, Jzz = (float(v) for v in J_override)
        if thrust2torque is not None:
            kappa = float(thrust2torque * c_T)
        else:
            kappa = abs(_sma0(Iyaw[sz], (Jzz * dWz - prop - (Jxx - Jyy) * Ixy)[sz]))
    elif thrust2torque is not None:
        # fix kappa from prop Q/T, identify Jzz from the yaw response
        kappa = float(thrust2torque * c_T)
        rhs = kappa * Iyaw + prop + (Jxx - Jyy) * Ixy
        Jzz = abs(_sma0(dWz[sz], rhs[sz]))
        Jzz = Jzz or (Jxx + Jyy)
    else:
        # Jzz anchored above; read kappa off the (Coriolis/prop-corrected) yaw
        Yz = Jzz * dWz - prop - (Jxx - Jyy) * Ixy
        kappa = abs(_sma0(Iyaw[sz], Yz[sz]))
    ratio = kappa / Jzz if Jzz else float("nan")

    def _corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
    corr_x = _corr(dWx[sx], Itx[sx] - (Jzz - Jyy) * Iyz[sx])
    corr_y = _corr(dWy[sy_], Ity[sy_] + (Jzz - Jxx) * Izx[sy_])
    corr_z = _corr(Iyaw[sz], (Jzz * dWz - prop - (Jxx - Jyy) * Ixy)[sz])

    if J_override is not None:
        jzz_mode = "hardcoded"
    elif thrust2torque is not None:
        jzz_mode = "yaw"
    else:
        jzz_mode = "planar"
    return dict(J=[[Jxx, 0, 0], [0, Jyy, 0], [0, 0, Jzz]],
                Jxx=Jxx, Jyy=Jyy, Jzz=Jzz, kappa=kappa,
                Jxx_ols=Jxx_ols, Jyy_ols=Jyy_ols,    # attenuated lower bracket
                Jxx_sma=Jxx_sma, Jyy_sma=Jyy_sma, Jzz_sma=Jzz_sma,  # high bracket
                kappa_over_Jzz=ratio, thrust2torque=kappa / c_T, jzz_mode=jzz_mode,
                rpm2torque=[0.0, 0.0, kappa], gyro_flips=flips.tolist(),
                corr_roll=corr_x, corr_pitch=corr_y, corr_yaw=corr_z, n=len(dWz))


# ---------------------------------------------------------------------------
# stage 4: static throttle->rpm map + rotor dynamics
# ---------------------------------------------------------------------------
def fit_rotor(pool, c_T: float) -> dict:
    mn, rp = [], []
    for s in pool:
        for run in _air_runs(s, min_len=8):
            mn.append(s.motor_norm[run].reshape(-1))
            rp.append(s.rpm[run].reshape(-1))
    mn = np.concatenate(mn); rp = np.concatenate(rp)
    bx, by = _binmed(mn, rp, nb=16)
    (rpm_idle, slope), *_ = np.linalg.lstsq(np.c_[np.ones_like(bx), bx], by,
                                            rcond=None)
    rpm_max = rpm_idle + slope
    static = (rpm_idle, rpm_max)

    # rotor dynamics: d(rpm)/dt = k_up*(cmd-rpm) [cmd>rpm] else k_dn*(cmd-rpm)
    # (derivative taken per contiguous run so the window never crosses a gap)
    du, eu, dd, ed = [], [], [], []   # up: (cmd-rpm), rpm_dot ; dn: ...
    for s in pool:
        for run in _air_runs(s, min_len=40):
            cmd = rpm_idle + slope * s.motor_norm[run]
            for i in range(4):
                rpm = s.rpm[run][:, i]
                rdot = _deriv(rpm, s.t[run], win=15)
                err = cmd[:, i] - rpm
                big = np.abs(err) > 1500          # transients only
                up = big & (err > 0); dn = big & (err < 0)
                du.append(err[up]); eu.append(rdot[up])
                dd.append(err[dn]); ed.append(rdot[dn])
    du = np.concatenate(du); eu = np.concatenate(eu)
    dd = np.concatenate(dd); ed = np.concatenate(ed)
    k_up = float(np.linalg.lstsq(du[:, None], eu, rcond=None)[0][0]) if len(du) else 0.0
    k_dn = float(np.linalg.lstsq(dd[:, None], ed, rcond=None)[0][0]) if len(dd) else 0.0
    k_up = max(k_up, 0.0); k_dn = max(k_dn, 0.0)
    k_simple = 0.5 * (k_up + k_dn) if (k_up and k_dn) else max(k_up, k_dn)

    thrust_min = c_T * rpm_idle ** 2
    thrust_max = c_T * rpm_max ** 2
    return dict(rpm_idle=float(rpm_idle), rpm_max=float(rpm_max),
                thrust_min=float(thrust_min), thrust_max=float(thrust_max),
                rotor_dyn_coef=[k_up, 0.0, k_dn, 0.0],
                rotor_dyn_coef_simple=float(k_simple),
                thrust_dyn_coef=float(k_simple),
                tau_up_ms=1000.0 / k_up if k_up else float("nan"),
                tau_dn_ms=1000.0 / k_dn if k_dn else float("nan"))


# ---------------------------------------------------------------------------
# stage 5: body-frame linear drag (needs Vicon)
# ---------------------------------------------------------------------------
def _quat_to_R(q):
    """xyzw body->world rotation matrices, batched [N,4] -> [N,3,3]."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def fit_drag(pool, c_T: float, mass_kg: float) -> dict:
    """Body-frame linear drag from the measured specific force.

    The accelerometer already reports body specific force = (thrust + drag)/m,
    so the body drag force is just  drag_body = m*acc_body - thrust_body  with
    thrust along +z only. The lateral components fall out directly as
    drag_xy = m*acc_xy -- no differentiation of Vicon velocity (which would be a
    second derivative of position and is the dominant noise source in the old
    m*a_world - thrust - g formulation). Vicon is used only for the body-velocity
    regressor. We restrict to low-angular-rate samples so the accelerometer
    lever-arm term (alpha x r + w x (w x r), it is offset from the CoM) does not
    masquerade as drag. Vertical drag stays weakly observable (thrust is colinear
    with body up and dwarfs it), so we lean on the lateral axes.
    """
    # NOTE: these are near-hover indoor flights (median lateral speed ~0.08 m/s),
    # so linear drag is only weakly excited. We keep samples that are (a) clearly
    # translating (|v_axis| > V_GATE) and (b) not in a violent maneuver (the
    # accelerometer is offset from the CoM, so high angular rate injects an
    # alpha x r / w x (w x r) term that masquerades as drag), then enforce the
    # physical sign (drag opposes velocity -> coefficient <= 0). Axes that fail
    # those gates fall back to a sign-correct borrowed magnitude. Drag matters
    # little for a hover policy; the DR range is what the policy should rely on.
    V_GATE = 0.5          # m/s, minimum body speed for an observable drag sample
    vb_all, fb_all = [], []
    for s in pool:
        if not s.has_vicon:
            continue
        for run in _air_runs(s, min_len=80):
            ok = (np.isfinite(s.vic_vel[run]).all(1)
                  & np.isfinite(s.vic_quat[run]).all(1)
                  & (np.linalg.norm(s.gyro[run], axis=1) < 3.0))  # exclude flips
            idx = run[ok]
            if len(idx) < 60:
                continue
            R = _quat_to_R(s.vic_quat[idx])
            vw = np.c_[[_smooth(s.vic_vel[idx][:, j]) for j in range(3)]].T
            vb = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), vw)  # body vel
            # body specific force -> body drag force (remove thrust from z only)
            Ftot = (c_T * s.rpm[idx] ** 2).sum(1)
            db = mass_kg * s.acc[idx]
            db[:, 2] -= Ftot
            vb_all.append(vb); fb_all.append(db)
    if not vb_all:
        return dict(drag_matrix=None, note="no vicon sessions")
    vb = np.vstack(vb_all); fb = np.vstack(fb_all)
    d = np.full(3, np.nan)
    n_axis = [0, 0, 0]
    for j in range(3):
        g = np.abs(vb[:, j]) > V_GATE        # observable (translating) samples
        n_axis[j] = int(g.sum())
        if g.sum() < 300:
            continue
        bx, by = _binmed(vb[g, j], fb[g, j], nb=10)
        if len(bx) >= 3:
            slope = np.linalg.lstsq(np.c_[np.ones_like(bx), bx], by, rcond=None)[0][1]
            if slope < 0:                    # keep only physical (drag) slopes
                d[j] = slope
    # lateral: average the observable, physical x/y slopes; else borrow
    lat = [v for v in (d[0], d[1]) if np.isfinite(v)]
    BORROW_XY = -0.010                       # modest physical default (whoop)
    dxy = float(np.mean(lat)) if lat else BORROW_XY
    dz = float(d[2]) if np.isfinite(d[2]) else dxy * (0.024961 / 0.014719)
    observed = "".join(ax for ax, v in zip("xyz", d) if np.isfinite(v)) or "none"
    return dict(drag_matrix=[[round(dxy, 6), 0.0, 0.0],
                             [0.0, round(dxy, 6), 0.0],
                             [0.0, 0.0, round(dz, 6)]],
                drag_xy=dxy, drag_z=dz, n=len(vb),
                observed_axes=observed, n_axis=n_axis,
                borrowed=(not lat))


# ---------------------------------------------------------------------------
# stage 6: differentiable trajectory-matching refinement (JAX)
# ---------------------------------------------------------------------------
# The staged fits are attenuation-prone (they differentiate / integrate noisy
# eRPM telemetry). The refinement instead drives the full coupled rigid-body
# model with the *motor command* (logged clean at 2 kHz, no telemetry lag),
# integrates rotor + rate dynamics forward from measured initial conditions over
# short windows, and minimises the trajectory error against measured rpm, body
# rates and body-z specific force. Integration is bias-free, and JAX autodiff
# gives exact gradients for a joint fit of all the rotor/thrust/torque/inertia
# parameters at once. Everything stays in the body frame (rates / rpm / specific
# force) so no attitude estimate is needed.

def _build_windows(pool, flips, win_s=0.12, stride_s=0.06, sub=2, max_windows=3000):
    """Slice flights into short fixed-length windows of (cmd, rpm, gyro, acc).

    Short windows keep J identified from the torque->rate relation rather than
    long-horizon integration drift. The gyro target is zero-phase smoothed (the
    rigid-body rate; the model cannot reproduce prop vibration, which would
    otherwise put a floor on the loss). flips align rate axes to the model frame.
    sub: decimation factor (2 -> ~1 kHz)."""
    cmd, rpm, gyro, acc = [], [], [], []
    dt_ref = None
    for s in pool:
        # contiguous airborne runs only (windows must be gap-free in time)
        for run in _air_runs(s, min_len=40):
            t = s.t[run]
            dt = float(np.median(np.diff(t))) * sub
            dt_ref = dt_ref or dt
            T = int(win_s / dt)
            step = max(int(stride_s / dt), 1)
            # smooth at full rate (zero-phase) then decimate -> rigid-body signals
            gy = np.c_[[_smooth(s.gyro[run, k], win=31) for k in range(3)]].T * flips
            rp = np.c_[[_smooth(s.rpm[run, k], win=31) for k in range(4)]].T
            cmdr = s.motor_norm[run][::sub]
            rpmr = rp[::sub]
            gyr = gy[::sub]
            acr = s.acc[run][::sub]              # specific force already +z up
            n = len(cmdr)
            for a in range(0, n - T, step):
                b = a + T
                cmd.append(cmdr[a:b]); rpm.append(rpmr[a:b])
                gyro.append(gyr[a:b]); acc.append(acr[a:b])
    cmd = np.asarray(cmd); rpm = np.asarray(rpm)
    gyro = np.asarray(gyro); acc = np.asarray(acc)
    # Excitation filter: the rate dynamics (hence J) are only identifiable where
    # the commanded torque actually varies. Select by INPUT excitation -- the
    # within-window spread of the per-axis torque command (rpm projected through
    # the mixer) -- NOT by the rate response gyro.std: selecting on the response
    # keeps windows where noise happened to move the rate, biasing J toward the
    # persistence (large-J) optimum.
    if len(rpm):
        torque_cmd = np.einsum("ntm,am->nta", rpm ** 2, MIX)   # [win,T,3]
        activity = torque_cmd.std(axis=1).max(axis=1)
    else:
        activity = np.zeros(len(cmd))
    keep = activity >= np.percentile(activity, 55)
    cmd, rpm, gyro, acc = cmd[keep], rpm[keep], gyro[keep], acc[keep]
    if len(cmd) > max_windows:                       # subsample for tractability
        idx = np.linspace(0, len(cmd) - 1, max_windows).astype(int)
        cmd, rpm, gyro, acc = cmd[idx], rpm[idx], gyro[idx], acc[idx]
    return dict(cmd=cmd, rpm=rpm, gyro=gyro, acc=acc, dt=dt_ref or 0.001)


def refine(pool, init: dict, mass: float, L: float, prop_inertia: float,
           flips, steps: int = 400, freeze_J: bool = False):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax

    W = _build_windows(pool, np.asarray(flips))
    if len(W["cmd"]) < 20:
        print("[refine ] too few windows; skipping")
        return init, {}
    cmd = jnp.asarray(W["cmd"]); rpm = jnp.asarray(W["rpm"])
    gyro = jnp.asarray(W["gyro"]); acc = jnp.asarray(W["acc"])
    dt = float(W["dt"])
    mix = jnp.asarray(MIX)
    rpm_to_rad = 2 * math.pi / 60.0
    print(f"[refine ] {len(W['cmd'])} windows x {cmd.shape[1]} steps @ {1/dt:.0f}Hz")

    # a2 (thrust) and L/mass are nailed by their direct fits and the acc channel;
    # they are FIXED here so the refinement cannot trade them off against the
    # weakly-observable rate channel (an earlier version drove a2 25% low to
    # shave the gyro loss). The refinement fits what it is genuinely good at:
    # the rotor dynamics + static map (rpm channel) and the yaw torque, and it
    # polishes J within a prior around the physically-grounded staged value.
    a2_fixed = float(init["a2"])
    # free params in log space (positivity); rpm_idle/span scaled to O(1)
    p0 = {
        "log_kappa": math.log(max(init["kappa"], 1e-15)),
        "log_Jxx": math.log(init["Jxx"]), "log_Jyy": math.log(init["Jyy"]),
        "log_Jzz": math.log(init["Jzz"]),
        "log_kup": math.log(max(init["rotor"]["rotor_dyn_coef"][0], 1.0)),
        "log_kdn": math.log(max(init["rotor"]["rotor_dyn_coef"][2], 1.0)),
        "idle": init["rotor"]["rpm_idle"] / 1e4,
        "span": (init["rotor"]["rpm_max"] - init["rotor"]["rpm_idle"]) / 1e4,
    }
    p0 = {k: jnp.asarray(float(v)) for k, v in p0.items()}

    # per-channel loss normalisation
    s_rpm = float(jnp.std(rpm)) or 1.0
    s_gyro = float(jnp.std(gyro)) or 1.0

    # Decoupled rollout: the torque path is driven by MEASURED rpm (so J/kappa are
    # identified from the true rotor speeds, not confounded by rotor-dynamics or
    # static-map error), while a separate predicted-rpm path (cmd -> rotor
    # dynamics) is matched to the measured rpm to fit kup/kdn and the static map.
    def rollout(p, cmd_w, rpm_meas_w, rpm0, w0):
        a2 = a2_fixed; kappa = jnp.exp(p["log_kappa"])
        J = jnp.array([jnp.exp(p["log_Jxx"]), jnp.exp(p["log_Jyy"]),
                       jnp.exp(p["log_Jzz"])])
        kup = jnp.exp(p["log_kup"]); kdn = jnp.exp(p["log_kdn"])
        idle = p["idle"] * 1e4; span = p["span"] * 1e4

        def step(carry, inp):
            rpm_pred, w, rpm_prev = carry
            c, rpm_m = inp
            # predicted-rpm path (rotor dynamics, for the rpm loss only)
            err = (idle + span * c) - rpm_pred
            rpm_pred_n = jnp.clip(
                rpm_pred + jnp.where(err > 0, kup * err, kdn * err) * dt, 0.0, None)
            # rigid-body path driven by measured rpm
            F = a2 * rpm_m ** 2
            Q = kappa * rpm_m ** 2
            rotor_dot_m = (rpm_m - rpm_prev) / dt
            gy = prop_inertia * jnp.dot(mix[2], rotor_dot_m * rpm_to_rad)
            tau = jnp.array([L * jnp.dot(mix[0], F), L * jnp.dot(mix[1], F),
                             jnp.dot(mix[2], Q) + gy])
            w_n = w + (tau - jnp.cross(w, J * w)) / J * dt
            spec = jnp.sum(F) / mass
            return (rpm_pred_n, w_n, rpm_m), (rpm_pred_n, w_n, spec)

        (_, _, _), (rpm_p, w_p, spec_p) = jax.lax.scan(
            step, (rpm0, w0, rpm0), (cmd_w, rpm_meas_w))
        return rpm_p, w_p, spec_p

    vroll = jax.vmap(rollout, in_axes=(None, 0, 0, 0, 0))

    # J is only weakly observable from this gentle data (eRPM telemetry lacks the
    # bandwidth to resolve the fast thrust differentials that rotate the frame),
    # so the rate-trajectory loss has a shallow persistence optimum at large J.
    # Anchor J (and the weakly-excited yaw kappa) to the physically-grounded
    # staged estimate with a log-space prior; thrust / rotor / static-map remain
    # freely fit (they are well-identified by the rpm and acc channels).
    logJ0 = jnp.array([math.log(init["Jxx"]), math.log(init["Jyy"]),
                       math.log(init["Jzz"])])
    logk0 = math.log(max(init["kappa"], 1e-15))
    # strong J prior: the dilution-free staged J is the trustworthy estimate;
    # the rate-trajectory loss has only a weak, persistence-biased pull on J, so
    # we keep it within ~15% of staged and let refinement own rotor/static/kappa.
    # freeze_J pins J hard (hardcoded/bench value) so refine only polishes the
    # rotor dynamics, static map and kappa.
    REG_J, REG_K = (1e7 if freeze_J else 30.0), 0.3

    def loss(p):
        rpm_p, w_p, _ = vroll(p, cmd, rpm, rpm[:, 0, :], gyro[:, 0, :])
        l_rpm = jnp.mean(((rpm_p - rpm) / s_rpm) ** 2)
        l_gyro = jnp.mean(((w_p - gyro) / s_gyro) ** 2)
        logJ = jnp.array([p["log_Jxx"], p["log_Jyy"], p["log_Jzz"]])
        reg = REG_J * jnp.mean((logJ - logJ0) ** 2) + REG_K * (p["log_kappa"] - logk0) ** 2
        return l_rpm + l_gyro + reg, (l_rpm, l_gyro)

    opt = optax.adam(3e-2)
    state = opt.init(p0)
    grad_fn = jax.jit(jax.value_and_grad(loss, has_aux=True))

    p = p0
    (l0, parts0), _ = grad_fn(p)
    hist = [float(l0)]
    for i in range(steps):
        (l, parts), g = grad_fn(p)
        updates, state = opt.update(g, state, p)
        p = optax.apply_updates(p, updates)
        hist.append(float(l))
    (lf, partsf), _ = grad_fn(p)

    out = dict(init)
    out["a2"] = a2_fixed
    out["kappa"] = float(jnp.exp(p["log_kappa"]))
    out["Jxx"] = float(jnp.exp(p["log_Jxx"]))
    out["Jyy"] = float(jnp.exp(p["log_Jyy"]))
    out["Jzz"] = float(jnp.exp(p["log_Jzz"]))
    out["rpm2thrust"] = [0.0, 0.0, out["a2"]]
    out["rpm2torque"] = [0.0, 0.0, out["kappa"]]
    out["J"] = [[out["Jxx"], 0, 0], [0, out["Jyy"], 0], [0, 0, out["Jzz"]]]
    kup = float(jnp.exp(p["log_kup"])); kdn = float(jnp.exp(p["log_kdn"]))
    out["rotor_dyn_coef"] = [kup, 0.0, kdn, 0.0]
    out["rpm_idle"] = float(p["idle"]) * 1e4
    out["rpm_max"] = out["rpm_idle"] + float(p["span"]) * 1e4
    out["thrust_min"] = out["a2"] * out["rpm_idle"] ** 2
    out["thrust_max"] = out["a2"] * out["rpm_max"] ** 2
    diag = dict(loss0=float(l0), lossf=float(lf),
                parts0=[float(x) for x in parts0],
                partsf=[float(x) for x in partsf], hist=hist)
    return out, diag


# ---------------------------------------------------------------------------
# stage 7: held-out validation (open-loop windowed rollout)
# ---------------------------------------------------------------------------
def _rollout_np(cmd, rpm0, w0, P, mass, L, prop_inertia, dt):
    """numpy open-loop rollout of one window (mirrors refine's model). Returns
    predicted (rpm, gyro, acc_z) trajectories [T,...]."""
    a2, kappa = P["a2"], P["kappa"]
    J = np.array([P["Jxx"], P["Jyy"], P["Jzz"]])
    kup, kdn = P["rotor_dyn_coef"][0], P["rotor_dyn_coef"][2]
    idle, span = P["rpm_idle"], P["rpm_max"] - P["rpm_idle"]
    rpm_to_rad = 2 * math.pi / 60.0
    rpm_s = rpm0.copy(); w = w0.copy()
    T = len(cmd)
    out_rpm = np.empty((T, 4)); out_w = np.empty((T, 3)); out_az = np.empty(T)
    for k in range(T):
        cmd_rpm = idle + span * cmd[k]
        err = cmd_rpm - rpm_s
        rdot = np.where(err > 0, kup * err, kdn * err)
        rpm_s = np.clip(rpm_s + rdot * dt, 0.0, None)
        F = a2 * rpm_s ** 2
        Q = kappa * rpm_s ** 2
        gy = prop_inertia * (MIX[2] @ (rdot * rpm_to_rad))
        tau = np.array([L * (MIX[0] @ F), L * (MIX[1] @ F), MIX[2] @ Q + gy])
        w = w + (tau - np.cross(w, J * w)) / J * dt
        out_rpm[k] = rpm_s; out_w[k] = w; out_az[k] = F.sum() / mass
    return out_rpm, out_w, out_az


def validate_open_loop(val, P, flips, mass, L, prop_inertia,
                       win_s=0.12, sub=2):
    """Open-loop rollout RMSE on the held-out flight (per-window, driven by the
    logged motor command from measured initial conditions). Compared against the
    zero-phase-smoothed (rigid-body) rate, matching the refinement target."""
    if val is None:
        return None
    flips = np.asarray(flips)
    m = np.where(val.air)[0]
    splits = np.where(np.diff(m) > 1)[0] + 1
    e_rpm, e_gyro, e_az, base_gyro = [], [], [], []
    for run in np.split(m, splits):
        if len(run) < 40:
            continue
        dt = float(np.median(np.diff(val.t[run]))) * sub
        T = int(win_s / dt)
        cmd = val.motor_norm[run][::sub]; rpm = val.rpm[run][::sub]
        gyro = (np.c_[[_smooth(val.gyro[run, k], win=31) for k in range(3)]].T
                * flips)[::sub]
        acc = val.acc[run][::sub]
        for a in range(0, len(cmd) - T, T):
            b = a + T
            pr, pw, paz = _rollout_np(cmd[a:b], rpm[a], gyro[a], P,
                                      mass, L, prop_inertia, dt)
            e_rpm.append(pr - rpm[a:b]); e_gyro.append(pw - gyro[a:b])
            e_az.append(paz - acc[a:b, 2]); base_gyro.append(gyro[a:b])
    if not e_rpm:
        return None
    e_rpm = np.vstack(e_rpm); e_gyro = np.vstack(e_gyro)
    e_az = np.concatenate(e_az); base = np.vstack(base_gyro)
    return dict(
        rpm_rmse=float(np.sqrt(np.mean(e_rpm ** 2))),
        gyro_rmse_dps=float(np.sqrt(np.mean(e_gyro ** 2)) * 180 / math.pi),
        gyro_nrmse=float(np.sqrt(np.mean(e_gyro ** 2)) / (np.std(base) + 1e-9)),
        acc_z_rmse=float(np.sqrt(np.mean(e_az ** 2))),
        n_windows=len(e_az) // 1)


# ---------------------------------------------------------------------------
# write params.toml
# ---------------------------------------------------------------------------
def _final_block(P, mass):
    """The [air75_wisp] keys to patch, formatted for params.toml."""
    return {
        "mass": round(mass, 5),
        "L": round(L_ARM, 5),
        "J": [[float(f"{P['Jxx']:.4e}"), 0.0, 0.0],
              [0.0, float(f"{P['Jyy']:.4e}"), 0.0],
              [0.0, 0.0, float(f"{P['Jzz']:.4e}")]],
        "rpm2thrust": [0.0, 0.0, float(f"{P['a2']:.6e}")],
        "rpm2torque": [0.0, 0.0, float(f"{P['kappa']:.6e}")],
        "thrust2torque": round(P["kappa"] / P["a2"], 6),
        "rotor_dyn_coef": [round(P["rotor_dyn_coef"][0], 4), 0.0,
                           round(P["rotor_dyn_coef"][2], 4), 0.0],
        "rotor_dyn_coef_simple": round(0.5 * (P["rotor_dyn_coef"][0]
                                              + P["rotor_dyn_coef"][2]), 4),
        "thrust_dyn_coef": round(0.5 * (P["rotor_dyn_coef"][0]
                                        + P["rotor_dyn_coef"][2]), 4),
        "thrust_min": round(P["thrust_min"], 5),
        "thrust_max": round(P["thrust_max"], 5),
        "prop_inertia": float(f"{PROP_INERTIA:.3e}"),
        "drag_matrix": P["drag_matrix"],
    }


def _dr_ranges(final, inert, mass):
    """Domain-randomisation [min, max] per parameter for sim-to-real training.

    Two sources of range, unioned: (1) IDENTIFICATION uncertainty -- what this
    fit could not pin -- for J (the OLS..SMA bracket) and the yaw torque kappa
    (the Q/T 0.003..0.006 prop band); (2) REAL-WORLD variation -- parameters the
    fit nails but that drift in flight -- mass and thrust (battery charge/pack,
    prop wear), rotor time constant, and the weakly-observed drag. Parameters
    that are both confident and stable (L, mixing, prop_inertia) are left fixed.
    Randomise log-uniform within each range."""
    a2 = final["a2"]
    # J: the full data bracket per axis; Jzz spans the planar sums of the bracket
    jxx = sorted((inert["Jxx_ols"], inert["Jxx_sma"]))
    jyy = sorted((inert["Jyy_ols"], inert["Jyy_sma"]))
    jzz = [jxx[0] + jyy[0], jxx[1] + jyy[1]]
    return {
        "_note": "log-uniform within [min,max]; union of fit uncertainty + "
                 "real-world (battery/prop) variation. central = params.toml.",
        "mass": [round(0.030, 5), round(0.040, 5)],          # 1S pack/payload spread
        "J_xx": [float(f"{jxx[0]:.3e}"), float(f"{jxx[1]:.3e}")],
        "J_yy": [float(f"{jyy[0]:.3e}"), float(f"{jyy[1]:.3e}")],
        "J_zz": [float(f"{jzz[0]:.3e}"), float(f"{jzz[1]:.3e}")],
        "rpm2thrust_c": [float(f"{0.88 * a2:.4e}"), float(f"{1.12 * a2:.4e}")],  # +-12%
        "rpm2torque_c": [float(f"{0.003 * a2:.4e}"), float(f"{0.006 * a2:.4e}")],  # Q/T band
        "rotor_dyn_up": [round(0.7 * final["rotor_dyn_coef"][0], 3),
                         round(1.3 * final["rotor_dyn_coef"][0], 3)],
        "rotor_dyn_dn": [round(0.7 * final["rotor_dyn_coef"][2], 3),
                         round(1.3 * final["rotor_dyn_coef"][2], 3)],
        "drag_xy": sorted([round(0.0, 6), round(2.0 * final["drag_matrix"][0][0], 6)]),
    }


def _patch_toml(path, block):
    """Replace the fitted [air75_wisp] keys in place, preserving comments and
    collapsing multi-line array values onto one line."""
    import re
    is_header = lambda s: re.fullmatch(r"\[[A-Za-z0-9_]+\]", s) is not None
    fmt = {k: json.dumps(v) for k, v in block.items()}
    lines = open(path).readlines()
    out, in_sec, i = [], False, 0
    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        if is_header(s):
            in_sec = (s == "[air75_wisp]")
        key = s.split("=")[0].strip()
        if in_sec and "=" in s and key in fmt:
            head, _, comment = ln.partition("#")
            depth = head.count("[") - head.count("]")
            while depth > 0 and i + 1 < len(lines):
                i += 1
                depth += lines[i].count("[") - lines[i].count("]")
            tail = f"  # {comment.strip()}" if comment.strip() else ""
            out.append(f"{key} = {fmt[key]}{tail}\n")
        else:
            out.append(ln)
        i += 1
    open(path, "w").writelines(out)


def _repo_dm_dir():
    """drone_models package dir inside THIS repo (the canonical, version-
    controlled source), not whatever copy happens to be installed."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "drone_models"))


def write_params(block):
    dm = _repo_dm_dir()
    toml_path = os.path.join(dm, "data", "params.toml")
    _patch_toml(toml_path, block)
    # load_params merges data/params.toml with first_principles/params.toml, so
    # the model file must carry an (empty) [air75_wisp] section too.
    fp = os.path.join(dm, "first_principles", "params.toml")
    if "[air75_wisp]" not in open(fp).read():
        with open(fp, "a") as f:
            f.write("\n[air75_wisp]\n")
    return toml_path, fp


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/home/luke/Downloads/flight_data/FLIGHT DATA")
    ap.add_argument("--mass-kg", type=float, default=DEFAULT_MASS_KG)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--refine-steps", type=int, default=400)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--val-session", default=None)
    ap.add_argument("--thrust2torque", type=float, default=None,
                    help="fix the prop aerodynamic Q/T (kappa = T2T*a2) and "
                         "identify Jzz from the yaw response instead of the "
                         "planar Jzz=Jxx+Jyy anchor. Use a bench/CAD/known-prop "
                         "value to break the kappa<->Jzz coupling and fix yaw.")
    ap.add_argument("--J", default=None,
                    help="hardcode the inertia diagonal as 'Jxx,Jyy,Jzz' (kg m^2), "
                         "overriding the weakly-identified fit and freezing J "
                         "through the refinement. The OLS..SMA bracket is still "
                         "reported and feeds the DR ranges.")
    ap.add_argument("--dr-out", default=None,
                    help="write the domain-randomisation ranges (JSON) to this path.")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print(f"Discovering sessions in {args.data_dir!r} ...")
    pool = discover_sessions(args.data_dir, args.mass_kg)
    if not pool:
        raise SystemExit("No usable airborne flights found.")

    # hold out one vicon flight for validation
    val_name = args.val_session
    if val_name is None:
        vic = [s for s in pool if s.has_vicon]
        val_name = (max(vic, key=lambda s: s.air.sum()).name if vic
                    else max(pool, key=lambda s: s.air.sum()).name)
    train = [s for s in pool if s.name != val_name]
    val = next((s for s in pool if s.name == val_name), None)
    print(f"\nTrain on {len(train)} flights; validate on {val_name}\n")

    thrust = fit_thrust(train, args.mass_kg)
    print(f"[thrust ] rpm2thrust a2={thrust['a2']:.4e} N/rpm^2  R2={thrust['r2']:.3f}  "
          f"resid={thrust['resid_N']*1000:.1f}mN  hover~{thrust['hover_total_N']:.3f}N")

    J_override = None
    if args.J:
        J_override = tuple(float(x) for x in args.J.split(","))
        assert len(J_override) == 3, "--J must be 'Jxx,Jyy,Jzz'"
    inert = fit_inertia_torque(train, thrust["a2"], L_ARM, PROP_INERTIA,
                               thrust2torque=args.thrust2torque,
                               J_override=J_override)
    jzz_tag = {"yaw": "(<-yaw, kappa fixed)", "hardcoded": "(hardcoded)"}.get(
        inert["jzz_mode"], "(=Jxx+Jyy anchor)")
    print(f"[inertia] Jxx={inert['Jxx']:.3e} Jyy={inert['Jyy']:.3e} "
          f"Jzz={inert['Jzz']:.3e}{jzz_tag}  kappa={inert['kappa']:.3e}  "
          f"Q/T={inert['thrust2torque']:.5f}\n"
          f"          J data bracket (OLS..SMA): Jxx[{inert['Jxx_ols']:.2e}..{inert['Jxx_sma']:.2e}] "
          f"Jyy[{inert['Jyy_ols']:.2e}..{inert['Jyy_sma']:.2e}]  "
          f"corr r/p/y={inert['corr_roll']:.2f}/{inert['corr_pitch']:.2f}/{inert['corr_yaw']:.2f}\n"
          f"          flips={inert['gyro_flips']}  ({inert['n']} windows)"
          + ("  *wide bracket -> J weakly identified; consider --thrust2torque or a bench J*"
             if inert['Jxx_sma'] > 2 * max(inert['Jxx_ols'], 1e-12) else ""))

    rotor = fit_rotor(train, thrust["a2"])
    print(f"[rotor  ] rpm idle/max={rotor['rpm_idle']:.0f}/{rotor['rpm_max']:.0f}  "
          f"thrust_min/max={rotor['thrust_min']:.4f}/{rotor['thrust_max']:.4f}N  "
          f"k_up/k_dn={rotor['rotor_dyn_coef'][0]:.2f}/{rotor['rotor_dyn_coef'][2]:.2f} "
          f"(tau {rotor['tau_up_ms']:.0f}/{rotor['tau_dn_ms']:.0f}ms)")

    drag = fit_drag(train, thrust["a2"], args.mass_kg)
    print(f"[drag   ] xy={drag.get('drag_xy', float('nan')):.4f} "
          f"z={drag.get('drag_z', float('nan')):.4f}  "
          f"observed_axes={drag.get('observed_axes', '?')}"
          + ("  *weakly observable (near-hover data) -> borrowed/DR*"
             if drag.get('borrowed') or drag.get('observed_axes') in ('none', 'x', 'y')
             else ""))

    flips = inert["gyro_flips"]
    # parameter dict shared by refine / validate / report
    staged = {
        "a2": thrust["a2"], "kappa": inert["kappa"],
        "Jxx": inert["Jxx"], "Jyy": inert["Jyy"], "Jzz": inert["Jzz"],
        "rpm2thrust": thrust["rpm2thrust"], "rpm2torque": inert["rpm2torque"],
        "J": inert["J"], "drag_matrix": drag.get("drag_matrix"),
        "rotor_dyn_coef": rotor["rotor_dyn_coef"],
        "rpm_idle": rotor["rpm_idle"], "rpm_max": rotor["rpm_max"],
        "thrust_min": rotor["thrust_min"], "thrust_max": rotor["thrust_max"],
        "rotor": rotor,
    }

    # ---- refinement -----------------------------------------------------
    if args.no_refine:
        final = staged
        rdiag = {}
    else:
        print("\nRefining (JAX trajectory matching) ...")
        final, rdiag = refine(train, staged, args.mass_kg, L_ARM, PROP_INERTIA,
                              flips, steps=args.refine_steps,
                              freeze_J=J_override is not None)
        final.setdefault("drag_matrix", staged["drag_matrix"])
        if rdiag:
            print(f"[refine ] loss {rdiag['loss0']:.4f} -> {rdiag['lossf']:.4f}  "
                  f"(rpm/gyro/acc {', '.join(f'{x:.3f}' for x in rdiag['partsf'])})")
            print(f"[refine ] J=({final['Jxx']:.3e},{final['Jyy']:.3e},"
                  f"{final['Jzz']:.3e})  a2={final['a2']:.4e}  kappa={final['kappa']:.3e}  "
                  f"k_up/dn={final['rotor_dyn_coef'][0]:.1f}/{final['rotor_dyn_coef'][2]:.1f}")

    # ---- validation: old vs staged vs refined --------------------------
    # Two horizons: 0.12 s (the refinement target -- torque/J fidelity) and
    # 0.50 s (re-init less often -> exposes slow drift / J & damping errors that
    # the short window hides because it re-anchors before they accumulate).
    print(f"\n=== held-out validation on {val_name} (open-loop rollout) ===")
    old = _old_params()
    rows = [("old (guess)", old), ("staged", staged), ("refined", final)]
    for horizon in (0.12, 0.50):
        print(f"  -- horizon {horizon*1000:.0f} ms --")
        for label, P in rows:
            if P is None:
                continue
            v = validate_open_loop(val, _vparams(P), flips, args.mass_kg, L_ARM,
                                   PROP_INERTIA, win_s=horizon)
            if v:
                print(f"    {label:12s} gyro {v['gyro_rmse_dps']:6.1f} deg/s "
                      f"(nrmse {v['gyro_nrmse']:.2f})  rpm {v['rpm_rmse']:6.0f}  "
                      f"acc_z {v['acc_z_rmse']:.2f} m/s^2")

    # ---- write ----------------------------------------------------------
    block = _final_block(final, args.mass_kg)
    print("\n=== final [air75_wisp] block ===")
    for k, val_ in block.items():
        print(f"  {k} = {json.dumps(val_)}")

    # ---- domain-randomisation ranges -----------------------------------
    dr = _dr_ranges(final, inert, args.mass_kg)
    print("\n=== domain-randomisation ranges (sim-to-real) ===")
    for k, v in dr.items():
        if not k.startswith("_"):
            print(f"  {k:14s} {json.dumps(v)}")
    if args.dr_out:
        with open(args.dr_out, "w") as f:
            json.dump(dr, f, indent=2)
        print(f"  -> wrote {args.dr_out}")

    if args.write:
        toml_path, fp = write_params(block)
        print(f"\nWROTE {toml_path}\n      {fp} (ensured [air75_wisp] section)")
    else:
        print("\n(dry-run; re-run with --write to patch params.toml)")
    if args.plots:
        make_plots(val, final, flips, args.mass_kg, rdiag)


def _vparams(P):
    """Coerce any param dict (old/staged/refined) into the flat keys the
    rollout needs."""
    if P is None:
        return None
    if "a2" in P and "rpm_idle" in P:
        return P
    a2 = P["rpm2thrust"][2]
    J = P["J"]
    rd = P.get("rotor_dyn_coef", [10.0, 0.0, 10.0, 0.0])
    a2 = P["rpm2thrust"][2]
    rpm_idle = math.sqrt(P["thrust_min"] / a2)
    rpm_max = math.sqrt(P["thrust_max"] / a2)
    return {"a2": a2, "kappa": P["rpm2torque"][2],
            "Jxx": J[0][0], "Jyy": J[1][1], "Jzz": J[2][2],
            "rotor_dyn_coef": rd, "rpm_idle": rpm_idle, "rpm_max": rpm_max,
            "thrust_min": P["thrust_min"], "thrust_max": P["thrust_max"]}


def _old_params():
    """The previous (pre-rigorous) air75_wisp block, for an honest before/after."""
    try:
        import tomllib
        table = os.path.join(_repo_dm_dir(), "data", "params.toml")
        with open(table, "rb") as f:
            return tomllib.load(f)["air75_wisp"]
    except Exception:
        return None


def make_plots(val, P, flips, mass, rdiag, win_s=0.12, sub=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flips = np.asarray(flips)
    out = os.path.join(os.path.dirname(__file__), "..", "reports", "sysid")
    os.makedirs(out, exist_ok=True)
    # a clean ~2 s airborne segment, rolled out in the SAME short windows the
    # validation uses (re-initialised from measured at each window start), then
    # stitched. A single multi-second open-loop rollout would diverge -- that is
    # expected for any drone model and is why both sim and validation re-anchor.
    m = np.where(val.air)[0]
    run = max(np.split(m, np.where(np.diff(m) > 1)[0] + 1), key=len)
    dt = float(np.median(np.diff(val.t[run]))) * sub
    T = int(win_s / dt)
    cmd = val.motor_norm[run][::sub]
    rpm = val.rpm[run][::sub]
    gyro = (np.c_[[_smooth(val.gyro[run, k], win=31) for k in range(3)]].T * flips)[::sub]
    seg = min(len(cmd) - 1, int(2.0 / dt))
    pw = np.full((seg, 3), np.nan); pr = np.full((seg, 4), np.nan)
    for a in range(0, seg - T, T):
        b = a + T
        r, w, _ = _rollout_np(cmd[a:b], rpm[a], gyro[a], _vparams(P),
                              mass, L_ARM, PROP_INERTIA, dt)
        pw[a:b] = w; pr[a:b] = r
    tt = np.arange(seg) * dt
    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    for j, name in enumerate(("roll", "pitch", "yaw")):
        c = f"C{j}"
        ax[0].plot(tt, np.degrees(gyro[:seg, j]), c, lw=0.8, label=f"meas {name}")
        ax[0].plot(tt, np.degrees(pw[:, j]), c, ls="--", lw=1.0)
    ax[0].set_ylabel("body rate (deg/s)"); ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_title(f"air75 sysid: {win_s*1000:.0f}ms re-init rollout (dashed) vs "
                    f"measured (solid) -- {val.name}")
    ax[0].set_xlabel("time (s)")
    ax[1].plot(tt, rpm[:seg, 0], "C0", lw=0.8, label="meas rpm0")
    ax[1].plot(tt, pr[:, 0], "C1--", lw=1.0, label="pred rpm0 (cmd->rotor dyn)")
    ax[1].set_ylabel("rpm motor0"); ax[1].legend(fontsize=8); ax[1].set_xlabel("time (s)")
    if rdiag.get("hist"):
        ax[2].semilogy(rdiag["hist"]); ax[2].set_ylabel("refine loss")
        ax[2].set_xlabel("Adam step"); ax[2].set_title("refinement convergence")
    fig.tight_layout()
    p = os.path.join(out, "rollout.png")
    fig.savefig(p, dpi=110)
    print(f"[plots  ] wrote {p}")


if __name__ == "__main__":
    main()
