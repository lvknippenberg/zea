"""Group shear-wave speed estimation from a space-time panel.

A shear wave propagating along the M-line appears in the panel as a tilted band.
Its slope ds/dt is the group speed. Three estimators are provided, because they
fail in different ways and the reference literature reports that the Radon
approach can break down at in-vivo SNR:

* :func:`radon_speed` - project the panel along candidate slopes and pick the one
  that concentrates the wave energy most. Global, uses the whole band, robust to
  gaps, but sensitive to strong off-wave clutter.
* :func:`tof_speed` - cross-correlate the velocity trace at each position along
  the septum against a reference trace, then robustly fit lag vs. distance.
  Local, degrades gracefully, and gives a per-position residual that is a useful
  quality flag.
* :func:`manual_speed` - slope through two user-picked points, which is what the
  reference study ultimately used.

All return a :class:`SpeedEstimate` carrying the speed *and* a quality metric, so
that unreliable acquisitions can be rejected rather than silently averaged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import theilslopes


@dataclass
class SpeedEstimate:
    """Result of a group-speed estimation."""

    speed: float                     # m/s (positive = travelling away from s=0)
    quality: float                   # 0-1, estimator-specific confidence
    method: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.speed:.2f} m/s (quality {self.quality:.2f}, {self.method})"


# ---------------------------------------------------------------------------
# Radon
# ---------------------------------------------------------------------------
def radon_speed(
    panel,
    speed_range: tuple[float, float] = (0.5, 10.0),
    n_speeds: int = 200,
    mask: np.ndarray | None = None,
) -> SpeedEstimate:
    """Estimate group speed by projecting the panel along candidate slopes.

    For each candidate speed ``c`` we sum the panel along the line
    ``t = t0 + s / c`` for every offset ``t0``. When ``c`` matches the true wave
    speed the summation is coherent (it follows the wavefront) and the projection
    develops a sharp peak; at other speeds the sum smears out. We therefore pick
    the speed whose projection has the greatest variance.

    Using the *squared* velocity (energy) makes the estimate insensitive to the
    wave's oscillating polarity, which would otherwise cancel in a plain sum.

    Args:
        panel: The (filtered, windowed) :class:`~swe.panel.Panel`.
        speed_range: Candidate speeds to search, in m/s.
        n_speeds: Number of candidates.
        mask: Optional ``(n_t, n_s)`` boolean mask of samples to use.

    Returns:
        :class:`SpeedEstimate` with ``quality`` = normalised peak sharpness.
    """
    data = np.where(panel.valid, panel.data, 0.0)
    if mask is not None:
        data = np.where(mask, data, 0.0)

    # Energy, so the oscillating wave does not cancel itself.
    energy = data**2
    # Normalise per position so that a single very bright depth cannot dominate.
    norm = np.sqrt(np.mean(energy, axis=0, keepdims=True))
    norm[norm == 0] = 1.0
    energy = energy / norm

    n_t, n_s = energy.shape
    dt, ds = panel.dt, panel.ds
    if dt <= 0 or ds <= 0:
        raise ValueError("panel must have positive dt and ds")

    speeds = np.linspace(speed_range[0], speed_range[1], n_speeds)
    s_idx = np.arange(n_s)

    scores = np.empty(n_speeds)
    projections = []
    for k, c in enumerate(speeds):
        # Time-shift (in frames) of the wavefront at each position for this speed.
        shift = (s_idx * ds / c) / dt                     # (n_s,)
        # For every starting frame t0, sample the panel along the tilted line.
        t0 = np.arange(n_t)[:, None]                      # (n_t, 1)
        tt = t0 + shift[None, :]                          # (n_t, n_s)
        ss = np.broadcast_to(s_idx[None, :], tt.shape)
        vals = map_coordinates(
            energy, np.stack([tt.ravel(), ss.ravel()]), order=1,
            mode="constant", cval=0.0,
        ).reshape(tt.shape)
        proj = vals.mean(axis=1)                          # (n_t,)
        projections.append(proj)
        scores[k] = proj.var()

    best = int(np.argmax(scores))
    speed = float(speeds[best])

    # Quality: how much sharper the best projection is than the median candidate.
    # A flat score curve (no real wave) gives quality near 0.
    med = float(np.median(scores))
    peak = float(scores[best])
    quality = 0.0 if peak <= 0 else float(np.clip(1.0 - med / peak, 0.0, 1.0))

    return SpeedEstimate(
        speed=speed,
        quality=quality,
        method="radon",
        diagnostics={
            "speeds": speeds,
            "scores": scores,
            "projection": projections[best],
        },
    )


# ---------------------------------------------------------------------------
# Time-of-flight cross-correlation
# ---------------------------------------------------------------------------
def tof_speed(
    panel,
    reference_index: int = 0,
    max_lag_frames: int | None = None,
    min_correlation: float = 0.3,
    robust: bool = True,
) -> SpeedEstimate:
    """Estimate group speed from wave arrival times along the septum.

    Cross-correlates the velocity trace at every position against a reference
    trace, giving the wave's arrival lag as a function of distance. The slope of
    lag-vs-distance is 1/speed. A robust (Theil-Sen) fit is used by default,
    because a handful of positions with spurious lags - which is exactly what
    reverberation clutter produces - would wreck a least-squares line.

    Args:
        panel: The (filtered, windowed) panel.
        reference_index: Index along ``s`` used as the reference trace. 0 = the
            wave origin (valve / push location).
        max_lag_frames: Restrict the lag search (rejects implausible speeds).
        min_correlation: Positions whose peak correlation is below this are
            discarded as unreliable.
        robust: Use Theil-Sen (robust) instead of least squares.

    Returns:
        :class:`SpeedEstimate`; ``quality`` = fraction of positions that produced
        a usable, well-correlated lag.
    """
    data = np.where(panel.valid, panel.data, 0.0)
    n_t, n_s = data.shape
    dt, ds = panel.dt, panel.ds

    ref = data[:, reference_index]
    ref = ref - ref.mean()
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        return SpeedEstimate(np.nan, 0.0, "tof", {"reason": "empty reference trace"})

    if max_lag_frames is None:
        max_lag_frames = n_t - 1

    lags = np.full(n_s, np.nan)
    peaks = np.full(n_s, np.nan)
    for j in range(n_s):
        x = data[:, j]
        x = x - x.mean()
        xn = np.linalg.norm(x)
        if xn == 0:
            continue
        cc = np.correlate(x, ref, mode="full") / (xn * ref_norm)
        centre = n_t - 1
        lo = max(0, centre - max_lag_frames)
        hi = min(cc.size, centre + max_lag_frames + 1)
        seg = cc[lo:hi]
        k = int(np.argmax(seg))
        peak = float(seg[k])
        lag = (lo + k) - centre

        # Sub-sample refinement by parabolic interpolation of the correlation peak.
        gi = lo + k
        if 0 < gi < cc.size - 1:
            y0, y1, y2 = cc[gi - 1], cc[gi], cc[gi + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                lag = lag + 0.5 * (y0 - y2) / denom

        lags[j] = lag * dt
        peaks[j] = peak

    good = np.isfinite(lags) & (peaks >= min_correlation)
    n_good = int(good.sum())
    if n_good < 3:
        return SpeedEstimate(
            np.nan, 0.0, "tof",
            {"reason": f"only {n_good} usable positions", "lags": lags, "peaks": peaks},
        )

    s = panel.s[good]
    lag = lags[good]

    if robust:
        slope, intercept, lo_slope, hi_slope = theilslopes(lag, s)
        # Slope is dt/ds = 1/speed.
        speed = np.inf if slope == 0 else float(1.0 / slope)
        slope_ci = (lo_slope, hi_slope)
    else:
        A = np.vstack([s, np.ones_like(s)]).T
        slope, intercept = np.linalg.lstsq(A, lag, rcond=None)[0]
        speed = np.inf if slope == 0 else float(1.0 / slope)
        slope_ci = None

    # Goodness of fit.
    pred = intercept + (lag_slope := (1.0 / speed if np.isfinite(speed) else 0.0)) * s
    ss_res = float(np.sum((lag - pred) ** 2))
    ss_tot = float(np.sum((lag - lag.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    quality = float(np.clip(r2, 0.0, 1.0) * (n_good / n_s))

    return SpeedEstimate(
        speed=speed,
        quality=quality,
        method="tof",
        diagnostics={
            "lags": lags,
            "peaks": peaks,
            "good": good,
            "r2": r2,
            "n_good": n_good,
            "slope_ci": slope_ci,
            "intercept": intercept,
        },
    )


# ---------------------------------------------------------------------------
# Manual
# ---------------------------------------------------------------------------
def manual_speed(p0: tuple[float, float], p1: tuple[float, float]) -> SpeedEstimate:
    """Group speed from two points picked on the panel.

    This is what the reference study used when the automatic estimators proved
    too fragile at in-vivo SNR, so it is kept as a first-class option rather than
    a fallback hack.

    Args:
        p0: ``(t, s)`` of the wavefront at its origin - seconds and metres.
        p1: ``(t, s)`` of the wavefront further along the septum.

    Returns:
        :class:`SpeedEstimate` with ``quality`` = 1.0 (a human asserted it).
    """
    dt = float(p1[0] - p0[0])
    dsd = float(p1[1] - p0[1])
    if dt == 0:
        return SpeedEstimate(np.inf, 0.0, "manual", {"reason": "zero time difference"})
    return SpeedEstimate(
        speed=abs(dsd / dt),
        quality=1.0,
        method="manual",
        diagnostics={"p0": p0, "p1": p1},
    )
