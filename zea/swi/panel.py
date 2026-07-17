"""Space-time panels: tissue velocity sampled along the anatomical M-line.

The panel is the central object of the analysis. Its axes are time (horizontal)
and distance along the septum (vertical); a propagating shear wave appears as a
tilted band whose slope is ds/dt = the group speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Panel:
    """A space-time panel of tissue velocity along an M-line.

    Attributes:
        data: ``(n_frames, n_points)`` velocity in m/s. Samples that fall outside
            the imaged sector are NaN.
        t: ``(n_frames,)`` time of each frame in seconds. For active mode t=0 is
            the push; for passive mode t=0 is the start of the acquisition.
        s: ``(n_points,)`` distance along the M-line in metres, from the line's
            origin (the valve, or the push location).
        label: Identifier of the M-line this panel came from.
    """

    data: np.ndarray
    t: np.ndarray
    s: np.ndarray
    label: str = ""

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.s = np.asarray(self.s, dtype=float).reshape(-1)
        if self.data.shape != (self.t.size, self.s.size):
            raise ValueError(
                f"panel data shape {self.data.shape} does not match "
                f"(n_t={self.t.size}, n_s={self.s.size})"
            )

    @property
    def dt(self) -> float:
        """Time step in seconds."""
        return float(self.t[1] - self.t[0]) if self.t.size > 1 else 0.0

    @property
    def ds(self) -> float:
        """Spatial step along the M-line in metres."""
        return float(self.s[1] - self.s[0]) if self.s.size > 1 else 0.0

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of samples that are inside the imaged sector."""
        return ~np.isnan(self.data)

    def window(self, t_range=None, s_range=None) -> "Panel":
        """Return a sub-panel restricted to a time and/or space range.

        Windowing is important for speed estimation: it lets you isolate the
        wave (e.g. the post-push interval, or one valve-closure event) and
        exclude reverberation and unrelated motion, which otherwise dominate the
        slope estimate.
        """
        ti = slice(None)
        si = slice(None)
        if t_range is not None:
            keep = np.flatnonzero((self.t >= t_range[0]) & (self.t <= t_range[1]))
            if keep.size == 0:
                raise ValueError(f"no frames within t_range={t_range}")
            ti = slice(keep[0], keep[-1] + 1)
        if s_range is not None:
            keep = np.flatnonzero((self.s >= s_range[0]) & (self.s <= s_range[1]))
            if keep.size == 0:
                raise ValueError(f"no points within s_range={s_range}")
            si = slice(keep[0], keep[-1] + 1)
        return Panel(
            data=self.data[ti, si],
            t=self.t[ti],
            s=self.s[si],
            label=self.label,
        )

    def energy(self) -> np.ndarray:
        """Per-frame wave energy (mean square velocity), useful to find wave events."""
        d = np.where(self.valid, self.data, 0.0)
        return np.mean(d**2, axis=1)


def build_panel(
    velocity: np.ndarray,
    mline,
    grid,
    t: np.ndarray,
    label: str = "",
) -> Panel:
    """Sample a velocity volume along an M-line to form a space-time panel.

    Args:
        velocity: ``(n_frames, n_rho, n_theta)`` tissue velocity.
        mline: :class:`~swe.geometry.MLine`.
        grid: :class:`~swe.geometry.PolarGrid`.
        t: ``(n_frames,)`` time of each velocity frame, in seconds.
        label: Panel label (defaults to the M-line's label).

    Returns:
        :class:`Panel`.
    """
    data = mline.sample(velocity, grid)  # (n_frames, n_points)
    return Panel(
        data=data,
        t=np.asarray(t, dtype=float),
        s=mline.arclength,
        label=label or mline.label,
    )


def find_wave_events(
    panel: Panel,
    n_events: int = 2,
    min_separation_s: float = 0.05,
    smooth_frames: int = 3,
):
    """Locate transient high-energy bursts in a panel (candidate wave events).

    For passive elastography the shear waves are generated at valve closure, so
    they appear as short bursts of elevated velocity energy at specific times in
    the cardiac cycle (mitral and aortic valve closure). This finds the strongest
    bursts so each can be analysed separately.

    Args:
        panel: The (filtered) velocity panel.
        n_events: How many events to return.
        min_separation_s: Minimum time between distinct events.
        smooth_frames: Boxcar smoothing of the energy trace, in frames.

    Returns:
        List of ``(t_peak, energy)`` sorted by descending energy.
    """
    e = panel.energy()
    if smooth_frames > 1:
        kern = np.ones(smooth_frames) / smooth_frames
        e = np.convolve(e, kern, mode="same")

    min_sep_frames = max(1, int(round(min_separation_s / panel.dt))) if panel.dt > 0 else 1

    events = []
    e_work = e.copy()
    for _ in range(n_events):
        i = int(np.argmax(e_work))
        if not np.isfinite(e_work[i]) or e_work[i] <= 0:
            break
        events.append((float(panel.t[i]), float(e[i])))
        lo = max(0, i - min_sep_frames)
        hi = min(e_work.size, i + min_sep_frames + 1)
        e_work[lo:hi] = -np.inf

    return events
