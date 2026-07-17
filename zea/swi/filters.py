"""Temporal filtering of tissue-velocity data.

The shear wave sits in a limited temporal band. Below it lies bulk cardiac motion
(large amplitude, low frequency) which would otherwise swamp the wave; above it
lies electronic and decorrelation noise. A bidirectional (zero-phase) Butterworth
bandpass isolates the wave.

Zero phase matters here more than usual: this is a *time-of-flight* measurement,
so any frequency-dependent phase delay introduced by the filter would shift the
apparent wave arrival and bias the estimated speed. ``filtfilt`` applies the
filter forwards and backwards, giving exactly zero phase distortion.

The passive and active modes need different bands, because the two wave sources
have very different frequency content:

* **Passive** (valve-closure waves): low frequency, and crucially the tracking
  PRF is low (~926 Hz -> Nyquist ~463 Hz). The literature's 750 Hz upper cutoff
  is *above Nyquist here* and cannot be used. :func:`band_for_prf` enforces this
  rather than letting the design silently alias.
* **Active** (acoustic radiation force): broader and higher band, and the PRF is
  high enough to support it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt


@dataclass(frozen=True)
class Band:
    """A temporal passband."""

    low: float   # Hz
    high: float  # Hz
    order: int = 6

    def __str__(self) -> str:
        return f"{self.low:g}-{self.high:g} Hz (order {self.order})"


# Mode presets. These match the values used by Keijzer et al. (Sci Rep 2020),
# a direct in-vivo comparison of the two methods, and are consistent with the
# distinct frequency content of the two wave sources.
#
# PASSIVE (natural valve-closure waves): low frequency content (<= ~150 Hz),
# long wavelength (a few cm). The band is 15-100 Hz. The low 15 Hz corner is
# important - it separates the wave from large-amplitude bulk cardiac motion -
# and the 100 Hz ceiling matches the natural-wave spectrum. This sits safely
# under the ~463 Hz Nyquist of the ~926 Hz passive tracking PRF.
PASSIVE_BAND = Band(low=15.0, high=100.0, order=6)

# ACTIVE (acoustic-radiation-force waves): broadband, higher frequency content
# (up to ~500 Hz), short wavelength (a few mm). The band is 75-750 Hz. The high
# 75 Hz low-corner is needed to reject the large low-frequency general heart
# motion that otherwise swamps the small (~5 um) ARF displacements. The 750 Hz
# ceiling needs the higher active tracking PRF (>=1852 Hz accumulated,
# ~3704 Hz sliding).
ACTIVE_BAND = Band(low=75.0, high=750.0, order=6)


class NyquistError(ValueError):
    """Raised when a requested passband cannot be realised at the given PRF."""


def band_for_prf(band: Band, prf: float, clip: bool = False) -> Band:
    """Validate a passband against the Nyquist limit of a tracking PRF.

    Args:
        band: Requested passband.
        prf: Tracking frame rate in Hz.
        clip: If ``True``, silently lower the upper cutoff to just under Nyquist
            instead of raising. If ``False`` (default), raise :class:`NyquistError`.

    Returns:
        A :class:`Band` guaranteed to be realisable at this PRF.

    Raises:
        NyquistError: if the band cannot be realised and ``clip`` is ``False``.
    """
    nyquist = prf / 2.0
    if band.low <= 0:
        raise NyquistError(f"lower cutoff must be > 0, got {band.low}")
    if band.low >= nyquist:
        raise NyquistError(
            f"lower cutoff {band.low:g} Hz is at or above the Nyquist frequency "
            f"({nyquist:g} Hz for PRF {prf:g} Hz). The wave cannot be recovered."
        )
    if band.high >= nyquist:
        if not clip:
            raise NyquistError(
                f"upper cutoff {band.high:g} Hz is at or above the Nyquist frequency "
                f"({nyquist:g} Hz for PRF {prf:g} Hz). Content above Nyquist is "
                f"aliased and cannot be filtered out afterwards. Either lower the "
                f"cutoff, raise the tracking PRF, or pass clip=True to accept a "
                f"reduced band."
            )
        # Leave a small margin below Nyquist for the filter transition.
        high = 0.95 * nyquist
        if high <= band.low:
            raise NyquistError(
                f"cannot clip: 95% of Nyquist ({high:g} Hz) is not above the lower "
                f"cutoff ({band.low:g} Hz)."
            )
        return Band(low=band.low, high=high, order=band.order)
    return band


def bandpass(
    v: np.ndarray,
    prf: float,
    band: Band,
    axis: int = 0,
    clip: bool = False,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the time axis.

    Args:
        v: Velocity array; time along ``axis`` (default 0, i.e.
            ``(n_frames, n_rho, n_theta)``).
        prf: Tracking frame rate in Hz.
        band: Passband to apply.
        axis: Time axis.
        clip: Passed to :func:`band_for_prf`.

    Returns:
        Filtered array of the same shape.
    """
    v = np.asarray(v, dtype=float)
    band = band_for_prf(band, prf, clip=clip)

    n = v.shape[axis]
    # filtfilt needs enough samples for its edge padding.
    min_len = 3 * (2 * band.order + 1)
    if n <= min_len:
        raise ValueError(
            f"need more than {min_len} time samples for an order-{band.order} "
            f"bidirectional filter, got {n}. Reduce the filter order or acquire "
            f"more frames."
        )

    nyquist = prf / 2.0
    # Second-order sections are numerically far better behaved than transfer-function
    # coefficients for higher-order bandpasses.
    sos = butter(
        band.order,
        [band.low / nyquist, band.high / nyquist],
        btype="bandpass",
        output="sos",
    )

    # NaNs (e.g. samples outside the imaged sector) would propagate through the
    # whole filtered trace. Replace them with zeros, filter, then restore.
    nan_mask = np.isnan(v)
    if nan_mask.any():
        v = np.where(nan_mask, 0.0, v)

    out = sosfiltfilt(sos, v, axis=axis)

    if nan_mask.any():
        out = np.where(nan_mask, np.nan, out)
    return out


def highpass(v: np.ndarray, prf: float, cutoff: float, order: int = 4, axis: int = 0) -> np.ndarray:
    """Zero-phase high-pass, e.g. to remove bulk motion without an upper cutoff."""
    v = np.asarray(v, dtype=float)
    nyquist = prf / 2.0
    if cutoff >= nyquist:
        raise NyquistError(f"cutoff {cutoff:g} Hz >= Nyquist {nyquist:g} Hz")
    sos = butter(order, cutoff / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, v, axis=axis)
