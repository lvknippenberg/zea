"""Axial tissue-velocity estimation from beamformed IQ data.

Two estimators are provided:

* :func:`kasai` - the classic one-lag autocorrelation (Kasai et al. 1985). This
  is what the reference cardiac-SWE literature uses. It assumes the Doppler
  frequency is small relative to the demodulation frequency and uses the
  *nominal* demodulation frequency in the velocity conversion.

* :func:`loupas` - the 2D autocorrelation estimator (Loupas et al. 1995), which
  additionally estimates the local centre frequency from the data instead of
  assuming it equals the demodulation frequency. It is more robust when the
  received spectrum shifts (e.g. frequency-dependent attenuation with depth),
  and the reference literature explicitly names it as a likely improvement over
  Kasai.

Both operate on IQ of shape ``(n_frames, n_rho, n_theta)`` and return velocity
of shape ``(n_frames - 1, n_rho, n_theta)``: the one-lag estimate lives *between*
consecutive frames.

Sign convention: positive velocity = motion **away** from the transducer
(increasing depth), which is the usual axial-Doppler convention.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

C_SOUND = 1540.0  # m/s, default speed of sound


def _smooth_complex(arr: np.ndarray, sigma_rho: float, sigma_theta: float) -> np.ndarray:
    """Gaussian-smooth a complex array over its two spatial axes.

    The smoothing is applied to the *complex* autocorrelation, before taking the
    phase angle. This is deliberate and matters: averaging the complex values
    and then taking the angle is a coherent (vector) average, which correctly
    down-weights low-magnitude noisy samples. Smoothing the angles instead would
    be an incoherent average and would break down across the +/-pi wrap.

    Args:
        arr: ``(..., n_rho, n_theta)`` complex array.
        sigma_rho: Smoothing sigma along the radial axis, in samples.
        sigma_theta: Smoothing sigma along the angular axis, in samples.
    """
    if sigma_rho <= 0 and sigma_theta <= 0:
        return arr
    sigma = [0.0] * (arr.ndim - 2) + [float(sigma_rho), float(sigma_theta)]
    real = gaussian_filter(arr.real, sigma=sigma, mode="nearest")
    imag = gaussian_filter(arr.imag, sigma=sigma, mode="nearest")
    return real + 1j * imag


# Spatial smoothing kernels for the autocorrelation, in physical units
# (radial mm, angular degrees), from Keijzer et al. (Sci Rep 2020). The passive
# kernel is ~3x larger than the active one, matching the ~10x longer wavelength
# of natural waves: a larger kernel buys SNR without blurring the (long) passive
# wavefront, whereas active waves need a tighter kernel to preserve their short
# (~few mm) wavelength.
PASSIVE_SMOOTHING_MM_DEG = (3.0, 5.6)   # (radial mm, angular deg)
ACTIVE_SMOOTHING_MM_DEG = (1.0, 1.9)


def smoothing_sigmas(
    grid,
    sigma_rho_mm: float = 1.0,
    sigma_theta_deg: float = 1.9,
) -> tuple[float, float]:
    """Convert physical smoothing widths to grid samples for a :class:`PolarGrid`.

    The reference literature specifies the spatial smoothing kernel in physical
    units (degrees by millimetres), which is natural on a polar grid. This maps
    those to sample counts for the given grid.

    The defaults are the ACTIVE kernel; for passive data use
    :data:`PASSIVE_SMOOTHING_MM_DEG` (or :func:`smoothing_sigmas_for_mode`).

    Returns:
        ``(sigma_rho_samples, sigma_theta_samples)``.
    """
    sigma_rho = (sigma_rho_mm * 1e-3) / grid.dr if grid.dr > 0 else 0.0
    dtheta_deg = np.rad2deg(grid.dtheta)
    sigma_theta = sigma_theta_deg / dtheta_deg if dtheta_deg > 0 else 0.0
    return float(sigma_rho), float(sigma_theta)


def smoothing_sigmas_for_mode(grid, mode: str) -> tuple[float, float]:
    """Smoothing sigmas (in grid samples) for ``"passive"`` or ``"active"``.

    Uses the mode-specific physical kernels
    (:data:`PASSIVE_SMOOTHING_MM_DEG` / :data:`ACTIVE_SMOOTHING_MM_DEG`).
    """
    if mode == "passive":
        mm, deg = PASSIVE_SMOOTHING_MM_DEG
    elif mode == "active":
        mm, deg = ACTIVE_SMOOTHING_MM_DEG
    else:
        raise ValueError(f"mode must be 'passive' or 'active', got {mode!r}")
    return smoothing_sigmas(grid, sigma_rho_mm=mm, sigma_theta_deg=deg)


def kasai(
    iq: np.ndarray,
    prf: float,
    demodulation_frequency: float,
    sound_speed: float = C_SOUND,
    sigma_rho: float = 0.0,
    sigma_theta: float = 0.0,
) -> np.ndarray:
    """Axial velocity via the one-lag autocorrelation (Kasai).

    Args:
        iq: ``(n_frames, n_rho, n_theta)`` complex IQ.
        prf: Frame rate of the tracking sequence, in Hz (i.e. 1 / dt between
            consecutive IQ frames).
        demodulation_frequency: Demodulation frequency in Hz.
        sound_speed: Speed of sound in m/s.
        sigma_rho: Spatial smoothing sigma (radial), in samples.
        sigma_theta: Spatial smoothing sigma (angular), in samples.

    Returns:
        ``(n_frames - 1, n_rho, n_theta)`` real axial velocity in m/s.
    """
    iq = np.asarray(iq)
    if not np.iscomplexobj(iq):
        raise TypeError("kasai expects complex IQ data")
    if iq.ndim != 3:
        raise ValueError(f"iq must be (n_frames, n_rho, n_theta), got {iq.shape}")
    if iq.shape[0] < 2:
        raise ValueError("need at least 2 frames to form a one-lag autocorrelation")

    # One-lag (slow-time) autocorrelation between consecutive frames.
    r1 = iq[1:] * np.conj(iq[:-1])

    # Smooth the COMPLEX autocorrelation, then take the angle (see _smooth_complex).
    r1 = _smooth_complex(r1, sigma_rho, sigma_theta)

    phase = np.angle(r1)
    return phase * sound_speed * prf / (4.0 * np.pi * demodulation_frequency)


def loupas(
    iq: np.ndarray,
    prf: float,
    demodulation_frequency: float,
    sampling_frequency: float,
    sound_speed: float = C_SOUND,
    sigma_rho: float = 0.0,
    sigma_theta: float = 0.0,
    freq_smoothing_rho: float = 8.0,
    freq_smoothing_theta: float = 4.0,
) -> np.ndarray:
    """Axial velocity via the 2D autocorrelation (Loupas).

    Unlike :func:`kasai`, this estimates the local centre frequency from the data
    (via the one-lag autocorrelation along the *fast-time* / depth axis) rather
    than assuming it equals the nominal demodulation frequency. The velocity is

        v = (c * prf / (4*pi)) * angle(R_slow) / (f_demod + f_shift)

    where ``f_shift`` is derived from the fast-time autocorrelation.

    Args:
        iq: ``(n_frames, n_rho, n_theta)`` complex IQ.
        prf: Tracking frame rate in Hz.
        demodulation_frequency: Nominal demodulation frequency in Hz.
        sampling_frequency: Axial (fast-time) sampling frequency in Hz. Needed to
            convert the fast-time phase into a frequency offset.
        sound_speed: Speed of sound in m/s.
        sigma_rho: Spatial smoothing sigma (radial), in samples.
        sigma_theta: Spatial smoothing sigma (angular), in samples.

    Returns:
        ``(n_frames - 1, n_rho, n_theta)`` real axial velocity in m/s.
    """
    iq = np.asarray(iq)
    if not np.iscomplexobj(iq):
        raise TypeError("loupas expects complex IQ data")
    if iq.ndim != 3 or iq.shape[0] < 2:
        raise ValueError("iq must be (n_frames >= 2, n_rho, n_theta) complex")

    # Slow-time (frame-to-frame) autocorrelation -> phase shift from motion.
    r_slow = iq[1:] * np.conj(iq[:-1])
    r_slow = _smooth_complex(r_slow, sigma_rho, sigma_theta)

    # Fast-time (depth) autocorrelation -> local centre frequency estimate.
    #
    # This MUST be averaged over a spatial neighbourhood. The centre frequency is
    # a slowly-varying property of the pulse/tissue, but its per-sample estimate
    # from a single speckle realisation is extremely noisy (empirically the
    # per-sample spread is a large fraction of f_demod). Since the velocity is
    # divided by this frequency, that noise would be injected straight into the
    # velocity estimate. We therefore always smooth it, with at least a modest
    # kernel, independently of the user's velocity-smoothing choice.
    r_fast = iq[:, 1:, :] * np.conj(iq[:, :-1, :])
    freq_sigma_rho = max(sigma_rho, freq_smoothing_rho)
    freq_sigma_theta = max(sigma_theta, freq_smoothing_theta)
    r_fast = _smooth_complex(r_fast, freq_sigma_rho, freq_sigma_theta)

    # Average over the frame pair so it aligns with r_slow's time axis, and pad
    # the depth axis back to n_rho (the estimate is between depth samples).
    r_fast_pair = 0.5 * (r_fast[1:] + r_fast[:-1])
    pad = ((0, 0), (0, 1), (0, 0))
    r_fast_pair = np.pad(r_fast_pair, pad, mode="edge")

    # Local frequency offset (Hz) from the fast-time phase.
    f_shift = np.angle(r_fast_pair) * sampling_frequency / (2.0 * np.pi)
    f_local = demodulation_frequency + f_shift

    # Guard against a degenerate local frequency (can happen in pure-noise
    # regions); fall back to the nominal demodulation frequency there.
    bad = np.abs(f_local) < 0.1 * abs(demodulation_frequency)
    f_local = np.where(bad, demodulation_frequency, f_local)

    phase = np.angle(r_slow)
    return phase * sound_speed * prf / (4.0 * np.pi * f_local)


def velocity_nyquist(prf: float, demodulation_frequency: float, sound_speed: float = C_SOUND) -> float:
    """Maximum unaliased axial velocity (m/s) for a one-lag estimator.

    Corresponds to a slow-time phase shift of +/-pi.
    """
    return sound_speed * prf / (4.0 * np.pi * demodulation_frequency) * np.pi
