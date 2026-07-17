"""Synthetic shear-wave IQ generator, for validating the SWE estimators.

This builds IQ data in which a shear wave of a *known* propagation speed travels
along a *known* path, so that the whole chain (velocity estimation -> filtering
-> panel -> speed estimation) can be checked against ground truth. Without this,
a wrong estimator and a noisy in-vivo measurement look identical.

Model
-----
Tissue is a field of static scatterers. A shear wave passing a point displaces it
axially by ``d(t, s)``, where ``s`` is the distance along the propagation path.
The wave is a windowed pulse travelling at speed ``c_shear``:

    d(t, s) = A * g(t - s / c_shear)

with ``g`` a modulated Gaussian of centre frequency ``f_shear``. An axial
displacement ``d`` shifts the received IQ phase by

    phi = 4 * pi * f_demod * d / c

which is exactly the relationship the Kasai/Loupas estimators invert. Generating
the data this way means the ground-truth *velocity* field is the analytic time
derivative ``dd/dt``, so we can validate the velocity estimator itself and not
just the final speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

C_SOUND = 1540.0


@dataclass
class SyntheticWave:
    """Parameters of a synthetic shear wave."""

    speed: float = 3.0            # m/s, the ground truth we want to recover
    amplitude: float = 20e-6      # m, peak axial displacement (20 um is realistic)
    centre_frequency: float = 150.0  # Hz, shear-wave centre frequency
    bandwidth: float = 100.0      # Hz, Gaussian envelope bandwidth
    t0: float = 2e-3              # s, time the wave leaves the origin


def _wave_displacement(t, s, wave: SyntheticWave):
    """Axial displacement d(t, s) of a travelling, windowed shear pulse."""
    tau = t - wave.t0 - s / wave.speed
    envelope = np.exp(-((tau * wave.bandwidth * np.pi) ** 2))
    return wave.amplitude * envelope * np.sin(2 * np.pi * wave.centre_frequency * tau)


def _wave_velocity(t, s, wave: SyntheticWave):
    """Analytic dd/dt: the ground-truth axial particle velocity."""
    tau = t - wave.t0 - s / wave.speed
    w = 2 * np.pi * wave.centre_frequency
    a = wave.bandwidth * np.pi
    envelope = np.exp(-((tau * a) ** 2))
    return wave.amplitude * envelope * (
        w * np.cos(w * tau) - 2 * (a**2) * tau * np.sin(w * tau)
    )


def make_synthetic_iq(
    grid,
    mline,
    n_frames: int,
    prf: float,
    demodulation_frequency: float,
    wave: SyntheticWave | None = None,
    sound_speed: float = C_SOUND,
    snr_db: float | None = None,
    speckle_seed: int = 0,
    background_velocity: float = 0.0,
    pulse_length_wavelengths: float = 2.0,
):
    """Generate synthetic IQ containing a shear wave travelling along ``mline``.

    The wave propagates along the M-line: a point at arclength ``s`` from the
    line's origin is displaced at time ``t`` by ``d(t, s)``. Points off the line
    are displaced according to their *projection* onto it, which makes the wave a
    plane wave travelling in the M-line direction - the geometry the panel-based
    speed estimators assume.

    Args:
        grid: :class:`~swe.geometry.PolarGrid` describing the imaging grid.
        mline: :class:`~swe.geometry.MLine` giving the propagation path.
        n_frames: Number of IQ frames to generate.
        prf: Tracking frame rate (Hz).
        demodulation_frequency: Demodulation frequency (Hz).
        wave: Wave parameters (defaults to :class:`SyntheticWave`).
        sound_speed: Speed of sound (m/s).
        snr_db: If given, add complex Gaussian noise at this SNR. ``None`` = noiseless.
        speckle_seed: RNG seed for the scatterer speckle pattern and noise.
        background_velocity: Constant axial velocity (m/s) added everywhere, to
            emulate bulk cardiac motion that the bandpass filter must reject.

    Returns:
        ``(iq, truth)`` where ``iq`` is ``(n_frames, n_rho, n_theta)`` complex and
        ``truth`` is a dict with the ground-truth ``velocity`` field
        ``(n_frames - 1, n_rho, n_theta)`` and the wave parameters.
    """
    wave = wave or SyntheticWave()
    rng = np.random.default_rng(speckle_seed)

    # Cartesian position of every grid point.
    ii, jj = np.meshgrid(np.arange(grid.n_rho), np.arange(grid.n_theta), indexing="ij")
    idx = np.stack([ii, jj], axis=-1).astype(float)
    xz = grid.index_to_xz(idx)  # (n_rho, n_theta, 2)

    # Project each point onto the M-line to get its arclength coordinate s.
    d_vec = mline.p1 - mline.p0
    d_hat = d_vec / np.linalg.norm(d_vec)
    rel = xz - mline.p0[None, None, :]
    s = rel[..., 0] * d_hat[0] + rel[..., 1] * d_hat[1]  # (n_rho, n_theta)

    # Static speckle: random scatterers convolved with the axial pulse envelope.
    #
    # The convolution matters. Real IQ is band-limited in fast time: the imaging
    # pulse spans several depth samples, so adjacent samples are CORRELATED.
    # Uncorrelated (white) scatterers would make the fast-time autocorrelation
    # pure noise, which breaks any estimator that derives a local centre
    # frequency from it (e.g. Loupas). Convolving with a Gaussian pulse of the
    # correct axial extent reproduces the real correlation length.
    scatterers = (
        rng.normal(size=xz.shape[:2]) + 1j * rng.normal(size=xz.shape[:2])
    ) / np.sqrt(2)

    # Axial pulse length ~ a few wavelengths of the imaging frequency.
    lambda_us = sound_speed / demodulation_frequency          # imaging wavelength (m)
    pulse_sigma_m = pulse_length_wavelengths * lambda_us / 2.355  # FWHM -> sigma
    pulse_sigma_samples = pulse_sigma_m / grid.dr if grid.dr > 0 else 0.0

    if pulse_sigma_samples > 0:
        speckle = (
            gaussian_filter1d(scatterers.real, pulse_sigma_samples, axis=0, mode="nearest")
            + 1j * gaussian_filter1d(scatterers.imag, pulse_sigma_samples, axis=0, mode="nearest")
        )
        # Restore unit power after smoothing.
        speckle = speckle / np.sqrt(np.mean(np.abs(speckle) ** 2))
    else:
        speckle = scatterers

    t = np.arange(n_frames) / prf
    iq = np.empty((n_frames, grid.n_rho, grid.n_theta), dtype=np.complex128)
    for k, tk in enumerate(t):
        disp = _wave_displacement(tk, s, wave)
        if background_velocity:
            disp = disp + background_velocity * tk
        # Axial displacement -> IQ phase shift.
        phase = 4.0 * np.pi * demodulation_frequency * disp / sound_speed
        iq[k] = speckle * np.exp(1j * phase)

    if snr_db is not None:
        sig_power = np.mean(np.abs(iq) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = (rng.normal(size=iq.shape) + 1j * rng.normal(size=iq.shape))
        noise *= np.sqrt(noise_power / 2.0)
        iq = iq + noise

    # Ground-truth velocity, evaluated at the mid-point of each frame pair (that
    # is where a one-lag estimator's estimate actually sits in time).
    t_mid = (t[:-1] + t[1:]) / 2.0
    truth_v = np.empty((n_frames - 1, grid.n_rho, grid.n_theta))
    for k, tk in enumerate(t_mid):
        truth_v[k] = _wave_velocity(tk, s, wave) + background_velocity

    truth = {
        "velocity": truth_v,
        "speed": wave.speed,
        "wave": wave,
        "arclength_field": s,
        "t": t,
        "t_mid": t_mid,
    }
    return iq, truth
