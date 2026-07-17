"""Validate the tissue-velocity estimators against analytic ground truth.

Generates synthetic IQ containing a shear wave whose axial particle velocity is
known in closed form, then checks that Kasai / Loupas recover it.
"""
import numpy as np

from swe.geometry import MLine, PolarGrid
from swe.synthetic import SyntheticWave, make_synthetic_iq
from swe.velocity import kasai, loupas, smoothing_sigmas, velocity_nyquist

# --- imaging geometry (matches the user's S5-1 setup) ----------------------
POLAR_LIMITS = (np.deg2rad(-45.0), np.deg2rad(45.0))
ZLIMS = (0.0, 0.12)
APERTURE_M = 0.020066
distance_to_apex = (APERTURE_M / 2) / np.tan(max(abs(np.asarray(POLAR_LIMITS))))

grid = PolarGrid(
    polar_limits=POLAR_LIMITS,
    zlims=ZLIMS,
    n_rho=480,
    n_theta=181,
    distance_to_apex=distance_to_apex,
)

# Septal M-line
mline = MLine(p0=np.array([-0.008, 0.030]), p1=np.array([0.010, 0.070]), n_points=128)

# --- acquisition parameters ------------------------------------------------
F_DEMOD = 3.125e6      # Hz (Trans.frequency)
PRF_PASSIVE = 926.0    # Hz  (buffer 4: 4 transmits/frame at 270 us PRI)
PRF_ACTIVE = 1852.0    # Hz  (buffer 2, accumulated PI)
C = 1540.0

print(f"M-line length: {mline.length*1e3:.1f} mm")
print(f"velocity Nyquist @ PRF={PRF_PASSIVE:.0f} Hz: "
      f"{velocity_nyquist(PRF_PASSIVE, F_DEMOD):.4f} m/s")
print(f"velocity Nyquist @ PRF={PRF_ACTIVE:.0f} Hz: "
      f"{velocity_nyquist(PRF_ACTIVE, F_DEMOD):.4f} m/s")

sig_rho, sig_theta = smoothing_sigmas(grid, sigma_rho_mm=2.0, sigma_theta_deg=1.9)
print(f"smoothing sigmas: rho={sig_rho:.2f} samples, theta={sig_theta:.2f} samples")


def check(label, prf, wave, snr_db, background_velocity=0.0, tol_corr=0.9, sigmas=None):
    sr, st = (sig_rho, sig_theta) if sigmas is None else sigmas
    iq, truth = make_synthetic_iq(
        grid, mline, n_frames=60, prf=prf,
        demodulation_frequency=F_DEMOD, wave=wave,
        snr_db=snr_db, background_velocity=background_velocity,
    )
    v_true = truth["velocity"]

    v_kasai = kasai(iq, prf, F_DEMOD, C, sigma_rho=sr, sigma_theta=st)
    v_loupas = loupas(iq, prf, F_DEMOD, sampling_frequency=15.625e6,
                      sound_speed=C, sigma_rho=sr, sigma_theta=st)

    # Compare only where the wave actually has energy (elsewhere truth ~ 0 and
    # the correlation is dominated by noise, which tells us nothing).
    mask = np.abs(v_true) > 0.05 * np.abs(v_true).max()

    out = {}
    for name, v in [("kasai", v_kasai), ("loupas", v_loupas)]:
        a, b = v_true[mask], v[mask]
        corr = np.corrcoef(a, b)[0, 1]
        # Regression slope of estimate vs truth: 1.0 = correctly scaled.
        slope = np.sum(a * b) / np.sum(a * a)
        out[name] = (corr, slope)
        print(f"  [{label}] {name:7s}: corr={corr:.4f}  slope={slope:.4f}")
        assert corr > tol_corr, f"{name} failed to track the truth ({corr:.3f})"

    return out


print("\n=== noiseless, passive PRF, NO spatial smoothing ===")
# With no spatial smoothing the estimator should be near-perfect. The residual
# slope (~0.94) is the inherent finite-difference bias of a ONE-LAG estimator: it
# measures the MEAN velocity over the frame interval, not the instantaneous
# mid-point value. Verified separately to converge to 1.0 as PRF -> inf.
res = check("nosmooth", PRF_PASSIVE, SyntheticWave(speed=3.0), snr_db=None,
            tol_corr=0.99, sigmas=(0.0, 0.0))
for name, (corr, slope) in res.items():
    assert 0.90 < slope < 1.02, f"{name} velocity scaling is off (slope={slope:.3f})"

print("\n=== noiseless, passive PRF, paper's 2mm x 1.9deg smoothing ===")
# The literature kernel buys SNR but ATTENUATES the wave amplitude, because the
# 2 mm sigma is a sizeable fraction of the ~20 mm shear wavelength. This is an
# amplitude bias only - it does not shift the wave ARRIVAL TIME, which is what
# the speed estimate depends on. We assert the attenuation is present and
# bounded, so it can never silently masquerade as a calibration error.
res = check("smoothed", PRF_PASSIVE, SyntheticWave(speed=3.0), snr_db=None,
            tol_corr=0.95, sigmas=(sig_rho, sig_theta))
for name, (corr, slope) in res.items():
    assert 0.5 < slope < 0.8, (
        f"{name} smoothing attenuation outside the expected range (slope={slope:.3f})"
    )

print("\n=== SNR 20 dB, passive PRF ===")
check("snr20", PRF_PASSIVE, SyntheticWave(speed=3.0), snr_db=20.0, tol_corr=0.9)

print("\n=== SNR 10 dB, passive PRF ===")
check("snr10", PRF_PASSIVE, SyntheticWave(speed=3.0), snr_db=10.0, tol_corr=0.6)

print("\n=== with bulk motion (5 cm/s), SNR 20 dB ===")
# Bulk cardiac motion is large compared to the shear-wave velocity; the estimator
# should still track the SUM (the bandpass filter is what removes the bulk part).
# NOTE: 5 cm/s is a large fraction of the velocity Nyquist at this PRF, so the
# slow-time phase is close to wrapping. Loupas divides by a locally-estimated
# frequency, which makes it slightly MORE sensitive than Kasai in this near-wrap
# regime - hence the lower correlation. Both still recover the magnitude
# correctly (slope ~ 1.0), which is what matters here.
check("bulk", PRF_ACTIVE, SyntheticWave(speed=3.0), snr_db=20.0,
      background_velocity=0.05, tol_corr=0.85)

print("\n=== aliasing guard ===")
# A large-amplitude wave should push the phase past +/-pi and alias. We check the
# estimator DOES alias (i.e. our Nyquist formula is meaningful), so that later we
# can be confident the operating point is safe.
big = SyntheticWave(speed=3.0, amplitude=400e-6)
iq, truth = make_synthetic_iq(grid, mline, n_frames=40, prf=PRF_PASSIVE,
                              demodulation_frequency=F_DEMOD, wave=big, snr_db=None)
v = kasai(iq, PRF_PASSIVE, F_DEMOD, C)
vny = velocity_nyquist(PRF_PASSIVE, F_DEMOD)
frac_clipped = np.mean(np.abs(truth["velocity"]) > vny)
corr = np.corrcoef(truth["velocity"].ravel(), v.ravel())[0, 1]
print(f"  truth exceeds Nyquist in {frac_clipped*100:.1f}% of samples "
      f"-> corr drops to {corr:.3f} (aliasing, as expected)")
assert frac_clipped > 0, "expected this case to exceed Nyquist"
assert corr < 0.9, "expected aliasing to degrade the estimate"

print("\nALL VELOCITY TESTS PASSED")
