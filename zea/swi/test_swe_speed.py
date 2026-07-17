"""End-to-end validation of the SWE chain against a KNOWN shear-wave speed.

synthetic IQ -> velocity -> bandpass -> panel -> speed

This is the test that matters: it checks the whole pipeline recovers the
ground-truth propagation speed, across noise levels, bulk motion, and both
acquisition modes.
"""
import numpy as np

from swe.filters import ACTIVE_BAND, PASSIVE_BAND, Band, NyquistError, band_for_prf, bandpass
from swe.geometry import MLine, PolarGrid
from swe.panel import build_panel
from swe.speed import radon_speed, tof_speed, manual_speed
from swe.synthetic import SyntheticWave, make_synthetic_iq
from swe.velocity import kasai, smoothing_sigmas

POLAR_LIMITS = (np.deg2rad(-45.0), np.deg2rad(45.0))
APERTURE_M = 0.020066
dap = (APERTURE_M / 2) / np.tan(max(abs(np.asarray(POLAR_LIMITS))))
grid = PolarGrid(POLAR_LIMITS, (0.0, 0.12), 480, 181, dap)

# Septal M-line, ~44 mm
mline = MLine(np.array([-0.008, 0.030]), np.array([0.010, 0.070]), n_points=128, label="septum")

F_DEMOD = 3.125e6
C = 1540.0
PRF_PASSIVE = 926.0
PRF_ACTIVE = 1852.0

sig_rho, sig_theta = smoothing_sigmas(grid, sigma_rho_mm=2.0, sigma_theta_deg=1.9)


def run_chain(prf, band, wave, snr_db, background_velocity=0.0, n_frames=120):
    iq, truth = make_synthetic_iq(
        grid, mline, n_frames=n_frames, prf=prf, demodulation_frequency=F_DEMOD,
        wave=wave, snr_db=snr_db, background_velocity=background_velocity,
    )
    v = kasai(iq, prf, F_DEMOD, C, sigma_rho=sig_rho, sigma_theta=sig_theta)
    vf = bandpass(v, prf, band)
    t_mid = truth["t_mid"]
    panel = build_panel(vf, mline, grid, t_mid)
    return panel, truth


# =========================================================================
print("=== Nyquist guard ===")
# The literature's 750 Hz upper cutoff is ABOVE Nyquist at the passive PRF.
# The filter must refuse to design it rather than silently alias.
try:
    band_for_prf(ACTIVE_BAND, PRF_PASSIVE)
    raise AssertionError("expected NyquistError for 750 Hz band at 926 Hz PRF")
except NyquistError as e:
    print(f"  correctly refused 75-750 Hz at PRF {PRF_PASSIVE:.0f} Hz")
    print(f"    ({str(e).split('.')[0]}.)")

ok = band_for_prf(PASSIVE_BAND, PRF_PASSIVE)
print(f"  passive band {ok} is valid at PRF {PRF_PASSIVE:.0f} Hz "
      f"(Nyquist {PRF_PASSIVE/2:.0f} Hz)")
ok = band_for_prf(ACTIVE_BAND, PRF_ACTIVE)
print(f"  active band  {ok} is valid at PRF {PRF_ACTIVE:.0f} Hz "
      f"(Nyquist {PRF_ACTIVE/2:.0f} Hz)")


# =========================================================================
def report(label, panel, truth, tol_rel=0.15):
    true_c = truth["speed"]
    rad = radon_speed(panel, speed_range=(0.5, 10.0), n_speeds=200)
    tof = tof_speed(panel, reference_index=0, min_correlation=0.2)

    print(f"\n  [{label}]  ground truth = {true_c:.2f} m/s")
    results = {}
    for est in (rad, tof):
        err = abs(est.speed - true_c) / true_c if np.isfinite(est.speed) else np.inf
        flag = "OK " if err <= tol_rel else "OFF"
        print(f"    {flag} {est.method:6s}: {est.speed:6.2f} m/s  "
              f"(err {err*100:5.1f}%, quality {est.quality:.2f})")
        results[est.method] = (est, err)
    return results


print("\n=== PASSIVE mode (PRF 926 Hz, 50-350 Hz band) ===")
for snr in [None, 20.0, 10.0]:
    label = "noiseless" if snr is None else f"SNR {snr:.0f} dB"
    panel, truth = run_chain(PRF_PASSIVE, PASSIVE_BAND,
                             SyntheticWave(speed=3.0, centre_frequency=120.0),
                             snr_db=snr)
    res = report(label, panel, truth)
    # Radon is the primary estimator; require it to be accurate when SNR allows.
    if snr is None or snr >= 20:
        assert res["radon"][1] < 0.15, f"radon failed at {label}"

print("\n=== PASSIVE with bulk cardiac motion (5 cm/s) ===")
# The bandpass must reject the bulk motion; the speed should be unaffected.
panel, truth = run_chain(PRF_PASSIVE, PASSIVE_BAND,
                         SyntheticWave(speed=3.0, centre_frequency=120.0),
                         snr_db=20.0, background_velocity=0.05)
res = report("bulk motion", panel, truth)
assert res["radon"][1] < 0.15, "bulk motion corrupted the speed estimate"

print("\n=== ACTIVE mode (PRF 1852 Hz, 75-750 Hz band) ===")
for snr in [None, 20.0, 10.0]:
    label = "noiseless" if snr is None else f"SNR {snr:.0f} dB"
    panel, truth = run_chain(PRF_ACTIVE, ACTIVE_BAND,
                             SyntheticWave(speed=3.0, centre_frequency=250.0),
                             snr_db=snr)
    res = report(label, panel, truth)
    if snr is None or snr >= 20:
        assert res["radon"][1] < 0.15, f"radon failed at {label}"

print("\n=== speed sweep (does it track the truth, or just guess ~3?) ===")
# A estimator that always returns ~3 m/s would pass every test above. Sweep the
# TRUE speed and check the estimate follows it.
for true_c in [1.5, 2.0, 3.0, 5.0, 7.0]:
    panel, truth = run_chain(PRF_ACTIVE, ACTIVE_BAND,
                             SyntheticWave(speed=true_c, centre_frequency=250.0),
                             snr_db=20.0)
    rad = radon_speed(panel, speed_range=(0.5, 12.0), n_speeds=300)
    tof = tof_speed(panel, reference_index=0, min_correlation=0.2)
    err_r = abs(rad.speed - true_c) / true_c
    err_t = abs(tof.speed - true_c) / true_c if np.isfinite(tof.speed) else np.inf
    print(f"  true {true_c:4.1f} -> radon {rad.speed:5.2f} ({err_r*100:4.1f}%)   "
          f"tof {tof.speed:6.2f} ({err_t*100:5.1f}%)")
    assert err_r < 0.20, f"radon failed to track true speed {true_c}"

print("\n=== manual estimator ===")
# Two points on the true wavefront: t = t0 + s/c
wave = SyntheticWave(speed=3.0)
s0, s1 = 0.005, 0.040
est = manual_speed((wave.t0 + s0 / wave.speed, s0), (wave.t0 + s1 / wave.speed, s1))
print(f"  {est}")
assert abs(est.speed - 3.0) < 1e-6

print("\nALL END-TO-END SPEED TESTS PASSED")
