"""Validate swe.geometry.PolarGrid against zea's real polar_pixel_grid.

The whole M-line sampling path depends on our analytic (x,z) -> (rho_idx, theta_idx)
inverse being exactly consistent with the grid zea actually builds. We therefore
extract zea's polar_pixel_grid / radial_pixel_grid and compare directly.
"""
import ast
import textwrap

import numpy as np

from swe.geometry import MLine, PolarGrid

# --- pull zea's real grid builders (no keras import needed) -----------------
SRC = "zea/zea-main/zea/beamform/pixelgrid.py"
src_text = open(SRC).read()
tree = ast.parse(src_text)
want = {"polar_pixel_grid", "radial_pixel_grid"}
ns = {"np": np}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(textwrap.dedent(ast.get_source_segment(src_text, node)), ns)
polar_pixel_grid = ns["polar_pixel_grid"]

# --- geometry from the user's setup ----------------------------------------
LAMBDA_M = 1540.0 / 3.125e6          # 0.4928 mm
APERTURE_M = 0.020066                # 80 elements, from probe_geometry x-range
POLAR_LIMITS = (np.deg2rad(-45.0), np.deg2rad(45.0))
ZLIMS = (0.0, 0.12)
N_RHO, N_THETA = 480, 181

distance_to_apex = (APERTURE_M / 2) / np.tan(max(abs(np.asarray(POLAR_LIMITS))))
print(f"distance_to_apex = {distance_to_apex * 1e3:.3f} mm")

zea_grid = polar_pixel_grid(POLAR_LIMITS, ZLIMS, N_RHO, N_THETA, distance_to_apex)
print(f"zea grid shape: {zea_grid.shape}  (expect ({N_RHO}, {N_THETA}, 3))")
assert zea_grid.shape == (N_RHO, N_THETA, 3)

grid = PolarGrid(
    polar_limits=POLAR_LIMITS,
    zlims=ZLIMS,
    n_rho=N_RHO,
    n_theta=N_THETA,
    distance_to_apex=distance_to_apex,
)

# --- 1. forward: does our index->xz reproduce zea's grid coordinates? -------
ii, jj = np.meshgrid(np.arange(N_RHO), np.arange(N_THETA), indexing="ij")
idx = np.stack([ii, jj], axis=-1).astype(float)
ours_xz = grid.index_to_xz(idx)                 # (n_rho, n_theta, 2)
zea_xz = zea_grid[..., [0, 2]]                  # take (x, z)

err = np.abs(ours_xz - zea_xz)
print(f"\n[forward] max |our (x,z) - zea (x,z)| = {err.max():.3e} m")
assert err.max() < 1e-9, "our index->xz does not match zea's grid"

# --- 2. inverse: xz -> index must round-trip to the original indices --------
# NOTE: at rho = 0 (the apex row, i = 0) every theta maps to the SAME physical
# point, so theta is mathematically undefined there and cannot be recovered.
# That row is a coordinate singularity, not a mapping error, and lies outside
# any real imaging region - exclude it from the round-trip check.
back = grid.xz_to_index(zea_xz)
idx_err = np.abs(back - idx)
apex_row = grid.rho <= 0.0
interior = ~apex_row
print(f"[inverse] apex rows excluded (rho == 0): {apex_row.sum()}")
print(f"[inverse] max |round-trip index error| = {idx_err[interior].max():.3e}")
assert idx_err[interior].max() < 1e-6, "xz->index is not the inverse of index->xz"

# --- 3. M-line sampling recovers a known field ------------------------------
# Build a field whose value equals the physical depth z; sampling along an
# M-line must then return the z of each sample point.
z_field = zea_grid[..., 2]                       # (n_rho, n_theta)
p0 = np.array([-0.010, 0.030])                   # (x, z) metres
p1 = np.array([0.012, 0.075])
line = MLine(p0=p0, p1=p1, n_points=64, label="septum")
sampled = line.sample(z_field, grid)
expected_z = line.sample_coords()[:, 1]
inside = grid.in_bounds(line.grid_indices(grid))
samp_err = np.abs(sampled[inside] - expected_z[inside])
print(f"\n[M-line] length = {line.length * 1e3:.2f} mm, ds = {line.ds * 1e3:.3f} mm")
print(f"[M-line] samples inside sector: {inside.sum()}/{line.n_points}")
print(f"[M-line] max |sampled z - true z| = {samp_err.max():.3e} m")
assert samp_err.max() < 5e-5, "M-line sampling does not recover the known field"

# --- 4. a point outside the sector is flagged, not silently wrapped ---------
far = MLine(p0=np.array([0.0, 0.02]), p1=np.array([0.20, 0.02]), n_points=32)
vals = far.sample(z_field, grid)
n_nan = np.isnan(vals).sum()
print(f"\n[bounds] out-of-sector M-line -> {n_nan}/{far.n_points} samples are NaN")
assert n_nan > 0, "expected out-of-sector samples to be NaN"

print("\nALL GEOMETRY TESTS PASSED")
