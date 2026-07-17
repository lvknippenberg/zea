"""Polar-grid geometry and anatomical M-line handling for cardiac SWE.

The beamforming grid produced by zea with ``grid_type="polar"`` has shape
``(n_rho, n_theta, 3)``: rows are depth (radial) samples, columns are scan-line
angles, and the last axis holds the Cartesian ``(x, y, z)`` coordinate of each
grid point in metres.

For a phased array the grid is a regular lattice in ``(rho, theta)`` emanating
from a virtual apex located at ``(0, 0, -distance_to_apex)``. That regularity is
what lets us map a physical point back to fractional grid indices analytically
(no search / no scan conversion), which is what :class:`MLine` relies on to
sample a space-time panel along the septum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class PolarGrid:
    """Description of a zea polar beamforming grid.

    This mirrors the parameters zea uses to build the grid, so that a point in
    physical space can be converted to fractional ``(rho_index, theta_index)``
    coordinates without needing to search the grid.

    Attributes:
        polar_limits: ``(theta_min, theta_max)`` in radians.
        zlims: ``(z_min, z_max)`` depth limits in metres (as passed to zea).
        n_rho: Number of radial (depth) samples.
        n_theta: Number of angular (scan-line) samples.
        distance_to_apex: Distance from the transducer face to the virtual apex,
            in metres. For a phased array zea derives this as
            ``(aperture / 2) / tan(max |theta|)`` unless set explicitly.
    """

    polar_limits: tuple[float, float]
    zlims: tuple[float, float]
    n_rho: int
    n_theta: int
    distance_to_apex: float = 0.0

    # -- derived axes ---------------------------------------------------
    @property
    def rlims(self) -> tuple[float, float]:
        """Radial limits measured from the apex (metres)."""
        return (self.zlims[0], self.zlims[1] + self.distance_to_apex)

    @property
    def dr(self) -> float:
        """Radial sample spacing (metres)."""
        return (self.rlims[1] - self.rlims[0]) / self.n_rho

    @property
    def theta(self) -> np.ndarray:
        """Angles of the scan lines (radians), shape ``(n_theta,)``."""
        return np.linspace(self.polar_limits[0], self.polar_limits[1], self.n_theta)

    @property
    def rho(self) -> np.ndarray:
        """Radial sample positions from the apex (metres), shape ``(n_rho,)``."""
        return self.rlims[0] + np.arange(self.n_rho) * self.dr

    @property
    def dtheta(self) -> float:
        """Angular sample spacing (radians)."""
        if self.n_theta < 2:
            return 0.0
        return (self.polar_limits[1] - self.polar_limits[0]) / (self.n_theta - 1)

    @property
    def apex(self) -> np.ndarray:
        """Apex position in Cartesian coordinates ``(x, z)`` in metres."""
        return np.array([0.0, -self.distance_to_apex])

    # -- coordinate transforms ------------------------------------------
    def xz_to_polar(self, xz: np.ndarray) -> np.ndarray:
        """Convert Cartesian ``(x, z)`` points to ``(rho, theta)`` about the apex.

        Args:
            xz: ``(..., 2)`` array of ``(x, z)`` coordinates in metres.

        Returns:
            ``(..., 2)`` array of ``(rho, theta)``: distance from the apex in
            metres and angle in radians (0 = straight ahead, +x = positive).
        """
        xz = np.asarray(xz, dtype=float)
        dx = xz[..., 0] - self.apex[0]
        dz = xz[..., 1] - self.apex[1]
        rho = np.hypot(dx, dz)
        # theta measured from the +z (depth) axis, positive toward +x, matching
        # the direction vectors zea builds from ``linspace(*polar_limits)``.
        theta = np.arctan2(dx, dz)
        return np.stack([rho, theta], axis=-1)

    def polar_to_xz(self, rho_theta: np.ndarray) -> np.ndarray:
        """Convert ``(rho, theta)`` about the apex to Cartesian ``(x, z)``."""
        rho_theta = np.asarray(rho_theta, dtype=float)
        rho = rho_theta[..., 0]
        theta = rho_theta[..., 1]
        x = self.apex[0] + rho * np.sin(theta)
        z = self.apex[1] + rho * np.cos(theta)
        return np.stack([x, z], axis=-1)

    def xz_to_index(self, xz: np.ndarray) -> np.ndarray:
        """Convert Cartesian ``(x, z)`` points to fractional grid indices.

        Returns:
            ``(..., 2)`` array of ``(rho_index, theta_index)``. Indices are
            fractional (suitable for interpolation) and may fall outside the
            valid range if the point lies outside the imaged sector - use
            :meth:`in_bounds` to check.
        """
        rho_theta = self.xz_to_polar(xz)
        rho_idx = (rho_theta[..., 0] - self.rlims[0]) / self.dr
        if self.n_theta < 2:
            theta_idx = np.zeros_like(rho_idx)
        else:
            theta_idx = (rho_theta[..., 1] - self.polar_limits[0]) / self.dtheta
        return np.stack([rho_idx, theta_idx], axis=-1)

    def index_to_xz(self, idx: np.ndarray) -> np.ndarray:
        """Convert fractional ``(rho_index, theta_index)`` to Cartesian ``(x, z)``."""
        idx = np.asarray(idx, dtype=float)
        rho = self.rlims[0] + idx[..., 0] * self.dr
        theta = self.polar_limits[0] + idx[..., 1] * self.dtheta
        return self.polar_to_xz(np.stack([rho, theta], axis=-1))

    def in_bounds(self, idx: np.ndarray) -> np.ndarray:
        """Boolean mask of which fractional indices lie inside the grid."""
        idx = np.asarray(idx, dtype=float)
        ok_rho = (idx[..., 0] >= 0) & (idx[..., 0] <= self.n_rho - 1)
        ok_theta = (idx[..., 1] >= 0) & (idx[..., 1] <= self.n_theta - 1)
        return ok_rho & ok_theta

    @classmethod
    def from_zea_grid(cls, grid: np.ndarray, polar_limits, zlims, distance_to_apex):
        """Build from an existing zea grid array plus the parameters used to make it."""
        grid = np.asarray(grid)
        n_rho, n_theta = grid.shape[0], grid.shape[1]
        return cls(
            polar_limits=tuple(np.asarray(polar_limits, dtype=float)),
            zlims=tuple(np.asarray(zlims, dtype=float)),
            n_rho=int(n_rho),
            n_theta=int(n_theta),
            distance_to_apex=float(distance_to_apex),
        )


@dataclass
class MLine:
    """An anatomical M-line: a straight path along the septum.

    Defined by two points in physical Cartesian coordinates (metres), matching
    the two-point selection used in the MATLAB workflow. The line is sampled at
    ``n_points`` evenly spaced positions; ``arclength`` gives the distance along
    the septum from ``p0``, which is the spatial axis of the space-time panel.

    Attributes:
        p0: ``(x, z)`` start point in metres (at the valve / push location).
        p1: ``(x, z)`` end point in metres (along the septum).
        n_points: Number of samples along the line.
        label: Optional identifier (e.g. ``"observer1_line3"``).
    """

    p0: np.ndarray
    p1: np.ndarray
    n_points: int = 128
    label: str = ""

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, dtype=float).reshape(2)
        self.p1 = np.asarray(self.p1, dtype=float).reshape(2)

    @property
    def length(self) -> float:
        """Total length of the M-line in metres."""
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def arclength(self) -> np.ndarray:
        """Distance from ``p0`` for each sample point (metres), shape ``(n_points,)``."""
        return np.linspace(0.0, self.length, self.n_points)

    @property
    def ds(self) -> float:
        """Spacing between samples along the line (metres)."""
        if self.n_points < 2:
            return 0.0
        return self.length / (self.n_points - 1)

    def sample_coords(self) -> np.ndarray:
        """Physical ``(x, z)`` coordinates of the samples, shape ``(n_points, 2)``."""
        t = np.linspace(0.0, 1.0, self.n_points)[:, None]
        return self.p0[None, :] + t * (self.p1 - self.p0)[None, :]

    def grid_indices(self, grid: PolarGrid) -> np.ndarray:
        """Fractional ``(rho_index, theta_index)`` of the samples, shape ``(n_points, 2)``."""
        return grid.xz_to_index(self.sample_coords())

    def sample(
        self,
        volume: np.ndarray,
        grid: PolarGrid,
        order: int = 1,
        cval: float = np.nan,
    ) -> np.ndarray:
        """Sample a volume along the M-line to build a space-time panel.

        Args:
            volume: ``(n_frames, n_rho, n_theta)`` array (e.g. tissue velocity).
                A single ``(n_rho, n_theta)`` frame is also accepted.
            grid: The polar grid the volume lives on.
            order: Interpolation order (1 = bilinear).
            cval: Value used for samples outside the grid.

        Returns:
            ``(n_frames, n_points)`` panel (or ``(n_points,)`` for a single frame).
            Points outside the imaged sector are set to ``cval``.
        """
        volume = np.asarray(volume)
        single = volume.ndim == 2
        if single:
            volume = volume[None, ...]
        if volume.ndim != 3:
            raise ValueError(
                f"volume must be (n_frames, n_rho, n_theta) or (n_rho, n_theta), "
                f"got shape {volume.shape}"
            )
        if volume.shape[1:] != (grid.n_rho, grid.n_theta):
            raise ValueError(
                f"volume spatial shape {volume.shape[1:]} does not match the grid "
                f"({grid.n_rho}, {grid.n_theta})"
            )

        idx = self.grid_indices(grid)  # (n_points, 2)
        coords = np.stack([idx[:, 0], idx[:, 1]], axis=0)  # (2, n_points)
        outside = ~grid.in_bounds(idx)

        panel = np.empty((volume.shape[0], self.n_points), dtype=float)
        for i, frame in enumerate(volume):
            vals = map_coordinates(
                np.asarray(frame, dtype=float),
                coords,
                order=order,
                mode="constant",
                cval=cval,
            )
            vals[outside] = cval
            panel[i] = vals

        return panel[0] if single else panel


@dataclass
class MLineSet:
    """A collection of M-lines (e.g. several drawn per observer).

    The reference literature draws multiple M-lines per acquisition to quantify
    the variability of the wave-speed estimate; this container keeps them
    together so a whole set can be analysed and saved as one unit.
    """

    lines: list[MLine] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)

    def __getitem__(self, i) -> MLine:
        return self.lines[i]

    def add(self, p0, p1, n_points: int = 128, label: str = "") -> MLine:
        """Add an M-line defined by two physical ``(x, z)`` points."""
        line = MLine(p0=p0, p1=p1, n_points=n_points, label=label or f"line{len(self)}")
        self.lines.append(line)
        return line
