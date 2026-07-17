"""Read Verasonics-specific acquisition metadata from the .mat workspace.

These fields (per-buffer frame rate, and the PData beamforming grid) live only in
the Verasonics workspace and have no home in the generic zea schema, so reading
them belongs here in the application layer rather than in the toolbox converter.

The combined ``.mat`` is a MATLAB v7.3 file, which is really HDF5, so it must be
read with an HDF5 reader (``scipy.io.loadmat`` only handles the older v5 format
and raises on v7.3). We reuse the already-open ``VerasonicsFile`` handle, which
subclasses ``h5py.File`` - so these fields are read through the same machinery
the converter uses, without reopening the file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Speed of sound / transmit frequency used to convert the PData grid, which is
# expressed in wavelengths, to metres.
SOUND_SPEED = 1540.0
TRANSMIT_FREQUENCY = 3.125e6
WAVELENGTH_M = SOUND_SPEED / TRANSMIT_FREQUENCY


@dataclass
class BufferGrid:
    """The Verasonics PData beamforming grid for one buffer, in metres.

    Verasonics stores PData in wavelengths with:
      * ``Size   = [n_z, n_x, n_sections]``  (rows = depth, cols = lateral)
      * ``Origin = [x, y, z]``               (upper-left corner)
      * ``PDelta = [dx, dy, dz]``            (pixel spacing)

    This converts those to zea's cartesian ``xlims`` / ``zlims`` (metres) and the
    matching grid size, so zea reconstructs the identical grid. zea builds its
    axes with ``linspace(lim0, lim1, n)`` (endpoint inclusive), which matches the
    Verasonics ``Origin + arange(Size) * PDelta`` convention exactly - no
    half-pixel offset.
    """

    n_z: int
    n_x: int
    xlims: tuple
    zlims: tuple
    pixel_size_m: float

    @classmethod
    def from_pdata(cls, size, origin, pdelta, wavelength_m=WAVELENGTH_M):
        size = np.asarray(size).ravel()
        origin = np.asarray(origin, dtype=float).ravel()
        pdelta = np.asarray(pdelta, dtype=float).ravel()

        n_z, n_x = int(size[0]), int(size[1])
        dx_wl, dz_wl = pdelta[0], pdelta[2]
        x0_wl, z0_wl = origin[0], origin[2]

        x0 = x0_wl * wavelength_m
        x1 = (x0_wl + (n_x - 1) * dx_wl) * wavelength_m
        z0 = z0_wl * wavelength_m
        z1 = (z0_wl + (n_z - 1) * dz_wl) * wavelength_m

        return cls(
            n_z=n_z,
            n_x=n_x,
            xlims=(float(x0), float(x1)),
            zlims=(float(z0), float(z1)),
            pixel_size_m=float(dx_wl * wavelength_m),
        )

    def __str__(self):
        return (
            f"{self.n_z}x{self.n_x} px, "
            f"x[{self.xlims[0]*1e3:.1f},{self.xlims[1]*1e3:.1f}]mm "
            f"z[{self.zlims[0]*1e3:.1f},{self.zlims[1]*1e3:.1f}]mm "
            f"@ {self.pixel_size_m*1e6:.0f}um"
        )


@dataclass
class VerasonicsMeta:
    """Per-buffer Verasonics metadata that is not in the zea file."""

    fps: dict           # 0-based buffer index -> ActualFPS (Hz)
    grids: dict         # 0-based buffer index -> BufferGrid


# Which parameter struct drives each 0-based RF buffer's frame rate. This mirrors
# Resource.RcvBuffer in the setup script.
_BUFFER_FPS_STRUCT = {
    0: "Bmode_WB",   # widebeam B-mode
    1: "SW",         # active shear wave (measurement rate)
    2: "Bmode_FC",   # focused B-mode
    3: "Bmode_DW",   # passive shear wave (ultrafast)
    4: "Bmode_WB",   # orientation B-mode
}


def _read_scalar(vf, struct_name, field):
    """Read a scalar struct field (e.g. Bmode_WB.ActualFPS) from a v7.3 .mat.

    Returns a float, or None if the struct/field is absent.
    """
    if struct_name not in vf:
        return None
    group = vf[struct_name]
    if field not in group:
        return None
    return float(np.asarray(group[field][:]).ravel()[0])


def _read_pdata_field(vf, pdata, index, field):
    """Read one PData[index].field, dereferencing if PData is a struct array.

    In a v7.3 .mat a struct ARRAY like PData stores each field as an (N,1) array
    of object references (one per struct element). A scalar struct stores the
    field as a plain dataset. ``dereference_index`` on the VerasonicsFile handles
    both, so we reuse it.
    """
    dataset = pdata[field]
    return np.asarray(vf.dereference_index(dataset, index)).ravel()


def read_verasonics_meta(vf) -> VerasonicsMeta:
    """Read per-buffer frame rate and beamforming grid via an open VerasonicsFile.

    Args:
        vf: An open :class:`zea.data.convert.verasonics.VerasonicsFile` (an
            ``h5py.File`` subclass) for the Verasonics ``.mat``.

    Returns:
        :class:`VerasonicsMeta` with ``fps`` and ``grids`` keyed by 0-based
        buffer index. Missing fields are simply absent, so callers can fall back.
    """
    fps = {}
    for b, struct_name in _BUFFER_FPS_STRUCT.items():
        val = _read_scalar(vf, struct_name, "ActualFPS")
        if val is not None:
            fps[b] = val

    grids = {}
    if "PData" in vf:
        pdata = vf["PData"]
        # Number of struct-array elements. get_reference_size returns 1 for a
        # scalar struct, else the number of references.
        try:
            n = vf.get_reference_size(pdata["Size"])
        except Exception:
            n = 1
        for b in range(n):
            try:
                size = _read_pdata_field(vf, pdata, b, "Size")
                origin = _read_pdata_field(vf, pdata, b, "Origin")
                pdelta = _read_pdata_field(vf, pdata, b, "PDelta")
                grids[b] = BufferGrid.from_pdata(size, origin, pdelta)
            except Exception:
                continue

    return VerasonicsMeta(fps=fps, grids=grids)
