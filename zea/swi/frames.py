"""Frame assembly for shear-wave tracking acquisitions.

The word "frame" means two different things here, and conflating them is what
makes zea try to compound a whole measurement into one image.

* A **Verasonics frame** (the first axis of ``raw_data``) is one complete
  shear-wave *measurement*: a reference block, a push, and a post-push tracking
  block. For the active (SW) buffer that is ~100 transmits.
* A **tracking frame** is a single snapshot of the tissue, used to follow the
  shear wave through time. Each is formed from ONE diverging-wave transmit
  (``SW.na = 1``), or from one pulse-inversion PAIR of transmits.

zea's ``Beamform`` treats the transmit axis as "transmits to compound into one
image". That is right for B-mode, but wrong here: compounding all 100 transmits
of a measurement would both destroy the time information we need and blow up GPU
memory by ~100x. So before beamforming we *reinterpret* the transmit axis as a
time axis, leaving a single transmit per tracking frame.

Layout of the SW buffer's transmit axis (from the Verasonics event loop
``for measurement -> for acquisition -> for angle``, with harmonic imaging
doubling each acquisition into a +/- pulse-inversion pair)::

    [acq0+, acq0-, acq1+, acq1-, ..., acqN+, acqN-]
     |<--- reference (pre-push) --->|<--- post-push tracking --->|
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackingFrames:
    """Tracking frames extracted from one or more Verasonics measurements.

    Attributes:
        data: ``(n_meas, n_track, n_tx=1, n_ax, n_el, n_ch)`` ready for zea's
            beamformer, which will now compound a single transmit per frame.
        t: ``(n_track,)`` time of each tracking frame, in seconds, relative to
            the push (negative = reference / pre-push).
        prf: Effective tracking frame rate, in Hz.
        i_push: Index in ``t`` at which the push occurs (the first post-push
            frame). Reference frames are those before it.
        mode: How pulse-inversion pairs were combined.
    """

    data: np.ndarray
    t: np.ndarray
    prf: float
    i_push: int
    mode: str

    @property
    def n_meas(self) -> int:
        return self.data.shape[0]

    @property
    def n_track(self) -> int:
        return self.data.shape[1]

    def flatten_for_beamforming(self) -> np.ndarray:
        """Collapse measurements and tracking frames into zea's frame axis.

        Returns ``(n_meas * n_track, 1, n_ax, n_el, n_ch)``. Beamform this, then
        use :meth:`unflatten` to restore the measurement structure.
        """
        s = self.data.shape
        return self.data.reshape(s[0] * s[1], *s[2:])

    def unflatten(self, beamformed: np.ndarray) -> np.ndarray:
        """Restore ``(n_meas, n_track, ...)`` from a flattened beamformed array."""
        beamformed = np.asarray(beamformed)
        return beamformed.reshape(self.n_meas, self.n_track, *beamformed.shape[1:])


def assemble_tracking_frames(
    raw_data: np.ndarray,
    n_reference_acqs: int,
    pri: float,
    harmonic_imaging: bool = True,
    na: int = 1,
    mode: str = "accumulate",
) -> TrackingFrames:
    """Reinterpret the transmit axis of an SW buffer as tracking frames.

    Args:
        raw_data: ``(n_meas, n_tx, n_ax, n_el, n_ch)`` as loaded from the zea file.
        n_reference_acqs: Number of *acquisitions* (not transmits) before the
            push, i.e. ``SW.Ndetect_pre``.
        pri: Pulse repetition interval between successive transmits, in seconds
            (``Bmode.PRI``).
        harmonic_imaging: If ``True``, consecutive transmit pairs are a
            pulse-inversion (+/-) pair belonging to one acquisition.
        na: Number of angles per acquisition (``SW.na``). Only ``na = 1`` is
            supported; anything else would require angular compounding, which is
            a genuine transmit-compounding operation and belongs in the
            beamformer, not here.
        mode: How to combine the pulse-inversion pairs.

            * ``"accumulate"`` - non-overlapping pairs ``(0,1), (2,3), ...``.
              This is the conventional harmonic sum. Tracking PRF = 1/(2*PRI).
            * ``"sliding"`` - overlapping pairs ``(0,1), (1,2), (2,3), ...``,
              which DOUBLES the tracking frame rate to 1/PRI.
            * ``"none"`` - do not combine; every transmit becomes its own
              tracking frame (fundamental, not harmonic).

    Returns:
        :class:`TrackingFrames`.

    Note:
        The ``"sliding"`` mode is not free. Overlapping pairs share a transmit,
        so consecutive frames are correlated, and the harmonic cancellation
        assumes the tissue is quasi-static between the two transmits of a pair -
        which is exactly the assumption that a fast-moving shear wavefront
        violates. It raises the apparent frame rate but can smear the wavefront.
        Compare against ``"accumulate"`` rather than assuming it is better.
    """
    raw_data = np.asarray(raw_data)
    if raw_data.ndim != 5:
        raise ValueError(
            f"expected (n_meas, n_tx, n_ax, n_el, n_ch), got shape {raw_data.shape}"
        )
    if na < 1:
        raise ValueError(f"na must be >= 1, got {na}")

    n_meas, n_tx = raw_data.shape[0], raw_data.shape[1]
    pi = 2 if harmonic_imaging else 1

    # Transmits per TRACKING FRAME. The Verasonics event loop is
    #   for measurement -> for acquisition -> for angle
    # with harmonic imaging doubling each (acquisition, angle) into a +/- pair.
    # So one tracking frame spans `na` angles, each a pulse-inversion pair:
    #   [a0+, a0-, a1+, a1-, ...]   (na = 2, harmonic)
    # The PI pairs are recombined here (they are two transmits of the SAME
    # acquisition), but the ANGLES are kept as separate transmits, because
    # compounding angles is genuine transmit compounding and belongs in the
    # beamformer.
    tx_per_frame = na * pi

    if n_tx % tx_per_frame != 0:
        raise ValueError(
            f"n_tx={n_tx} is not divisible by na*({pi}) = {tx_per_frame} transmits "
            f"per tracking frame; the pulse-inversion pairing would be misaligned."
        )
    n_acqs = n_tx // tx_per_frame

    if not 0 <= n_reference_acqs <= n_acqs:
        raise ValueError(
            f"n_reference_acqs={n_reference_acqs} is outside the {n_acqs} "
            f"acquisitions present in this buffer."
        )

    if mode == "sliding" and na != 1:
        raise NotImplementedError(
            "sliding-window pulse inversion is only defined for na=1; with multiple "
            "angles a sliding pair would combine different transmit angles."
        )

    if not harmonic_imaging or mode == "none":
        # No PI recombination: every transmit is its own transmit, grouped into
        # tracking frames of `na` angles.
        frames = raw_data.reshape(n_meas, n_acqs, tx_per_frame, *raw_data.shape[2:])
        dt = tx_per_frame * pri
        i_push = n_reference_acqs
        n_track = n_acqs
        used_mode = "none"

    elif mode == "accumulate":
        # Group into (acquisition, angle, polarity), then sum over polarity.
        grouped = raw_data.reshape(
            n_meas, n_acqs, na, pi, *raw_data.shape[2:]
        ).astype(np.float32)
        if pi == 2:
            # pos + neg: the fundamental cancels, the harmonic survives.
            frames = grouped[:, :, :, 0] + grouped[:, :, :, 1]
        else:
            frames = grouped[:, :, :, 0]
        # frames: (n_meas, n_acqs, na, n_ax, n_el, n_ch) -> na is the transmit axis
        dt = tx_per_frame * pri
        i_push = n_reference_acqs
        n_track = n_acqs
        used_mode = "accumulate"

    elif mode == "sliding":
        # na == 1 (enforced above). Overlapping PI pairs: (0,1), (1,2), ...
        # doubles the tracking frame rate.
        pos = raw_data[:, :-1, ...].astype(np.float32)
        neg = raw_data[:, 1:, ...].astype(np.float32)
        # Successive transmits alternate polarity, so the sliding sum must
        # alternate its sign to keep forming a (+,-) harmonic pair.
        sign = np.ones(pos.shape[1], dtype=np.float32)
        sign[1::2] = -1.0
        summed = pos + neg * sign[None, :, None, None, None]
        frames = summed[:, :, None, ...]          # single transmit per frame
        dt = pri
        i_push = n_reference_acqs * pi
        n_track = frames.shape[1]
        used_mode = "sliding"

    else:
        raise ValueError(f"unknown mode {mode!r}; expected accumulate|sliding|none")

    t = (np.arange(n_track) - i_push) * dt
    prf = 1.0 / dt

    return TrackingFrames(
        data=frames,
        t=t,
        prf=prf,
        i_push=int(i_push),
        mode=used_mode,
    )


def remove_push_reverberation(
    frames: TrackingFrames,
    reverb_duration_s: float = 500e-6,
    interpolate: bool = True,
):
    """Discard tracking frames corrupted by post-push reverberation.

    The acoustic-radiation-force push leaves the transducer and tissue ringing
    for a few hundred microseconds. The first post-push tracking frames within
    that window are dominated by this reverberation rather than the shear wave,
    and using them corrupts the wave-arrival estimate exactly where the wave
    begins. Following the reference processing, these frames are removed; the
    gap between the reference block and the first clean post-push frame is then
    linearly interpolated so the frame axis stays uniformly sampled for the
    downstream temporal filter.

    Only frames *after the push* are affected; reference frames are untouched.

    Args:
        frames: The assembled :class:`TrackingFrames` (active mode).
        reverb_duration_s: Duration of the reverberation to discard after the
            push. ~500 us in the reference literature.
        interpolate: If ``True``, linearly interpolate the removed frames from
            the surrounding clean frames so the time axis remains uniform. If
            ``False``, the frames are dropped and the time axis is shortened.

    Returns:
        A new :class:`TrackingFrames` with the reverberation frames handled.
    """
    n_reverb = int(np.ceil(reverb_duration_s * frames.prf))
    if n_reverb <= 0:
        return frames

    i0 = frames.i_push
    i1 = min(i0 + n_reverb, frames.n_track)
    if i1 <= i0:
        return frames

    data = frames.data.copy().astype(np.float32)

    if interpolate:
        # Linearly interpolate each corrupted frame between the last reference
        # frame (just before the push) and the first clean post-push frame.
        left = i0 - 1 if i0 > 0 else i1
        right = i1
        if right < frames.n_track and left >= 0 and left != right:
            for k in range(i0, i1):
                w = (k - left) / (right - left)
                data[:, k] = (1.0 - w) * data[:, left] + w * data[:, right]
        else:
            # No clean frame on one side; zero the corrupted frames instead.
            data[:, i0:i1] = 0.0
        new_data, new_t = data, frames.t
    else:
        keep = np.r_[0:i0, i1:frames.n_track]
        new_data = data[:, keep]
        new_t = frames.t[keep]

    return TrackingFrames(
        data=new_data,
        t=new_t,
        prf=frames.prf,
        i_push=frames.i_push,
        mode=frames.mode,
    )


def split_reference_postpush(frames: TrackingFrames):
    """Split tracking frames into the pre-push reference and post-push blocks.

    The reference block carries the tissue motion present *before* the push
    (cardiac motion, clutter). It is used to reject that background from the
    post-push data and to flag measurements taken during large bulk motion, which
    are unreliable.

    Returns:
        ``(reference, postpush)`` - each ``(n_meas, n_k, 1, n_ax, n_el, n_ch)``.
    """
    i = frames.i_push
    return frames.data[:, :i], frames.data[:, i:]


# Scan parameters that carry one entry PER TRANSMIT. When the transmit axis is
# reinterpreted as time, these must be reduced to match the new transmit count,
# or the beamformer will reject the data (it checks e.g. t0_delays against
# (n_tx, n_el)).
PER_TRANSMIT_PARAMS = (
    "t0_delays",
    "tx_apodizations",
    "polar_angles",
    "azimuth_angles",
    "focus_distances",
    "transmit_origins",
    "initial_times",
    "tx_waveform_indices",
)


def reduce_scan_parameters(params, n_tx_out: int, tx_indices=None, strict: bool = True):
    """Point the scan parameters at the transmits of a reassembled tracking frame.

    ``assemble_tracking_frames`` moves the transmit axis into the time axis, so a
    tracking frame has far fewer transmits than the buffer originally stored
    (typically 1). The scan parameters, however, still describe every original
    transmit - ``t0_delays`` is ``(100, 80)`` when the beamformer now wants
    ``(1, 80)``.

    zea already has the right mechanism for this: :meth:`Parameters.set_transmits`
    chooses which transmits are active, ``n_tx`` becomes the number selected, and
    every per-transmit property (``t0_delays``, ``focus_distances``, ...) is
    sliced through that selection automatically. We therefore *select* the
    transmits rather than rewriting the underlying arrays - rewriting them leaves
    ``selected_transmits`` pointing at rows that no longer exist.

    This is only valid because every tracking transmit of the SW sequence uses the
    SAME transmit geometry: with ``SW.na = 1`` the event loop issues
    ``Event.tx = k + l`` with ``l`` always 1, so all tracking transmits point at
    one TX. The per-transmit arrays are therefore identical row-wise, and using
    any one of them represents all of them.

    Args:
        params: The zea ``Parameters`` object loaded alongside the raw data.
            Modified in place and also returned.
        n_tx_out: Transmits per tracking frame after reassembly (1 for ``na=1``).
        tx_indices: Which original transmit indices to select. Defaults to the
            first ``n_tx_out``.
        strict: If ``True``, verify that the per-transmit arrays really are
            identical across transmits before collapsing to a subset of them.
            Collapsing non-identical transmits would silently discard real
            differences, and would mean the transmit axis is NOT purely a time
            axis - which invalidates the whole reassembly.

    Returns:
        The same ``params`` object, with its transmit selection reduced.
    """
    if tx_indices is None:
        tx_indices = list(range(n_tx_out))
    tx_indices = [int(i) for i in np.asarray(tx_indices).reshape(-1)]

    if strict:
        # Select all transmits so we can inspect the full arrays, then verify
        # they are interchangeable.
        params.set_transmits("all")
        for name in PER_TRANSMIT_PARAMS:
            value = getattr(params, name, None)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.shape[0] <= 1:
                continue
            first = arr[tx_indices[0]]
            if not np.allclose(arr, first[None, ...], equal_nan=True):
                raise ValueError(
                    f"Cannot reduce to {n_tx_out} transmit(s): the rows of "
                    f"'{name}' are not identical across the {arr.shape[0]} "
                    "transmits, so they do NOT all share one transmit geometry. "
                    "Reinterpreting the transmit axis as a time axis would then be "
                    "wrong. Pass strict=False only if you are certain this is "
                    "intended."
                )

    params.set_transmits(tx_indices)
    return params
