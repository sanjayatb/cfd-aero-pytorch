"""Compatibility patch for trimesh with NumPy 2.x."""

from __future__ import annotations

import numpy as np
import trimesh
from trimesh import triangles, tol
from trimesh.base import float64, log

_PATCHED = False


def ensure_trimesh_numpy_ptp() -> None:
    """Monkey patch Trimesh.face_normals to avoid ndarray.ptp usage."""

    global _PATCHED
    if _PATCHED:
        return

    prop = trimesh.base.Trimesh.face_normals

    def _patched_face_normals(self: trimesh.Trimesh, values):
        if values is None:
            return

        values = np.asanyarray(values, order="C", dtype=float64)
        if len(values) == 0 or values.shape != self.faces.shape:
            log.debug("face_normals incorrect shape, ignoring!")
            return

        ptp = np.ptp(values)
        if not np.isfinite(ptp):
            log.debug("face_normals contain NaN, ignoring!")
            return
        if ptp < tol.merge:
            log.debug("face_normals all zero, ignoring!")
            return

        check, valid = triangles.normals(
            self.vertices.view(np.ndarray)[self.faces[:20]]
        )
        compare = np.zeros((len(valid), 3))
        compare[valid] = check
        if not np.allclose(compare, values[:20]):
            log.debug("face_normals didn't match triangles, ignoring!")
            return

        self._cache["face_normals"] = values

    trimesh.base.Trimesh.face_normals = property(
        prop.fget, _patched_face_normals, prop.fdel, prop.__doc__
    )

    _PATCHED = True

