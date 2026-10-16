# src/geometry/constants.py
from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]
Vec3: TypeAlias = FloatArray
IntArray: TypeAlias = NDArray[np.signedinteger[Any]]

EPS = 1e-12  # Shared geometric and near-zero comparison tolerance.
TWO_PI = 2.0 * np.pi  # Full circle in radians
