from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PeakResult:
    selected_x: int
    max_energy: float
    plateau_start_x: int
    plateau_end_x: int
    plateau_width: int


def ensure_2d_image(image: np.ndarray, source: Path | str = "<array>") -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D en {source}, pero llego con shape {image.shape}")
    return np.asarray(image, dtype=np.float64)


def compute_vertical_profile(image: np.ndarray) -> np.ndarray:
    image_2d = ensure_2d_image(image)
    return np.sum(image_2d, axis=0)


def select_peak_with_plateau(
    profile_1d: np.ndarray,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> PeakResult:
    if profile_1d.ndim != 1:
        raise ValueError("El perfil de energia debe ser 1D.")
    if profile_1d.size == 0:
        raise ValueError("El perfil de energia esta vacio.")

    profile = np.asarray(profile_1d, dtype=np.float64)
    max_energy = float(np.max(profile))
    tolerance = max(float(atol), abs(max_energy) * float(rtol))
    plateau_mask = profile >= (max_energy - tolerance)

    best_start = -1
    best_end = -1
    best_width = -1

    current_start: int | None = None
    for index, is_on_plateau in enumerate(plateau_mask):
        if is_on_plateau and current_start is None:
            current_start = index
        elif not is_on_plateau and current_start is not None:
            current_end = index - 1
            current_width = current_end - current_start + 1
            if current_width > best_width:
                best_start = current_start
                best_end = current_end
                best_width = current_width
            current_start = None

    if current_start is not None:
        current_end = profile.size - 1
        current_width = current_end - current_start + 1
        if current_width > best_width:
            best_start = current_start
            best_end = current_end
            best_width = current_width

    if best_start < 0:
        peak_index = int(np.argmax(profile))
        best_start = peak_index
        best_end = peak_index
        best_width = 1

    selected_x = (best_start + best_end) // 2
    return PeakResult(
        selected_x=selected_x,
        max_energy=max_energy,
        plateau_start_x=best_start,
        plateau_end_x=best_end,
        plateau_width=best_width,
    )
