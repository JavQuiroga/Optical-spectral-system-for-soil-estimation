from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class RegionStats:
    name: str
    start_nm: float
    end_nm: float
    mean_dispersion: float
    mean_resolution: float
    band_count: float


def _read_one_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"wavelength_nm", "x_centroid_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    return df


def _resolve_csv_paths(csv_arg: str) -> list[Path]:
    path = Path(csv_arg)
    if "*" in csv_arg or "?" in csv_arg:
        return sorted(Path(".").glob(csv_arg))
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path]


def load_dispersion_data(csv_arg: str) -> pd.DataFrame:
    paths = _resolve_csv_paths(csv_arg)
    if not paths:
        raise ValueError(f"No CSV files found for: {csv_arg}")

    frames = []
    for p in paths:
        try:
            frames.append(_read_one_csv(p))
        except ValueError:
            # Skip unrelated CSVs when using a directory or glob
            if len(paths) > 1:
                continue
            raise

    if not frames:
        raise ValueError(f"No valid dispersion CSVs found for: {csv_arg}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("wavelength_nm").reset_index(drop=True)
    df = df.drop_duplicates(subset=["wavelength_nm"], keep="first")
    return df


def compute_dispersion(wavelength_nm: np.ndarray, x_centroid_px: np.ndarray) -> np.ndarray:
    dispersion = np.gradient(x_centroid_px, wavelength_nm)
    return dispersion


def compute_spectral_resolution(
    dispersion_px_per_nm: np.ndarray,
    fwhm_px: float,
) -> np.ndarray:
    dispersion_mag = np.abs(dispersion_px_per_nm)
    dispersion_mag = np.where(dispersion_mag == 0, np.nan, dispersion_mag)
    resolution_nm = fwhm_px / dispersion_mag
    return resolution_nm


def analyze_regions(
    df: pd.DataFrame,
    regions: Iterable[tuple[str, float, float]],
) -> list[RegionStats]:
    stats: list[RegionStats] = []
    for name, start_nm, end_nm in regions:
        mask = (df["wavelength_nm"] >= start_nm) & (df["wavelength_nm"] < end_nm)
        region_df = df.loc[mask]
        if region_df.empty:
            continue

        mean_disp = float(region_df["dispersion_px_per_nm"].abs().mean())
        mean_res = float(region_df["resolution_nm"].mean())
        span = end_nm - start_nm
        band_count = span / mean_res if mean_res > 0 else float("nan")

        stats.append(
            RegionStats(
                name=name,
                start_nm=start_nm,
                end_nm=end_nm,
                mean_dispersion=mean_disp,
                mean_resolution=mean_res,
                band_count=band_count,
            )
        )
    return stats


def simulate_gaussian_bands(
    wavelengths_nm: np.ndarray,
    x_centroid_px: np.ndarray,
    fwhm_px: float,
    max_bands: int = 40,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    if len(wavelengths_nm) == 0:
        return np.array([]), [], np.array([])

    if len(wavelengths_nm) > max_bands:
        idx = np.linspace(0, len(wavelengths_nm) - 1, max_bands, dtype=int)
        centers = x_centroid_px[idx]
    else:
        centers = x_centroid_px

    sigma = fwhm_px / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    x_min = float(np.min(x_centroid_px) - 4.0 * fwhm_px)
    x_max = float(np.max(x_centroid_px) + 4.0 * fwhm_px)
    x_axis = np.linspace(x_min, x_max, 800)

    gaussians = []
    total = np.zeros_like(x_axis)
    for center in centers:
        g = np.exp(-0.5 * ((x_axis - center) / sigma) ** 2)
        gaussians.append(g)
        total += g

    return x_axis, gaussians, total


def plot_results(
    df: pd.DataFrame,
    fwhm_px: float,
    output_dir: Path,
    max_gaussians: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(df["wavelength_nm"], df["x_centroid_px"], marker="o", linewidth=1)
    ax1.set_title("Spectral dispersion curve")
    ax1.set_xlabel("Wavelength [nm]")
    ax1.set_ylabel("Centroid X [px]")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["wavelength_nm"], df["dispersion_px_per_nm"], linewidth=1)
    ax2.set_title("Local dispersion")
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel("dx/dlambda [px/nm]")
    ax2.grid(True, alpha=0.3)

    ax3.plot(df["wavelength_nm"], df["resolution_nm"], linewidth=1)
    ax3.set_title("Spectral resolution")
    ax3.set_xlabel("Wavelength [nm]")
    ax3.set_ylabel("Delta lambda [nm]")
    ax3.grid(True, alpha=0.3)

    x_axis, gaussians, total = simulate_gaussian_bands(
        df["wavelength_nm"].to_numpy(),
        df["x_centroid_px"].to_numpy(),
        fwhm_px,
        max_bands=max_gaussians,
    )

    if len(x_axis) > 0:
        for g in gaussians:
            ax4.plot(x_axis, g, color="0.7", linewidth=0.8)
        ax4.plot(x_axis, total, color="tab:blue", linewidth=1.5, label="Sum")
    ax4.set_title("Gaussian band simulation")
    ax4.set_xlabel("Sensor X [px]")
    ax4.set_ylabel("Relative intensity")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "spectral_separability_plots.png", dpi=200)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze spectral separability from a dispersion curve CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path("analysis_dispersion") / "spectral_dispersion.csv"),
        help="Input CSV, directory, or glob with wavelength_nm and x_centroid_px columns.",
    )
    parser.add_argument(
        "--fwhm-px",
        type=float,
        default=4.0,
        help="Assumed spatial FWHM in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_dispersion"),
        help="Directory for plots and CSV output.",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=40,
        help="Max number of Gaussian bands to plot.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    df = load_dispersion_data(args.csv)

    wavelength_nm = df["wavelength_nm"].to_numpy(dtype=float)
    x_centroid_px = df["x_centroid_px"].to_numpy(dtype=float)

    dispersion = compute_dispersion(wavelength_nm, x_centroid_px)
    resolution = compute_spectral_resolution(dispersion, args.fwhm_px)

    df["dispersion_px_per_nm"] = dispersion
    df["resolution_nm"] = resolution

    regions = [
        ("Region 1", 400.0, 550.0),
        ("Region 2", 550.0, 700.0),
        ("Region 3", 700.0, 850.0),
        ("Region 4", 850.0, 1000.0),
        ("Region 5", 1000.0, 1300.0),
        ("Region 6", 1300.0, 1700.0),
    ]

    region_stats = analyze_regions(df, regions)
    total_span = 1700.0 - 400.0
    mean_res_total = float(df["resolution_nm"].mean())
    total_bands = total_span / mean_res_total if mean_res_total > 0 else float("nan")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "spectral_separability_analysis.csv"
    df.to_csv(output_csv, index=False)

    plot_results(df, args.fwhm_px, output_dir, args.max_gaussians)

    print("Spectral separability analysis")
    print(f"Data range: {wavelength_nm.min():.0f}-{wavelength_nm.max():.0f} nm")
    for stat in region_stats:
        print(
            f"{stat.name} {stat.start_nm:.0f}-{stat.end_nm:.0f} nm | "
            f"mean disp = {stat.mean_dispersion:.4f} px/nm | "
            f"mean res = {stat.mean_resolution:.4f} nm | "
            f"bands = {stat.band_count:.1f}"
        )
    print(f"Total bands (400-1700 nm): {total_bands:.1f}")
    print("Output CSV:", output_csv)
    print("Plots:", output_dir / "spectral_separability_plots.png")


if __name__ == "__main__":
    main()
