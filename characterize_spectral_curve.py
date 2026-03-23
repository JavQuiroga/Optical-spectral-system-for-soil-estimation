from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

MPL_CONFIG_DIR = Path(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt

from spectral_peak_utils import compute_vertical_profile, ensure_2d_image, select_peak_with_plateau


BLOCK_CONFIG = {
    "B1": {"start_nm": 400, "end_nm": 410, "step_nm": 2},
    "B2": {"start_nm": 411, "end_nm": 450, "step_nm": 2},
    "B3": {"start_nm": 451, "end_nm": 500, "step_nm": 2},
    "B4": {"start_nm": 501, "end_nm": 700, "step_nm": 2},
    "B5": {"start_nm": 701, "end_nm": 1000, "step_nm": 2},
    "B6": {"start_nm": 1001, "end_nm": 1300, "step_nm": 2},
    "B7": {"start_nm": 1301, "end_nm": 1700, "step_nm": 2},
}

BLOCK_DIR_PATTERN = re.compile(r"^(B\d+)_")
WAVELENGTH_PATTERN = re.compile(r"(?<!\d)(\d{3,4})\s*nm(?!\d)", re.IGNORECASE)
LEADING_INDEX_PATTERN = re.compile(r"^(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye una curva de caracterizacion espectral a partir de archivos .npy."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("scan_blocks_output/Try_diferentes_times"),
        help="Carpeta raiz que contiene carpetas por tiempo de exposicion.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scan_blocks_output") / "spectral_characterization.csv",
        help="Ruta del CSV consolidado de salida.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("scan_blocks_output") / "spectral_characterization_plots",
        help="Carpeta donde se guardaran las graficas por tiempo de exposicion.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra las graficas en pantalla ademas de guardarlas.",
    )
    return parser.parse_args()


def is_block_dir(path: Path) -> bool:
    return path.is_dir() and BLOCK_DIR_PATTERN.match(path.name) is not None


def discover_exposure_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta raiz: {root_dir}")

    direct_block_dirs = [path for path in root_dir.iterdir() if is_block_dir(path)]
    if direct_block_dirs:
        return [root_dir]

    exposure_dirs = []
    for path in sorted(root_dir.iterdir()):
        if not path.is_dir():
            continue
        if any(is_block_dir(child) for child in path.iterdir()):
            exposure_dirs.append(path)
    return exposure_dirs


def get_block_name(block_dir: Path) -> str:
    match = BLOCK_DIR_PATTERN.match(block_dir.name)
    if match is None:
        raise ValueError(f"No se pudo identificar el bloque en: {block_dir}")
    return match.group(1)


def load_block_metadata(block_dir: Path) -> pd.DataFrame | None:
    metadata_path = block_dir / "metadata.csv"
    if not metadata_path.exists():
        return None

    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        return None

    metadata.columns = [str(column).strip() for column in metadata.columns]
    return metadata


def build_metadata_lookup(metadata: pd.DataFrame | None) -> dict[str, float]:
    if metadata is None:
        return {}

    filename_column = None
    wavelength_column = None
    for column in metadata.columns:
        lowered = column.lower()
        if lowered in {"filename_npy", "filename", "file", "archivo"}:
            filename_column = column
        if lowered in {"wavelength_nm", "wavelength", "lambda_nm", "nm"}:
            wavelength_column = column

    if filename_column is None or wavelength_column is None:
        return {}

    lookup: dict[str, float] = {}
    for _, row in metadata.iterrows():
        filename = str(row[filename_column]).strip()
        if not filename:
            continue
        wavelength = pd.to_numeric(row[wavelength_column], errors="coerce")
        if pd.isna(wavelength):
            continue
        lookup[filename] = float(wavelength)
    return lookup


def extract_wavelength_from_filename(filename: str) -> float | None:
    match = WAVELENGTH_PATTERN.search(filename)
    if match is None:
        return None
    return float(match.group(1))


def estimate_order_index(npy_path: Path, fallback_index: int) -> int:
    match = LEADING_INDEX_PATTERN.match(npy_path.stem)
    if match is not None:
        return int(match.group(1))
    return fallback_index


def wavelength_from_block_and_order(block: str, order_index: int) -> float:
    if block not in BLOCK_CONFIG:
        raise KeyError(f"No existe configuracion para el bloque {block}")
    block_cfg = BLOCK_CONFIG[block]
    wavelength = block_cfg["start_nm"] + block_cfg["step_nm"] * order_index
    return float(wavelength)


def resolve_wavelength_nm(
    npy_path: Path,
    block: str,
    fallback_index: int,
    metadata_lookup: dict[str, float],
) -> float:
    wavelength = extract_wavelength_from_filename(npy_path.name)
    if wavelength is not None:
        return wavelength

    metadata_value = metadata_lookup.get(npy_path.name)
    if metadata_value is not None:
        return metadata_value

    order_index = estimate_order_index(npy_path, fallback_index)
    return wavelength_from_block_and_order(block, order_index)


def process_npy_file(
    npy_path: Path,
    exposure_folder: str,
    block: str,
    fallback_index: int,
    metadata_lookup: dict[str, float],
) -> dict[str, object]:
    image = ensure_2d_image(np.load(npy_path), npy_path)
    profile_x = compute_vertical_profile(image)
    peak = select_peak_with_plateau(profile_x)
    wavelength_nm = resolve_wavelength_nm(npy_path, block, fallback_index, metadata_lookup)

    return {
        "exposure_folder": exposure_folder,
        "block": block,
        "filename": npy_path.name,
        "wavelength_nm": wavelength_nm,
        "selected_x": peak.selected_x,
        "max_energy": peak.max_energy,
        "plateau_start_x": peak.plateau_start_x,
        "plateau_end_x": peak.plateau_end_x,
        "plateau_width": peak.plateau_width,
    }


def process_block(block_dir: Path, exposure_folder: str) -> list[dict[str, object]]:
    block = get_block_name(block_dir)
    metadata_lookup = build_metadata_lookup(load_block_metadata(block_dir))
    npy_files = sorted(block_dir.glob("*.npy"))

    rows = []
    for fallback_index, npy_path in enumerate(npy_files):
        rows.append(
            process_npy_file(
                npy_path=npy_path,
                exposure_folder=exposure_folder,
                block=block,
                fallback_index=fallback_index,
                metadata_lookup=metadata_lookup,
            )
        )
    return rows


def process_exposure_dir(exposure_dir: Path) -> list[dict[str, object]]:
    exposure_folder = exposure_dir.name
    rows: list[dict[str, object]] = []
    block_dirs = sorted(path for path in exposure_dir.iterdir() if is_block_dir(path))
    for block_dir in block_dirs:
        rows.extend(process_block(block_dir, exposure_folder))
    return rows


def build_results_dataframe(root_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exposure_dir in discover_exposure_dirs(root_dir):
        rows.extend(process_exposure_dir(exposure_dir))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(
        by=["exposure_folder", "wavelength_nm", "block", "filename"],
        kind="stable",
    ).reset_index(drop=True)


def save_plot_for_exposure(
    exposure_df: pd.DataFrame,
    exposure_folder: str,
    plots_dir: Path,
    show_plot: bool,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ordered_df = exposure_df.sort_values("wavelength_nm", kind="stable")
    ax.plot(
        ordered_df["wavelength_nm"],
        ordered_df["selected_x"],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4,
    )
    ax.set_title(f"Caracterizacion espectral - {exposure_folder}")
    ax.set_xlabel("Longitud de onda (nm)")
    ax.set_ylabel("Posicion x seleccionada (px)")
    ax.grid(True, alpha=0.3)

    output_path = plots_dir / f"{exposure_folder}_wavelength_vs_x.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)
    return output_path


def save_outputs(df: pd.DataFrame, output_csv: Path, plots_dir: Path, show_plots: bool) -> list[Path]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    plot_paths: list[Path] = []
    if df.empty:
        return plot_paths

    for exposure_folder, exposure_df in df.groupby("exposure_folder", sort=True):
        plot_paths.append(
            save_plot_for_exposure(exposure_df, str(exposure_folder), plots_dir, show_plots)
        )

    return plot_paths


def main() -> None:
    args = parse_args()
    df = build_results_dataframe(args.root)

    if df.empty:
        print(f"No se encontraron archivos .npy procesables dentro de {args.root}")
        return

    plot_paths = save_outputs(df, args.output_csv, args.plots_dir, args.show)

    print(f"CSV guardado en: {args.output_csv}")
    print(f"Archivos procesados: {len(df)}")
    print(f"Graficas generadas: {len(plot_paths)}")


if __name__ == "__main__":
    main()
