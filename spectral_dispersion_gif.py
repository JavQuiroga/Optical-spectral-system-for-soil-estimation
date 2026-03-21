from __future__ import annotations

from io import BytesIO
from pathlib import Path
import argparse
import re

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def parse_wavelength_from_filename(path: Path) -> int:
    match = re.search(r"_(\d+)nm", path.stem)
    if not match:
        raise ValueError(f"No pude extraer la longitud de onda del nombre: {path.name}")
    return int(match.group(1))


def list_all_npy(scan_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in scan_root.rglob("*.npy"):
        if re.search(r"_(\d+)nm$", path.stem):
            files.append(path)
    return sorted(files)


def smooth_1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    k = int(k)
    if k < 3:
        return x
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(x, kernel, mode="same")


def compute_x_peak_guided_centroid(
    img: np.ndarray,
    crop_half_height: int = 30,
    background_percentile: float = 20.0,
    clip_percentile: float = 99.7,
    smooth_k: int = 9,
    threshold_ratio_local: float = 0.2,
    local_half_width: int = 60,
    prev_x: float | None = None,
    track_max_jump: int = 120,
) -> tuple[float, int, np.ndarray, np.ndarray, tuple[int, int]]:
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float64)

    h, w = img.shape[:2]
    y0 = max(0, h // 2 - crop_half_height)
    y1 = min(h, h // 2 + crop_half_height)
    band = img[y0:y1, :].copy()

    bg = np.percentile(band, background_percentile)
    band = band - bg
    band[band < 0] = 0.0

    hi = np.percentile(band, clip_percentile)
    if hi > 0:
        band = np.clip(band, 0.0, hi)

    profile_x = band.sum(axis=0)
    prof_s = smooth_1d(profile_x, k=smooth_k)

    if prev_x is not None:
        lo = int(max(0, round(prev_x) - track_max_jump))
        hiw = int(min(w, round(prev_x) + track_max_jump + 1))
        if hiw - lo >= 5:
            x_peak = int(lo + np.argmax(prof_s[lo:hiw]))
        else:
            x_peak = int(np.argmax(prof_s))
    else:
        x_peak = int(np.argmax(prof_s))

    x0 = max(0, x_peak - local_half_width)
    x1 = min(w, x_peak + local_half_width + 1)
    local = prof_s[x0:x1]
    if local.size == 0:
        return float(x_peak), x_peak, profile_x, img, (y0, y1)

    local_max = local.max()
    if local_max <= 0:
        return float(x_peak), x_peak, profile_x, img, (y0, y1)

    thresh = threshold_ratio_local * local_max
    mask = local > thresh
    if not np.any(mask):
        return float(x_peak), x_peak, profile_x, img, (y0, y1)

    xs = np.arange(x0, x1, dtype=np.float64)
    weights = local * mask
    x_centroid = (xs * weights).sum() / weights.sum()
    return float(x_centroid), x_peak, profile_x, img, (y0, y1)


def collect_dispersion_data(scan_root: Path) -> list[dict]:
    files = list_all_npy(scan_root)
    if not files:
        raise SystemExit(f"No encontre archivos .npy con patron _###nm.npy dentro de {scan_root}")

    data_raw: list[tuple[int, Path]] = []
    for file_path in files:
        wavelength_nm = parse_wavelength_from_filename(file_path)
        data_raw.append((wavelength_nm, file_path))
    data_raw.sort(key=lambda item: item[0])

    results: list[dict] = []
    prev_x: float | None = None

    for wavelength_nm, file_path in data_raw:
        img = np.load(file_path)
        x_centroid, x_peak, profile_x, image_2d, band_limits = compute_x_peak_guided_centroid(
            img,
            crop_half_height=30,
            background_percentile=20.0,
            clip_percentile=99.7,
            smooth_k=9,
            threshold_ratio_local=0.20,
            local_half_width=60,
            prev_x=prev_x,
            track_max_jump=140,
        )

        if prev_x is not None and abs(x_centroid - prev_x) > 250:
            x_centroid, x_peak, profile_x, image_2d, band_limits = compute_x_peak_guided_centroid(
                img,
                prev_x=None,
            )

        results.append(
            {
                "wavelength_nm": wavelength_nm,
                "x_centroid_px": float(x_centroid),
                "x_peak_px": int(x_peak),
                "profile_x": profile_x,
                "image_2d": image_2d,
                "band_limits": band_limits,
                "file_path": file_path,
            }
        )
        prev_x = x_centroid

    x0 = results[0]["x_centroid_px"]
    for item in results:
        item["displacement_px"] = item["x_centroid_px"] - x0
    return results


def render_frame(
    entry: dict,
    wavelengths_nm: np.ndarray,
    displacements_px: np.ndarray,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    fig, (ax_img, ax_plot) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    image_2d = entry["image_2d"]
    y0, y1 = entry["band_limits"]
    vmin = float(np.percentile(image_2d, 5))
    vmax = float(np.percentile(image_2d, 99.7))
    if vmax <= vmin:
        vmax = vmin + 1.0

    ax_img.imshow(image_2d, cmap="inferno", aspect="auto", vmin=vmin, vmax=vmax)
    ax_img.axvline(entry["x_peak_px"], color="cyan", linestyle="--", linewidth=1.2, label="Pico")
    ax_img.axvline(entry["x_centroid_px"], color="lime", linestyle="-", linewidth=1.5, label="Centroide")
    ax_img.axhline(y0, color="white", linestyle=":", linewidth=1.0, label="Banda analizada")
    ax_img.axhline(y1, color="white", linestyle=":", linewidth=1.0)
    ax_img.set_title(
        f"Imagen {frame_idx + 1}/{total_frames}\n"
        f"{entry['wavelength_nm']} nm | desplazamiento = {entry['displacement_px']:.2f} px"
    )
    ax_img.set_xlabel("Pixel X")
    ax_img.set_ylabel("Pixel Y")
    ax_img.legend(loc="upper right")

    ax_plot.plot(wavelengths_nm, displacements_px, color="0.8", linewidth=1.0, zorder=1)
    ax_plot.plot(
        wavelengths_nm[: frame_idx + 1],
        displacements_px[: frame_idx + 1],
        color="tab:blue",
        marker="o",
        markersize=4,
        linewidth=1.5,
        zorder=2,
    )
    ax_plot.scatter(
        wavelengths_nm[frame_idx],
        displacements_px[frame_idx],
        color="crimson",
        s=45,
        zorder=3,
    )
    ax_plot.set_title("Dispersion espectral acumulada")
    ax_plot.set_xlabel("Longitud de onda [nm]")
    ax_plot.set_ylabel("Desplazamiento [px]")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())

    y_margin = max(1.0, 0.05 * float(np.ptp(displacements_px)) if len(displacements_px) > 1 else 1.0)
    ax_plot.set_ylim(displacements_px.min() - y_margin, displacements_px.max() + y_margin)

    fig.suptitle(Path(entry["file_path"]).parent.name, fontsize=12)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


def save_csv(results: list[dict], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8") as file:
        file.write("wavelength_nm,x_centroid_px,x_peak_px,displacement_px,filepath\n")
        for entry in results:
            file.write(
                f"{entry['wavelength_nm']},"
                f"{entry['x_centroid_px']:.6f},"
                f"{entry['x_peak_px']},"
                f"{entry['displacement_px']:.6f},"
                f"{entry['file_path'].as_posix()}\n"
            )


def build_gif(
    scan_root: Path,
    output_gif: Path,
    fps: float,
    max_frames: int | None = None,
) -> tuple[Path, Path]:
    results = collect_dispersion_data(scan_root)
    if max_frames is not None:
        results = results[:max_frames]

    wavelengths_nm = np.array([item["wavelength_nm"] for item in results], dtype=np.float64)
    displacements_px = np.array([item["displacement_px"] for item in results], dtype=np.float64)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_gif.with_suffix(".csv")
    save_csv(results, output_csv)

    frames = []
    total_frames = len(results)
    for idx, entry in enumerate(results):
        frames.append(render_frame(entry, wavelengths_nm, displacements_px, idx, total_frames))

    duration_s = 1.0 / fps if fps > 0 else 0.2
    imageio.mimsave(output_gif, frames, duration=duration_s, loop=0)
    return output_gif, output_csv


def find_scan_dirs(scan_root: Path) -> list[Path]:
    if any(scan_root.glob("*.npy")):
        return [scan_root]

    scan_dirs = []
    for path in sorted(scan_root.iterdir()):
        if path.is_dir() and any(path.glob("*.npy")):
            scan_dirs.append(path)
    return scan_dirs


def build_output_path(base_output: Path, scan_dir: Path, multiple_dirs: bool) -> Path:
    if base_output.suffix.lower() == ".gif" and not multiple_dirs:
        return base_output

    stem = f"{scan_dir.name}_spectral_dispersion_progress"
    return base_output / f"{stem}.gif"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un GIF que recorre las imagenes de un barrido y, en paralelo, "
            "va construyendo la grafica de dispersion espectral punto a punto."
        )
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=Path("scan_blocks_output"),
        help="Carpeta raiz con bloques o carpeta puntual con archivos .npy del barrido.",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=Path("analysis_dispersion") / "gifs",
        help="GIF de salida o carpeta de salida si se procesan varios bloques.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Cuadros por segundo del GIF.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limita la cantidad de frames para pruebas rapidas.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    scan_dirs = find_scan_dirs(args.scan_root)
    if not scan_dirs:
        raise SystemExit(f"No encontre carpetas con .npy dentro de {args.scan_root}")

    multiple_dirs = len(scan_dirs) > 1
    outputs = []
    for scan_dir in scan_dirs:
        output_gif = build_output_path(args.output_gif, scan_dir, multiple_dirs)
        gif_path, csv_path = build_gif(
            scan_root=scan_dir,
            output_gif=output_gif,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        outputs.append((gif_path, csv_path))

    for gif_path, csv_path in outputs:
        print("GIF generado:", gif_path)
        print("CSV generado:", csv_path)


if __name__ == "__main__":
    main()
