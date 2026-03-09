from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


# --------- Utilidades ---------
def parse_wavelength_from_filename(path: Path) -> int:
    m = re.search(r"_(\d+)nm", path.stem)
    if not m:
        raise ValueError(f"No pude extraer λ del nombre: {path.name}")
    return int(m.group(1))


def list_all_npy(scan_root: Path) -> list[Path]:
    files = []
    for p in scan_root.rglob("*.npy"):
        if re.search(r"_(\d+)nm$", p.stem):
            files.append(p)
    return sorted(files)


def smooth_1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    """Suavizado simple por media móvil (k impar)."""
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
) -> tuple[float, int, np.ndarray]:
    """
    Devuelve:
      x_centroid_local (subpíxel),
      x_peak (entero),
      profile_x (para debug)
    """

    # --- a 2D ---
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float64)

    h, w = img.shape[:2]

    # --- banda central en Y ---
    y0 = max(0, h // 2 - crop_half_height)
    y1 = min(h, h // 2 + crop_half_height)
    band = img[y0:y1, :].copy()

    # --- background subtraction (robusto) ---
    bg = np.percentile(band, background_percentile)
    band = band - bg
    band[band < 0] = 0.0

    # --- clip para quitar hot pixels extremos ---
    hi = np.percentile(band, clip_percentile)
    if hi > 0:
        band = np.clip(band, 0.0, hi)

    # --- perfil en X ---
    profile_x = band.sum(axis=0)

    # --- suaviza perfil para que el pico sea estable ---
    prof_s = smooth_1d(profile_x, k=smooth_k)

    # --- busca pico (con tracking opcional) ---
    if prev_x is not None:
        lo = int(max(0, round(prev_x) - track_max_jump))
        hiw = int(min(w, round(prev_x) + track_max_jump + 1))
        if hiw - lo >= 5:
            x_peak = int(lo + np.argmax(prof_s[lo:hiw]))
        else:
            x_peak = int(np.argmax(prof_s))
    else:
        x_peak = int(np.argmax(prof_s))

    # --- ventana local alrededor del pico ---
    x0 = max(0, x_peak - local_half_width)
    x1 = min(w, x_peak + local_half_width + 1)

    local = prof_s[x0:x1]
    if local.size == 0:
        return float(x_peak), x_peak, profile_x

    local_max = local.max()
    if local_max <= 0:
        return float(x_peak), x_peak, profile_x

    # --- umbral local ---
    thresh = threshold_ratio_local * local_max
    mask = local > thresh
    if not np.any(mask):
        return float(x_peak), x_peak, profile_x

    xs = np.arange(x0, x1, dtype=np.float64)
    weights = local * mask

    x_centroid = (xs * weights).sum() / weights.sum()
    return float(x_centroid), x_peak, profile_x


# --------- Programa principal ---------
def main():
    scan_root = Path("scan_blocks_output")
    out_dir = Path("analysis_dispersion")
    out_dir.mkdir(exist_ok=True)

    debug_dir = out_dir / "debug_centroid"
    debug_dir.mkdir(exist_ok=True)

    files = list_all_npy(scan_root)
    if not files:
        raise SystemExit(f"No encontré .npy con patrón _###nm.npy dentro de {scan_root}")

    # orden por lambda
    data_raw = []
    for fp in files:
        lam = parse_wavelength_from_filename(fp)
        data_raw.append((lam, fp))
    data_raw.sort(key=lambda t: t[0])

    data = []
    prev_x = None

    for lam, fp in data_raw:
        img = np.load(fp)

        x_centroid, x_peak, profile_x = compute_x_peak_guided_centroid(
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

        # si pega un salto absurdo, resetea tracking y recalcula libre
        if prev_x is not None and abs(x_centroid - prev_x) > 250:
            x_centroid, x_peak, profile_x = compute_x_peak_guided_centroid(
                img,
                prev_x=None
            )

        # debug si pico y centroide difieren mucho (típico cuando algo está mal)
        if abs(x_centroid - x_peak) > 12:
            # imagen rápida de la banda + marcas (sin depender de libs extra)
            if img.ndim == 3:
                img2 = img.mean(axis=2)
            else:
                img2 = img
            h, w = img2.shape
            y0 = max(0, h // 2 - 30)
            y1 = min(h, h // 2 + 30)
            band = img2[y0:y1, :]

            plt.figure(figsize=(7, 3))
            plt.imshow(band, aspect="auto")
            plt.axvline(x_peak, linestyle="--")
            plt.axvline(x_centroid, linestyle="-")
            plt.title(f"λ={lam}nm | peak={x_peak} | centroid={x_centroid:.2f}")
            plt.tight_layout()
            plt.savefig(debug_dir / f"debug_{lam:04d}nm.png", dpi=150)
            plt.close()

        data.append((lam, x_centroid, x_peak, fp))
        prev_x = x_centroid

    lambdas = np.array([d[0] for d in data], dtype=np.int32)
    x_centroids = np.array([d[1] for d in data], dtype=np.float64)
    x_peaks = np.array([d[2] for d in data], dtype=np.int32)

    # referencia
    x0 = x_centroids[0]
    displacement_px = x_centroids - x0

    # --- CSV ---
    csv_path = out_dir / "spectral_dispersion.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("wavelength_nm,x_centroid_px,x_peak_px,displacement_px,filepath\n")
        for (lam, xc, xp, fp), dpx in zip(data, displacement_px):
            f.write(f"{lam},{xc:.6f},{xp},{dpx:.6f},{fp.as_posix()}\n")

    # --- plot ---
    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, displacement_px, marker="o", linewidth=1)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Accumulated displacement [px]")
    plt.title("Spectral dispersion characterization (peak-guided centroid)")
    plt.grid(True)
    plt.tight_layout()

    fig_path = out_dir / "spectral_dispersion.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()

    print("Listo ✅")
    print("CSV :", csv_path)
    print("Figura :", fig_path)
    print("Debug (si hubo casos raros):", debug_dir)


if __name__ == "__main__":
    main()