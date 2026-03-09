import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def to_2d(img: np.ndarray) -> np.ndarray:
    """Convierte a 2D si llega HxWxC."""
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def smooth2d_mean(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Suavizado 2D simple (media móvil separable) sin SciPy."""
    k = int(k)
    if k < 3:
        return img
    if k % 2 == 0:
        k += 1

    kernel = np.ones(k, dtype=np.float64) / k

    # suaviza filas
    tmp = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, img)
    # suaviza columnas
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, tmp)
    return out


def robust_preprocess(img: np.ndarray, bg_p: float = 20.0, clip_p: float = 99.7) -> np.ndarray:
    """Resta fondo (percentil bajo) y recorta outliers (percentil alto)."""
    img = img.astype(np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    bg = np.percentile(img, bg_p)
    img = img - bg
    img[img < 0] = 0.0

    hi = np.percentile(img, clip_p)
    if hi > 0:
        img = np.clip(img, 0.0, hi)

    return img


def peak_guided_centroid(
    img: np.ndarray,
    roi_hw: int = 70,
    roi_hh: int = 70,
    thr_ratio: float = 0.20,
    smooth_k: int = 5,
):
    """
    Retorna:
      cx, cy (centroide subpíxel),
      x_peak, y_peak (pico entero),
      mask_global (máscara usada),
      img_p (imagen preprocesada)
    """
    img2 = to_2d(img)
    img_p = robust_preprocess(img2)

    # Suaviza para que el pico no sea un hot pixel
    img_s = smooth2d_mean(img_p, k=smooth_k)

    # Pico (en la imagen suavizada)
    y_peak, x_peak = np.unravel_index(np.argmax(img_s), img_s.shape)

    h, w = img_s.shape
    x0 = max(0, x_peak - roi_hw)
    x1 = min(w, x_peak + roi_hw + 1)
    y0 = max(0, y_peak - roi_hh)
    y1 = min(h, y_peak + roi_hh + 1)

    roi = img_p[y0:y1, x0:x1]  # usa ROI en preprocesada (no suavizada)
    if roi.size == 0 or roi.max() <= 0:
        return float(x_peak), float(y_peak), x_peak, y_peak, np.zeros_like(img2, dtype=bool), img_p

    # Umbral LOCAL (clave)
    thr = thr_ratio * roi.max()
    roi_w = roi - thr
    roi_w[roi_w < 0] = 0.0  # pesos solo por encima del umbral

    if roi_w.sum() <= 0:
        # fallback: el pico
        mask = np.zeros_like(img2, dtype=bool)
        return float(x_peak), float(y_peak), x_peak, y_peak, mask, img_p

    yy, xx = np.indices(roi.shape)
    total = roi_w.sum()
    cx_local = (xx * roi_w).sum() / total
    cy_local = (yy * roi_w).sum() / total

    cx = x0 + cx_local
    cy = y0 + cy_local

    # Máscara global (para contorno)
    mask = np.zeros_like(img2, dtype=bool)
    mask_roi = roi > thr
    mask[y0:y1, x0:x1] = mask_roi

    return float(cx), float(cy), int(x_peak), int(y_peak), mask, img_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy_path", type=str, help="Ruta al archivo .npy")
    ap.add_argument("--thr", type=float, default=0.20, help="Umbral relativo local (0-1)")
    ap.add_argument("--roi", type=int, default=70, help="Mitad del ROI en px (cuadrado)")
    ap.add_argument("--roi_h", type=int, default=None, help="Mitad del ROI en Y (si quieres distinto)")
    ap.add_argument("--smooth", type=int, default=5, help="Kernel impar suavizado (>=3)")
    args = ap.parse_args()

    path = Path(args.npy_path)
    if not path.exists():
        print("No existe:", path)
        return

    img = np.load(path)

    img2 = to_2d(img).astype(np.float64)
    print("Archivo:", path.name)
    print("shape:", img.shape, "-> 2D:", img2.shape)
    print("min/max:", float(np.min(img2)), float(np.max(img2)))

    roi_hw = int(args.roi)
    roi_hh = int(args.roi_h) if args.roi_h is not None else roi_hw

    cx, cy, xpk, ypk, mask, img_p = peak_guided_centroid(
        img2,
        roi_hw=roi_hw,
        roi_hh=roi_hh,
        thr_ratio=float(args.thr),
        smooth_k=int(args.smooth),
    )

    print("\nPeak (suavizado):")
    print(f"  x_peak = {xpk} px")
    print(f"  y_peak = {ypk} px")

    print("\nCentroide (peak-guided, ROI local):")
    print(f"  cx = {cx:.2f} px")
    print(f"  cy = {cy:.2f} px")
    print(f"  |cx-x_peak| = {abs(cx-xpk):.2f} px")
    print(f"  |cy-y_peak| = {abs(cy-ypk):.2f} px")
    print(f"  thr_local_ratio = {args.thr}")
    print(f"  ROI = ±{roi_hw}px en X, ±{roi_hh}px en Y")

    # Visualización
    plt.figure(figsize=(7, 5))
    plt.imshow(img_p, cmap="jet")
    plt.colorbar(label="Intensidad (preprocesada)")

    # Peak y Centroide
    plt.scatter(xpk, ypk, c="white", s=80, marker="x", linewidths=2, label="Peak (suavizado)")
    plt.scatter(cx, cy, c="cyan", s=140, marker="+", linewidths=2, label="Centroide (ROI local)")

    # Contorno del área usada
    if np.any(mask):
        plt.contour(mask, levels=[0.5], colors="red", linewidths=1)

    plt.title(path.name + " | peak-guided centroid")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()