import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def to_2d(img: np.ndarray) -> np.ndarray:
    """Convierte a 2D si llega HxWxC."""
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Abrir y visualizar imágenes guardadas en .npy")
    ap.add_argument("npy_path", type=str, help="Ruta al archivo .npy")
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

    plt.figure(figsize=(7, 5))
    plt.imshow(img2, cmap="jet")
    plt.colorbar(label="Intensidad")
    plt.title(path.name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
