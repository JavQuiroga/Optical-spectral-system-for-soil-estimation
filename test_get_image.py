import gxipy as gx
import numpy as np
from pathlib import Path

def save_png_uint8(img: np.ndarray, out_path: Path):
    """
    Convierte la imagen a 8-bit SOLO para visualizar (PNG),
    escalando por min/max. No sirve para análisis radiométrico.
    """
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        img8 = np.zeros_like(img, dtype=np.uint8)
    else:
        img8 = ((img - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)

    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(out_path), img8)
        print("PNG guardado:", out_path)
    except Exception:
        # fallback si no tienes imageio
        try:
            import cv2
            cv2.imwrite(str(out_path), img8)
            print("PNG guardado (cv2):", out_path)
        except Exception as e:
            print("No pude guardar PNG. Instala imageio o opencv-python.")
            print("Error:", e)

def set_exposure_time(cam, exposure_time_us: float):
    """
    Establece el tiempo de exposición de la cámara en microsegundos.
    """
    remote = cam.get_remote_device_feature_control()
    
    # Desactiva la exposición automática
    if remote.is_implemented("ExposureAuto") and remote.is_writable("ExposureAuto"):
        remote.get_enum_feature("ExposureAuto").set("Off")
    
    # Establece el tiempo de exposición como float
    if remote.is_implemented("ExposureTime") and remote.is_writable("ExposureTime"):
        remote.get_float_feature("ExposureTime").set(float(exposure_time_us))  # Convertido a float
    print(f"Tiempo de exposición configurado a {exposure_time_us} microsegundos.")

def main(exposure_time_us: float = 50000):  # Default: 50ms
    out_dir = Path("test_capture")
    out_dir.mkdir(exist_ok=True)

    # Inicializa el gestor de dispositivos y obtiene la cámara
    dm = gx.DeviceManager()
    num, info = dm.update_all_device_list()
    print("Cámaras detectadas:", num)
    if num == 0:
        raise SystemExit("No detecté cámara. Revisa Galaxy Viewer / red / firewall.")

    cam = dm.open_device_by_index(1)
    cam.stream_on()

    # Configurar tiempo de exposición
    set_exposure_time(cam, exposure_time_us)

    # Captura la imagen
    print("Capturando imagen...")
    raw = cam.data_stream[0].get_image()
    if raw is None:
        cam.stream_off()
        cam.close_device()
        raise RuntimeError("get_image() devolvió None (no llegó frame).")

    img = raw.get_numpy_array()
    if img is None:
        cam.stream_off()
        cam.close_device()
        raise RuntimeError("No pude convertir la imagen a numpy array.")

    print("Frame capturado:")
    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
    print("Min/Max:", img.min(), img.max())

    # Guarda NPY (sin perder info)
    npy_path = out_dir / "frame_raw.npy"
    np.save(npy_path, img)
    print("NPY guardado:", npy_path)

    # Guarda PNG (solo visualización)
    png_path = out_dir / "frame_preview.png"
    save_png_uint8(img, png_path)

    cam.stream_off()
    cam.close_device()
    print("OK captura y guardado.")

if __name__ == "__main__":
    # Llama a main() pasando el tiempo de exposición que desees (en microsegundos)
    # Ejemplo: para 50ms (50,000 microsegundos)
    main(exposure_time_us=10000)