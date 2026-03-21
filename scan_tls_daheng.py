import time
import csv
from pathlib import Path
import numpy as np
import gxipy as gx
from monochromator import Monochromator


# ---------- Cámara Daheng ----------
class DahengCam:
    def __init__(self, index=1):
        self.dm = gx.DeviceManager()
        self.index = index
        self.cam = None
        self.remote = None

    def open(self):
        num, _ = self.dm.update_all_device_list()
        if num == 0:
            raise RuntimeError("No se detectó cámara Daheng.")
        self.cam = self.dm.open_device_by_index(self.index)
        self.remote = self.cam.get_remote_device_feature_control()

        # Defaults
        self.remote.get_enum_feature("UserSetSelector").set("Default")
        self.remote.get_command_feature("UserSetLoad").send_command()

        # Modo continuo (robusto)
        if self.remote.is_implemented("TriggerMode") and self.remote.is_writable("TriggerMode"):
            self.remote.get_enum_feature("TriggerMode").set("Off")

        self.cam.stream_on()
        time.sleep(0.3)

    def close(self):
        try:
            if self.cam:
                self.cam.stream_off()
        except Exception:
            pass
        try:
            if self.cam:
                self.cam.close_device()
        except Exception:
            pass

    def grab_frame(self) -> np.ndarray:
        raw = self.cam.data_stream[0].get_image()
        if raw is None:
            raise RuntimeError("get_image() devolvió None.")
        img = raw.get_numpy_array()
        if img is None:
            raise RuntimeError("No pude convertir el frame a numpy.")
        return img

    def set_exposure_time(self, exposure_time_us: float):
        """
        Establece el tiempo de exposición de la cámara en microsegundos.
        """
        # Desactiva la exposición automática
        if self.remote.is_implemented("ExposureAuto") and self.remote.is_writable("ExposureAuto"):
            self.remote.get_enum_feature("ExposureAuto").set("Off")

        # Establece el tiempo de exposición
        if self.remote.is_implemented("ExposureTime") and self.remote.is_writable("ExposureTime"):
            self.remote.get_float_feature("ExposureTime").set(float(exposure_time_us))  # Convertido a float
        print(f"Tiempo de exposición configurado a {exposure_time_us} microsegundos.")


def mean_of_n(cam: DahengCam, n=5, inter_delay_s=0.05) -> np.ndarray:
    frames = []
    for _ in range(n):
        frames.append(cam.grab_frame())
        time.sleep(inter_delay_s)
    return np.mean(np.stack(frames, axis=0).astype(np.float32), axis=0)


def extra_wait_for_switches(w: int) -> float:
    # Si quieres, ajusta estos puntos (cambios típicos de filtro/grating)
    points = [625, 690, 1100, 1580]
    for p in points:
        if abs(w - p) <= 1:
            return 2.0
    return 0.0


def run_block(mono: Monochromator,
              cam: DahengCam,
              out_dir: Path,
              block_name: str,
              start_nm: int,
              end_nm: int,
              step_nm: int,
              grating: int,
              filt: int,
              exposure_time_us: float,  # Parámetro para el tiempo de exposición
              settle_s: float = 2.0,
              n_avg: int = 5,
              grating_wait_s: float = 10.0,
              filter_wait_s: float = 5.0):

    block_dir = out_dir / block_name
    block_dir.mkdir(parents=True, exist_ok=True)

    # Configura grating/filtro al inicio del bloque
    mono.set_grating(grating)
    time.sleep(grating_wait_s)
    mono.set_filter(filt)
    time.sleep(filter_wait_s)

    # Establece el tiempo de exposición
    cam.set_exposure_time(exposure_time_us)

    meta_path = block_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([  # Títulos del archivo CSV
            "idx", "wavelength_nm", "grating", "filter",
            "n_avg", "settle_s", "filename_npy", "timestamp_unix"
        ])

        idx = 0
        for w in range(start_nm, end_nm + 1, step_nm):
            print(f"[{block_name}] λ={w} nm (G{grating}, F{filt})")

            mono.set_wavelength(w)
            time.sleep(settle_s + extra_wait_for_switches(w))

            img = mean_of_n(cam, n=n_avg) if n_avg > 1 else cam.grab_frame().astype(np.float32)

            fname = f"{idx:05d}_{w:04d}nm.npy"
            np.save(block_dir / fname, img)

            wr.writerow([idx, w, grating, filt, n_avg, settle_s, fname, time.time()])
            idx += 1


def main():
    out_dir = Path("scan_blocks_output")
    out_dir.mkdir(exist_ok=True)

    # Ajusta estos dos:
    mono = Monochromator(port="COM3")
    cam = DahengCam(index=1)

    # -------- Define tus bloques aquí --------
    # Nota: los valores grating/filtro son ejemplo: cámbialos por los tuyos reales.
    exposure_time = 50000
    blocks = [
        ("B1_0400_0410", 400, 410, 2, 1, 2, exposure_time),
        ("B2_0411_0450", 411, 450, 2, 1, 2, exposure_time),
        ("B3_0451_0500", 451, 500, 2, 1, 2, exposure_time),
        ("B4_0501_0700", 501, 700, 2, 1, 2, exposure_time),
        ("B5_0701_1000", 701, 1000, 2, 2, 3, exposure_time),  # 40ms para el segundo bloque
        ("B6_1001_1300", 1001, 1300, 2, 2, 4, exposure_time),  # 60ms para el tercer bloque
        ("B7_1301_1700", 1301, 1700, 2, 2, 4, exposure_time),  # 30ms para el cuarto bloque
        # ("B10_00701_1000_prueba", 701, 800, 2, 2, 3, 50000),
        # ("B11_00701_1000_prueba", 801, 900, 2, 2, 4, 50000)
    ]

    # Parámetros de captura:
    settle_s = 2.0     # como tu MATLAB
    n_avg = 5          # promedio de 5 frames
    grating_wait_s = 10.0
    filter_wait_s = 5.0

    try:
        cam.open()

        # Prueba corta (descomenta si quieres test rápido):
        # run_block(mono, cam, out_dir, "TEST_0400_0410", 400, 410, 2, grating=1, filt=2, settle_s=2.0, n_avg=3)

        for (name, s, e, step, g, flt, exposure_time_us) in blocks:
            run_block(
                mono, cam, out_dir,
                block_name=name,
                start_nm=s, end_nm=e, step_nm=step,
                grating=g, filt=flt,
                exposure_time_us=exposure_time_us,  # Le pasas el tiempo de exposición por bloque
                settle_s=settle_s,
                n_avg=n_avg,
                grating_wait_s=grating_wait_s,
                filter_wait_s=filter_wait_s
            )

        print("Listo ✅ Revisa:", out_dir)

    finally:
        cam.close()
        mono.close()


if __name__ == "__main__":
    main()