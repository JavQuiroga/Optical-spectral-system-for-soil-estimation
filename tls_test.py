import time
from monochromator import Monochromator

m = Monochromator(port="COM3")   # cambia COM si aplica

m.set_filter(2)
time.sleep(5)

m.set_grating(1)   # si ya le agregaste set_grating
time.sleep(5)

m.set_wavelength(600)
time.sleep(5)

m.close()
print("OK TLS")