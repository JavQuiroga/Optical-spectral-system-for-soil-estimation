'''
Created by: David Santiago Morales Norato 2022.
'''

import serial # Pyserial
import io
import time
# LIST SERIAL PORTS python -m serial.tools.list_ports
class Monochromator():

    def __init__(self, port):
        self.port = port
        self.serial = serial.Serial(port = port, timeout = 3)
        self.sio = io.TextIOWrapper(io.BufferedRWPair(self.serial, self.serial))
        self.text_set_wave = "GOWAVE "
        self.text_set_filter = "FILTER "

    
    def send_string(self, string: str):
        self.sio.write(string)
        self.sio.flush()

    def set_grating(self, n_grating: int):
        #selecciona el grating 1 o 2
        cmd = "GRAT " + str(n_grating) + "\n"
        self.send_string(cmd)


    def set_wavelength(self, wavelength: int):
        string = self.text_set_wave + str(wavelength) + "\n"
        self.send_string(string)

    def set_filter(self, n_filter: int):
        string = self.text_set_filter + str(n_filter) + "\n"
        self.send_string(string)

    def close(self):
        self.serial.close()

if __name__ == "__main__":

    mono = Monochromator(port = "COM3")
    mono.set_wavelength(640)
    mono.set_filter(0)
    time.sleep(5)
    mono.close()
    #mono.close()