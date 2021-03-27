import numpy as np
import matplotlib.pyplot as plt
from . import functions as func
from . import fit
from . import read

# file_name requires
class resonance:
    def __init__(self, file_name, linear_background = True):
        self.current = read.current(file_name)
        self.file_name = file_name
        freq, X, Y, R, phi = read.file(file_name, 0, 1, 3, 5, 7)
        self.freq = freq
        self.sig = X + 1j*Y
        self.R = R

        if linear_background:
            self.model = func.complex_lorentz_lin_back
            self.popt, self.pcov = fit.complex_lorentz_lin_back(freq, self.sig)
        else:
            self.model = func.complex_lorentz
            self.popt, self.pcov = fit.complex_lorentz(freq, self.sig)

    def get_current(self):
        return self.current

    # private method returns i-th fit parameter and it's error
    def __get(self, i):
        param = self.popt[i]
        err = np.sqrt(self.pcov[i][i])
        return param, err

    # returns f0 and it's error
    def get_f0(self):
        return self.__get(0)

    # returns gamma and it's error
    def get_gamma(self):
        return self.__get(2)

    def plot_real(self):
        plt.plot(self.freq, self.sig.real)
        plt.plot(self.freq, self.model(self.freq, *self.popt).real)
        plt.xlabel("Frequency [Hz]")
        plt.show()

    def plot_imag(self):
        plt.plot(self.freq, self.sig.imag)
        plt.plot(self.freq, self.model(self.freq, *self.popt).imag)
        plt.xlabel("Frequency [Hz]")
        plt.show()

    def plot_abs(self):
        plt.plot(self.freq, np.abs(self.sig))
        plt.plot(self.freq, np.abs(self.model(self.freq, *self.popt)))
        plt.xlabel("Frequency [Hz]")
        plt.show()

class FID(resonance):
    def __init__(self, file_name, linear_background = False, a = 0, b = -1, fmin = 0, fmax = 0 ):
        self.current = read.current(file_name)
        self.file_name = file_name
        time, time_sig = read.file(file_name, 0, 1)
        self.time = time
        self.time_sig = time_sig
        freq, fft = read.comp_fft(time, time_sig, a = a, b = b, fmin = fmin, fmax = fmax)
        self.freq = freq
        self.sig = fft

        if linear_background:
            self.model = func.complex_lorentz_lin_back
            self.popt, self.pcov = fit.complex_lorentz_lin_back(freq, self.sig)
        else:
            self.model = func.complex_lorentz
            self.popt, self.pcov = fit.complex_lorentz(freq, self.sig)

    def plot_FID(self):
        plt.plot(self.time, self.time_sig)
        plt.xlabel("Time [s]")
        plt.ylabel("Signal")
        plt.show()
