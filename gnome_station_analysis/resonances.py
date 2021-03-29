import numpy as np
import matplotlib.pyplot as plt
from . import functions as func
from . import fit
from . import read

# file_name requires
class resonance:
    """Fit to the measurement data is run automatically. Fit parameters are available.

    Parameters
    ----------
    file_name : string
        path to file with measured resonance
    linear_background : bool
        when true fit.complex_lorentz_lin_back() is used, when false - fit.complex_lorentz()
    """

    def __init__(self, file_name, linear_background = True):
        """Fit to the measurement data is run automatically. Fit parameters are available.

        Parameters
        ----------
        file_name : string
            path to file with measured resonance
        linear_background : bool
            when true fit.complex_lorentz_lin_back() is used, when false - fit.complex_lorentz()
        """
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
        """Gets current from the name of the analyzed file.

        Returns
        -------
        current : float
        """
        return self.current

    # private method returns i-th fit parameter
    def get_param(self, i):
        """Method returns i-th fit parameter

        Parameters
        ----------
        i : int
            number of the fit parameter

        Returns
        -------
        param : float
            fit parameter

        Notes
        -----
        Please see fit.complex_lorentz() or fit.complex_lorentz_lin_back() methods to obtain the order of parameters

        """
        param = self.popt[i]
        return param

    # private method returns i-th fit parameter's error
    def get_err(self, i):
        """Method returns i-th fit parameter's error

        Parameters
        ----------
        i : int
            number of the fit parameter

        Returns
        -------
        err : float
            square root of the diagonal element of covariance matrix pcov[i][i]

        Notes
        -----
        Please see fit.complex_lorentz() or fit.complex_lorentz_lin_back() methods to obtain the order of parameters

        """
        err = np.sqrt(self.pcov[i][i])
        return err

    # returns f0
    def get_f0(self):
        """

        Returns
        -------
        f0 : float
            resonant frequency in Hz
        """
        return self.get_param(0)

    # returns f0 error
    def get_f0_err(self):
        """

        Returns
        -------
        f0_err : float
            resonant frequency error in Hz
        """
        return self.get_err(0)

    # returns gamma
    def get_gamma(self):
        """

        Returns
        -------
        gamma : float
            resonance width in Hz
        """
        return self.get_param(2)

    # returns gamma error
    def get_gamma_err(self):
        """

        Returns
        -------
        gamma_err : float
            resonance width error in Hz
        """
        return self.get_err(2)

    def plot_real(self):
        """Plots real part of the resonance (measurement and fit function)

        """
        plt.plot(self.freq, self.sig.real)
        plt.plot(self.freq, self.model(self.freq, *self.popt).real)
        plt.xlabel("Frequency [Hz]")
        plt.show()

    def plot_imag(self):
        """Plots imaginary part of the resonance (measurement and fit function)

        """
        plt.plot(self.freq, self.sig.imag)
        plt.plot(self.freq, self.model(self.freq, *self.popt).imag)
        plt.xlabel("Frequency [Hz]")
        plt.show()

    def plot_abs(self):
        """Plots absolute value of the resonance (measurement and fit function)

        """
        plt.plot(self.freq, np.abs(self.sig))
        plt.plot(self.freq, np.abs(self.model(self.freq, *self.popt)))
        plt.xlabel("Frequency [Hz]")
        plt.show()

class FID(resonance):
    def __init__(self, file_name, linear_background = False, a = 0, b = -1, fmin = 0, fmax = 0 ):
        """Converting time signal (FID) into frequency domain complex lorentzian resonance. Fitting complex lorentzian function.

        Parameters
        ----------
        file_name : string
            path to file with measured resonance
        linear_background : bool
            when true fit.complex_lorentz_lin_back() is used, when false - fit.complex_lorentz()
        a : int
            element of the time array defining the beginning of the interval that is going to be used in FFT (optional)
        b : int
            element of the time array defining the end of the interval that is going to be used in FFT (optional)
        fmin : float
            minimal value of the frequency scale in Hz (optional)
        fmax : float
            maximal value of the frequency scale in Hz (optional) \n
            if fmax = 0 the maximal value of freq scale is taken (Nyquist frequency)

        Notes
        -----
        FID class is child of resonance class, so all resonance methods are available here.

        """
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
        """Plots FID signal in time domain.

        """
        plt.plot(self.time, self.time_sig)
        plt.xlabel("Time [s]")
        plt.ylabel("Signal")
        plt.show()
