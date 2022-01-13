import numpy as np
import matplotlib.pyplot as plt
from . import functions as func
from . import fit
from . import read

# file_name requires
class resonance:
    """Reads data from the given file. Fit to the measurement data isn't run automatically. Please use fit method before using fit parameters.

    Parameters
    ----------
    file_name : string
        path to file with measured resonance
    columns : int (multiple)
        specifies 3 columns in the data file that include frequency, X and Y measurements
    linear_background : bool
        when true fit.complex_lorentz_lin_back() is used, when false - fit.complex_lorentz()
    """

    def __init__(self, file_name, linear_background = True, columns = [0,1,2], freq_min = 0, freq_max=0):
        """Reads data from the given file.

        Parameters
        ----------
        file_name : string
            path to file with measured resonance
        freq_min : float
            Minimal frequency to read
        freq_max : float
            Maximal frequency to read
        """
        try:
            self.current = read.current(file_name)
        except:
            self.current = np.nan
            print("Achtung! Reading current value from the file name was unsuccesfull.")
            print("You may still use the class but remember that get_current() will return fake value.")
        self.file_name = file_name
        freq, X, Y = read.file(file_name, *columns) # uwaga na liczenie phi - wykorzystać numpy.arctan2()
        if freq_max != 0:
            inds = (np.array(freq) >= freq_min) * (np.array(freq) <= freq_max)
            freq = freq[inds]
            X = X[inds]
            Y = Y[inds]
        R = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(X,Y)

        self.freq = freq
        self.sig = X + 1j*Y
        self.R = R
        self.phi = phi
        self.fit_bool = False
        self.read_bool = True


    def fit(self, p0 = [], linear_background = True):
        """Fitting complex lorentzian function (with or without linear background).

        Parameters
        ----------
        p0 : list
            [f0, A, gamma, phi] initial parameters
        linear_background : bool
            when true fit.complex_lorentz_lin_back() is used, when false - fit.complex_lorentz()

        """
        if not self.read_bool:
            print("Error: Please use comp_fft method to get complex lorentzian before fitting.")
        if linear_background:
            self.model = func.complex_lorentz_lin_back
            self.popt, self.pcov = fit.complex_lorentz_lin_back(self.freq, self.sig, p0)
        else:
            self.model = func.complex_lorentz
            self.popt, self.pcov = fit.complex_lorentz(self.freq, self.sig, p0)

        self.fit_bool = True

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
        if self.fit_bool:
            return self.popt[i]
        print("Error: Please run fit method before getting parameters!")
        return -1

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
        if self.fit_bool:
            return np.sqrt(self.pcov[i][i])
        print("Error: Please run fit method before getting parameters!")
        return 0

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

    def plot_real(self, plot_fit = False):
        """Plots real part of the measured resonance and fitted function (if chosen).

        Parameters
        ----------
        plot_fit : bool
            True for plotting fitted function together with the measured data (optional).
        """
        plt.plot(self.freq, self.sig.real)
        if self.fit_bool and plot_fit:
            plt.plot(self.freq, self.model(self.freq, *self.popt).real)
        plt.xlabel("Frequency [Hz]")
        plt.show()
        if plot_fit and not self.fit_bool:
            print("Please use fit method before plotting fitted function.")


    def plot_imag(self, plot_fit = False):
        """Plots imaginary part of the measured resonance and fitted function (if chosen).

        Parameters
        ----------
        plot_fit : bool
            True for plotting fitted function together with the measured data.
        """
        plt.plot(self.freq, self.sig.imag)
        if self.fit_bool and plot_fit:
            plt.plot(self.freq, self.model(self.freq, *self.popt).imag)
        plt.xlabel("Frequency [Hz]")
        plt.show()
        if plot_fit and not self.fit_bool:
            print("Please use fit method before plotting fitted function.")

    def plot_abs(self, plot_fit = False):
        """Plots absolute value of the resonance (measurement and fit function)

        Parameters
        ----------
        plot_fit : bool
            True for plotting fitted function together with the measured data.
        """
        plt.plot(self.freq, np.abs(self.sig))
        if self.fit_bool and plot_fit:
            plt.plot(self.freq, np.abs(self.model(self.freq, *self.popt)))
        plt.xlabel("Frequency [Hz]")
        plt.show()
        if plot_fit and not self.fit_bool:
            print("Please use fit method before plotting fitted function.")


        # spróbować nadpisać metodę fit w klasie FID z domyślnym parametrem background = False

class FID(resonance):
    """Reads FID data from the given file.
        Fit to the measurement data isn't run automatically. Please use fit method before getting parameters.

        Parameters
        ----------
        file_name : string
            path to file with measured resonance

        Notes
        -----
        FID class is child of resonance class, so all resonance methods are available here.

        Important: before using fit method please run comp_fft!

        """

    def __init__(self, file_name):
        """Converting time signal (FID) into frequency domain complex lorentzian resonance.
        Fit to the measurement data isn't run automatically. Please use fit method before getting parameters.


        Parameters
        ----------
        file_name : string
            path to file with measured resonance

        Notes
        -----
        FID class is child of resonance class, so all resonance methods are available here.

        """
        self.current = read.current(file_name)
        self.file_name = file_name
        time, time_sig = read.file(file_name, 0, 1)
        self.time = time
        self.time_sig = time_sig
        self.read_bool = False # read_bool is True when complex lorentzian signal is ready - in case of FID class this is after comp_fft method is run
        self.fit_bool = False
        self.sig_gap = 0

    def comp_fft(self, a = 0, b = -1, fmin = 0, fmax = 0 ):
        """Converting time signal (FID) into frequency domain complex lorentzian resonance.
        This must be run before using fit method in FID class!

        Parameters
        ----------
        a : int
            element of the time array defining the beginning of the interval that is going to be used in FFT (optional)
        b : int
            element of the time array defining the end of the interval that is going to be used in FFT (optional)
        fmin : float
            minimal value of the frequency scale in Hz (optional)
        fmax : float
            maximal value of the frequency scale in Hz (optional) \n
            if fmax = 0 the maximal value of freq scale is taken (Nyquist frequency)

        """
        freq, fft = read.comp_fft(self.time, self.time_sig, a = a, b = b, fmin = fmin, fmax = fmax)
        self.freq = freq
        self.sig = fft
        self.read_bool = True

    def plot_FID(self):
        """Plots FID signal in time domain.

        """
        plt.plot(self.time, self.time_sig)
        plt.xlabel("Time [s]")
        plt.ylabel("Signal")
        plt.show()

    def fit_gap(self):
        """Obtains gap between two FIDs that are included in single file.
        This is only for "step change" measurements (for example when obtaining compensation point).

        Returns
        -------
        sig_gap : float
            gap between the two FIDs

        Notes
        -----
        Each FID in the given file must cover exactly half of the measurement points.

        """
        length = len(self.time_sig)
        avg1 = np.mean(self.time_sig[int(0.8*length):])
        avg2 = np.mean(self.time_sig[int(0.3*length):int(0.5*length)])
        self.sig_gap = avg2 - avg1
        return self.sig_gap

    def get_sig_gap(self):
        """Gets signal gap if fit_gap method was previously used.
        This is only for "step change" measurements (for example when obtaining compensation point).

        Returns
        -------
        sig_gap : float
            gap between the two FIDs

        """
        if self.sig_gap != 0:
            return self.sig_gap
        print("Warning: Did you use fit_gap before get_sig_gap?")
        return self.sig_gap
