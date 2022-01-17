import numpy as np
from scipy.optimize import curve_fit
from . import functions as func

# params: freq scale and complex signal, returns guess initial params for the fit
def guess_initial(freq, sig, phi = 0):
    """Guessing initial parameters for fitting complex lorentzian function to the given data set.

    Parameters
    ----------
    freq : array like
        frequency scale in Hz (from measurement)
    sig : complex array like
        complex lorentzian signal (from measurement)
    phi : float
        initial guess for phase (optional)

    Returns
    -------
    p0 : array
        [f0_guess, A_guess, gamma_guess, phi_guess] - table of initial parameters for fitting functions.complex_lorentz() and functions.complex_lorentz_lin_back() to 'freq' and 'sig' data set.

    Notes
    -----
    The method is used in fit.complex_lorentz() and fit.complex_lorentz_lin_back().
    Please look at the code to understand the method. Note that it works correctly only for a good quality measurements (for example the lorentzian peak must be the largest value of the data set).

    """
    R = np.abs(sig)
    ind_max = np.where(R == np.max(R))  # R peak
    ind_min = np.where(R == np.min(R))  # R minimum
    #f0_guess = np.abs(freq[ind_max])[0]
    A_guess = R[ind_max][0] - R[ind_min][0] # I don't know why it works better this way...
    gamma_guess = np.abs(freq[sig.imag.argmax()] - freq[sig.imag.argmin()])
    f0_guess = freq[int(np.mean([sig.imag.argmax(), sig.imag.argmin()]))]
    #A_guess = R[int(np.mean([sig.imag.argmax(), sig.imag.argmin()]))]
    phi_guess = phi
    p0 = [f0_guess, A_guess, gamma_guess, phi_guess]
    return p0

# params: freq scale and complex signal, returns popt and pcov of the complex_lorentz function fit
def complex_lorentz(freq, sig, p0 = [], phi = 0):
    """Fitting functions.lorentz_functions() to the given data set.

    Parameters
    ----------
    freq : array like
        frequency scale in Hz (from measurement)
    sig : complex array like
        complex lorentzian signal (from measurement)
    p0 : list
        [f0, A, gamma, phi] - initial parameters
    phi : float
        initial guess for phase (optional)

    Returns
    -------
    popt, pcov : arrays
        popt = [f0_fit, A_fit, gamma_fit, phi_fit] and covariance matrix pcov

    Notes
    -----

    Initial conditions are given by fit.guess_initial().

    To obtain the error of a chosen parameter popt[i] please use np.sqrt(pcov[i][i]).

    """
    if len(p0) == 0:
        p0 = guess_initial(freq, sig, phi)
    sig_vector = np.hstack([sig.real, sig.imag])
    popt, pcov = curve_fit(func.vec_model, freq, sig_vector, p0 = p0)
    return popt, pcov

def complex_lorentz_lin_back(freq, sig, p0 = [], phi = 0):
    """Fitting functions.complex_lorentz_lin_back() to the data set.

    Parameters
    ----------
    freq : array like
        frequency scale in Hz (from measurement)
    sig : complex array like
        complex lorentzian signal (from measurement)
    p0 : list
        [f0, A, gamma, phi] - initial fit parameters
    phi : float
        initial guess for phase (optional)

    Returns
    -------
    popt, pcov : arrays
        popt = [f0_fit, A_fit, gamma_fit, phi_fit, a_real_fit, b_real_fit, a_imag_fit, b_imag_fit] and covariance matrix pcov

    Notes
    -----
    Initial conditions are given by fit.guess_initial().

    To obtain the error of a chosen parameter popt[i] please use np.sqrt(pcov[i][i]).

    """
    if len(p0) == 0:
        p0 = guess_initial(freq, sig, phi)
    p0 = p0 + [0,0,0,0] # adding initial params for linear background
    sig_vector = np.hstack([sig.real, sig.imag])
    popt, pcov = curve_fit(func.vec_model_lin_back, freq, sig_vector, p0 = p0)
    return popt, pcov
