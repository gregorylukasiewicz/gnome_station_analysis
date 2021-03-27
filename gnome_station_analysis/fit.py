import numpy as np
from scipy.optimize import curve_fit
from . import functions as func

# params: freq scale and complex signal, returns guess initial params for the fit
def guess_initial(freq, sig, phi = 0):
    R = np.abs(sig)
    ind_max = np.where(R == np.max(R))  # R peak
    f0_guess = np.abs(freq[ind_max])[0]
    A_guess = R[ind_max][0]
    gamma_guess = np.abs(freq[sig.imag.argmax()] - freq[sig.imag.argmin()])
    phi_guess = phi
    p0 = [f0_guess, A_guess, gamma_guess, phi_guess]
    return p0

# params: freq scale and complex signal, returns popt and pcov of the complex_lorentz function fit
def complex_lorentz(freq, sig, phi = 0):
    p0 = guess_initial(freq, sig, phi)
    sig_vector = np.hstack([sig.real, sig.imag])
    popt, pcov = curve_fit(func.vec_model, freq, sig_vector, p0 = p0)
    return popt, pcov

def complex_lorentz_lin_back(freq, sig, phi = 0):
    p0 = guess_initial(freq, sig, phi)
    p0 = p0 + [0,0,0,0] # adding initial params for liear background
    sig_vector = np.hstack([sig.real, sig.imag])
    popt, pcov = curve_fit(func.vec_model_lin_back, freq, sig_vector, p0 = p0)
    return popt, pcov