import numpy as np

# complex lorentz function
def complex_lorentz(f, f0, A,gamma, phi):
    """Complex lorentzian function

    Parameters
    ----------
    f : float or array_like
        frequency
    f0 : float
        resonant frequency
    A : float
        scaling factor
    gamma : float
        :math:`\gamma` -- full width at half maximum (FWHM)
    phi : float
        :math:`\phi` -- phase

    Returns
    ----------
    value : complex float or array
        complex lorentzian

    Notes
    ----------
    Used formula

    .. math::
        Ae^{i\phi}\\frac{\gamma - i(f-f_0)}{\gamma^2 + (f-f_0)^2}

    """
    return A*np.exp(1j*phi)*(gamma-1j*(f-f0))/(gamma**2+(f-f0)**2)

# vector model for complex_lorentz
def vec_model(freq, f0, A, gamma, phi):
    """Vector model for complex lorentzian function (complex_lorentz)

    Parameters
    ----------
    freq : array_like
        frequency
    f0 : float
        resonant frequency
    A : float
        scaling factor
    gamma : float
        :math:`\gamma` -- full width at half maximum (FWHM)
    phi : float
        :math:`\phi` -- phase

    Returns
    ----------
    vec_model : array
        length(vec_model) = 2*length(freq)

    """
    real_part = np.real(complex_lorentz(freq, f0, A,gamma, phi))
    imag_part = np.imag(complex_lorentz(freq, f0, A,gamma, phi))
    return np.hstack([real_part,imag_part])

# complex_lorentz + complex linear background (4 additional params)
def complex_lorentz_lin_back(f, f0, A, gamma, phi, a_real, b_real, a_imag, b_imag):
    """Complex lorentzian function with complex linear background

    Parameters
    ----------
    f : float or array_like
        frequency
    f0 : float
        resonant frequency
    A : float
        scaling factor
    gamma : float
        :math:`\gamma` -- full width at half maximum (FWHM)
    phi : float
        :math:`\phi` -- phase
    a_real : float
        real slope
    b_real : float
        real intersection
    a_imag : float
        imaginary slope
    b_imag : float
        imaginary intersection

    Returns
    ----------
    value : complex float or array
        complex lorentzian with complex linear background

    Notes
    ----------
    Used formula

    .. math::
        Ae^{i\phi}\\frac{\gamma - i(f-f_0)}{\gamma^2 + (f-f_0)^2} + (a_{imag}+a_{real})f + (b_{imag} + b_{real})

    """
    lor = complex_lorentz(f, f0, A, gamma, phi)
    back = (a_real + 1j * a_imag) * f + (b_real + 1j * b_imag)
    return lor + back

# vector model for complex_lorentz_lin_back
def vec_model_lin_back(freq, f0, A, gamma, phi, a_real, b_real, a_imag, b_imag):
    """Vector model for complex lorentzian function with complex linear background (complex_lorentz_lin_back)

    Parameters
    ----------
    freq : float or array_like
        frequency
    f0 : float
        resonant frequency
    A : float
        scaling factor
    gamma : float
        :math:`\gamma` -- full width at half maximum (FWHM)
    phi : float
        :math:`\phi` -- phase
    a_real : float
        real slope
    b_real : float
        real intersection
    a_imag : float
        imaginary slope
    b_imag : float
        imaginary intersection

    Returns
    ----------
    vector_model : array
        length(vec_model) = 2*length(freq)

    """
    real_part = np.real(complex_lorentz(freq, f0, A, gamma, phi)) + a_real * freq + b_real
    imag_part = np.imag(complex_lorentz(freq, f0, A, gamma, phi)) + a_imag * freq + b_imag
    return np.hstack([real_part, imag_part])



