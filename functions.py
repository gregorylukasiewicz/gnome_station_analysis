import numpy as np

# complex lorentz function
def complex_lorentz(f,f0, A,gamma, phi):
        return A*np.exp(1j*phi)*(gamma-1j*(f-f0))/(gamma**2+(f-f0)**2)

# vector model for complex_lorentz
def vec_model(f, f0, A, gamma, phi):
    real_part = np.real(complex_lorentz(f, f0, A,gamma, phi))
    imag_part = np.imag(complex_lorentz(f, f0, A,gamma, phi))
    return np.hstack([real_part,imag_part])

# complex_lorentz + complex linear background (4 additional params)
def complex_lorentz_lin_back(f, f0, A, gamma, phi, a_real, b_real, a_imag, b_imag):
        lor = complex_lorentz(f, f0, A, gamma, phi)
        back = (a_real + 1j * a_imag) * f + (b_real + 1j * b_imag)
        return lor + back

# vector model for complex_lorentz_lin_back
def vec_model_lin_back(f, f0, A, gamma, phi, a_real, b_real, a_imag, b_imag):
    real_part = np.real(complex_lorentz(f, f0, A, gamma, phi)) + a_real * f + b_real
    imag_part = np.imag(complex_lorentz(f, f0, A, gamma, phi)) + a_imag * f + b_imag
    return np.hstack([real_part, imag_part])



