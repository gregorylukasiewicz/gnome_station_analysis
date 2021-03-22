import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from scipy.optimize import curve_fit


# function for reading of the current fom the file name.
# The pattern Curr_***_uA, where *** is the current value
def get_curr(file_name):
    current_deg_ex = re.compile(r'Curr_(-)*\d(\d)?(\d)?(\d)?(\d)?_uA')
    current_val = current_deg_ex.search(file_name)
    current_val_float = float(current_val.group().split('_')[1])
    return current_val_float


def complex_lorentz(f, f_0, A, gamma, phi):
    return A * np.exp(1j * phi) * (gamma - 1j * (f - f_0)) / (gamma ** 2 + (f - f_0) ** 2)


def vec_model(f, f_0, A, gamma, phi):
    real_part = np.real(complex_lorentz(f, f_0, A, gamma, phi))
    imag_part = np.imag(complex_lorentz(f, f_0, A, gamma, phi))
    return np.hstack([real_part, imag_part])


def sine_mod(t, f, A, gamma, phi, B):
    return A * np.sin(2 * np.pi * t * f + phi) * np.exp(-gamma * t) + B


# class with the main functions needed for data analysis
class FID_out:
    # initialisation of the class with file name only
    def __init__(self, file_name):
        self.file_name = file_name
        self.curr = get_curr(file_name)

    # loading data
    def load_data(self, a=101, b=-1):
        self.data = np.loadtxt(self.file_name)
        self.sig = self.data[a:b, 1]
        self.time = self.data[a:b, 0]

    # computing fft and freq (only the output is the positive-freq part of the douple-side fft)
    def comp_fft(self, a, b):
        self.fft = np.fft.rfft(self.data[a:b, 1])  # rfft is fft of real input, the output is only for freq > 0
        self.freq = np.fft.rfftfreq(len(self.fft), self.data[1, 0] - self.data[0, 0])
        # self.sig = self.data[a:b, 1]
        # self.time = self.data[a:b, 0]

    # saving the fft to npy file
    def save_fft(self, directory=''):
        np.save(self.file_name + '_FFT', np.array([self.freq, self.fft]))

    # loading the fft from the npy file
    def load_fft(self):
        x = np.load(self.file_name + '_FFT.npy')
        self.fft = x[1]
        self.freq = np.real(x[0])

    # the funtion allows to find the indexes for the desired frequnecy range
    def find_ind(self, f_min=0.1, f_max=10):
        ind_1 = np.where(self.freq >= f_min)
        ind_2 = np.where(self.freq <= f_max)
        self.ind = np.intersect1d(ind_1, ind_2)

    # fitting of the complex lorentzian
    def lorentzian_fit(self, f=1, A=1e4, gamma=1e-3, phi=0):
        fft = self.fft[self.ind]
        self.fft_fit = fft
        r = np.abs(fft)
        freq = self.freq[self.ind]
        self.freq_fit = freq
        meas_real = np.real(fft)
        meas_imag = np.imag(fft)
        meas = np.hstack([meas_real, meas_imag])
        ind_max = np.where(r == np.max(r))
        A_guess = r[ind_max][0]
        f_guess = np.abs(freq[ind_max])[0]
        phi_guess = phi
        gamma_guess = np.abs(freq[meas_imag.argmax()] - freq[meas_imag.argmin()])
        if gamma_guess > 0.1:
            gamma_guess = gamma
        popt, pcov = curve_fit(vec_model, freq, meas, p0=[f_guess, A_guess, gamma_guess, phi_guess])
        self.fit_opt = popt
        self.cov = pcov
        self.t2_s = 1 / popt[2]
        self.t2_min = (1 / (60 * popt[2]))
        return popt, pcov

    def sine_mod_fit(self, f=10, A=100, gamma=0.005, phi=0):
        time = self.time
        sig = self.sig
        B = np.mean(sig)
        A = np.max(np.abs(sig)) - np.abs(B)
        print(A)
        print(B)

        # plt.plot(time, sine_mod(time, f, A, gamma, 0, B))
        # plt.show()

        popt, pcov = curve_fit(sine_mod, time, sig, p0=[1, 1, 1, 1, 1])
        print(popt)
        # print(pcov)
        return popt, pcov

print("Hi! I'm FID_out")
print("Try me out:")
print(sine_mod(1,1,1,1,1,1))