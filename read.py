import numpy as np
import re
import glob

# returns single dimension tables
# after file_name one specifies which colums of data should be returned (beginning with 0!)
def file(file_name, *columns):
    data = np.loadtxt(file_name)
    tab = []
    for i in columns:
        tab.append(data[:,i])
    return list(tab)

# returns table of file names in a given directory (not sorted!)
def all_names(directory):
    if directory[-1] != '/':
        directory += '/'
    return np.array(glob.glob(directory + "*.dat"))

# reads current value from file name there must be such a pattern in the file name: "Curr_iii_uA"
def current(file_name):
    current_reg_ex = re.compile(r'Curr_(-)*\d(\d)?(\d)?(\d)?(\d)?_uA')
    current_val = current_reg_ex.search(file_name)
    current_val_float = float(current_val.group().split('_')[1])
    return current_val_float

# returns all current values (not sorted!)
def all_currents(directory):
    file_names = all_names(directory)
    curr = []
    for file_name in file_names:
        curr.append(current(file_name))
    return np.array(curr)

# sorts the input tables by current values
def sort_by_current(file_names, currents):
    currents = np.array(currents)
    inds = np.argsort(currents)
    currents = currents[inds]
    file_names = file_names[inds]
    return file_names, currents

# returns currents and names sorted by current value
def all_names_currents(directory):
    names = all_names(directory)
    currs = all_currents(directory)
    return sort_by_current(names, currs)

# params: time series and signal time_sig, returns frequency scale freq and complex valued FFT
# FFT is real-input FFT - computes values only for freq >= 0
# a and b defines the interval in the time domain that is taken to calculate fft
# fmin and fmax defines the interval in the frequency domain
def comp_fft(time, time_sig, a = 0, b = -1, fmin = 0, fmax = 0):
    fft = np.fft.rfft(time_sig[a:b])
    freq = np.fft.rfftfreq(len(time[a:b]), time[1]-time[0])

    if fmax == 0:
        fmax = np.max(freq)
    ind_1 = np.where(freq >= fmin)
    ind_2 = np.where(freq <= fmax)
    inds = np.intersect1d(ind_1, ind_2)
    return freq[inds], fft[inds]