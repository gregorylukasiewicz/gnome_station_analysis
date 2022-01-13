import numpy as np
import re
import glob

# returns single dimension tables
# after file_name one specifies which colums of data should be returned (beginning with 0!)
def file(file_name, *columns):
    """Reads selected columns in a chosen file

    Parameters
    ----------
    file_name : string
    columns : int (multiple)
        specifies wchich colums od data should be returned (beginning with 0!)

    Returns
    -------
    (columns) : list of arrays
        columns of data listed in parameters after file_name

    Examples
    --------

    >time, Ch2 = gnome_station_analysis.read.file("name", 0, 2) \n
    returns 2 arrays of data: time and signal in channel 2.

    >col = [0, 2, 3] \n
    >data = gnome_station_analysis.read.file("name", *col) \n
    data[0] - 1st column of data \n
    data[1] - 3rd column of data \n
    data[2] - 4th column of data

    """
    data = np.loadtxt(file_name)
    tab = []
    for i in columns:
        tab.append(np.array(data[:,i]))
    return list(tab)

# returns table of file names in a given directory (not sorted!)
def all_names(directory):
    """Finding all .dat files in a given directory

    Parameters
    ----------
    directory : string
        path to a direcotry with a series of measurement files

    Returns
    -------
        file_names : array of strings
            all .dat files in a given directory

    """
    if directory[-1] != '/':
        directory += '/'
    return np.array(glob.glob(directory + "*.dat"))

# reads current value from file name there must be such a pattern in the file name: "Curr_iii_uA"
def current(file_name):
    """ Returns current value saved in the file name

    Parameters
    ----------
    file_name : string
        single file path

    Returns
    -------
    curr : float
        current value in uA

    Notes
    -----

    The file name must be 'path.../name...Curr_iii_uA.dat'. The function reads 'iii' and returns the float value.

    """
    current_reg_ex = re.compile(r'Curr_(-)*\d(\d)?(\d)?(\d)?(\d)?_uA')
    current_val = current_reg_ex.search(file_name)
    current_val_float = float(current_val.group().split('_')[1])
    return current_val_float

# returns all current values (not sorted!)
def all_currents(directory):
    """Reads all currents saved in file names in a given direcotory

    Parameters
    ----------
    directory : string

    Returns
    -------
    currs : numpy array
        values of current in measurements saved in the given directory

    """
    file_names = all_names(directory)
    curr = []
    for file_name in file_names:
        curr.append(current(file_name))
    return np.array(curr)

# sorts the input tables by current values
def sort_by_current(file_names, currents):
    """Sorts both arrays according to rising current value

    Parameters
    ----------
    file_names : array of strings
        table of file names
    currents : array of floats
        array of current values corresponding to file_names

    Returns
    -------
    file_names, currents : array of strings, array of floats
        arrays sorted by rising current values

    """
    currents = np.array(currents)
    inds = np.argsort(currents)
    currents = currents[inds]
    file_names = file_names[inds]
    return file_names, currents

# returns currents and names sorted by current value
def all_names_currents(directory):
    """Reads all file_names and currents saved in file names in a given direcotory

    Parameters
    ----------
    directory : string
        path to a direcotry with a series of measurements

    Returns
    -------
    file_names, currents : array of strings, array of floats
        arrays sorted by rising current values
    """
    names = all_names(directory)
    currs = all_currents(directory)
    return sort_by_current(names, currs)

# params: time series and signal time_sig, returns frequency scale freq and complex valued FFT
# FFT is real-input FFT - computes values only for freq >= 0
# a and b defines the interval in the time domain that is taken to calculate fft
# fmin and fmax defines the interval in the frequency domain
def comp_fft(time, time_sig, a = 0, b = -1, fmin = 0, fmax = 0):
    """Computes FFT of real signal in time domain

    Parameters
    ----------
    time : array
        time scale in s
    time_sig : array of real floats
        array of real-valued signal in time domain
    a : int
        element of the time array defining the beginning of the interval that is going to be used in FFT (optional)
    b : int
        element of the time array defining the end of the interval that is going to be used in FFT (optional)
    fmin : float
        minimal value of the returned frequency scale in Hz (optional)
    fmax : float
        maximal value of the frequency scale in Hz (optional) \n
        if fmax = 0 the maximal value of freq scale is taken (Nyquist frequency)

    Returns
    -------
    freq, sig : array of floats, array of complex floats
        frequency scale (only positive values) in Hz and complex signal in frequency domain

    Notes
    -----
    The function uses numpy.fft.rfft. Please read numpy documentation for more details.

    """
    fft = np.fft.rfft(time_sig[a:b])
    freq = np.fft.rfftfreq(len(time[a:b]), time[1]-time[0])

    if fmax == 0:
        fmax = np.max(freq)
    ind_1 = np.where(freq >= fmin)
    ind_2 = np.where(freq <= fmax)
    inds = np.intersect1d(ind_1, ind_2)
    return freq[inds], fft[inds]