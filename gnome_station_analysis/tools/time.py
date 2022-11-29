import re 
import numpy as np

# function for reading of time from the file name
def get_time(file_name):
    temp = re.findall(r'\d+', file_name)
    res = list(map(int, temp))
    #returns day,  hh,mm,ss
    size = len(res)
    #return np.array([res[3],  res[6], res[7], res[8]])
    return np.array([res[size-6], res[size-3], res[size-2], res[size-1]])

# function for calculation of time period between defined start and time of measurement
def time_from_start(time, start = np.array([23, 18, 0,0])):
    t_h = (time-start)*np.array([24,1,1/60,1/3600]) # <-
    return np.sum(t_h)
# reads times and calculates time from the start
def get_time_from_start(file_name, start = np.array([0, 0, 0,0])):
    time = get_time(file_name)
    return time_from_start(time, start = start)
