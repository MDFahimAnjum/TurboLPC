import numpy as np
from scipy.signal import lfilter

def gen_ts(a,sigma=1,n_samples=1000,data=None):
    """
    Generate time series using LPC coefficients
    Input:
    - a: _numpy.ndarray_
        - LPC coefficients [a1, a2, ...] (without the leading 1).
    - sigma: _float_
        - Noise power
        - Defaults to 1
    - n_samples: _int_
        - Specifies the total number of generated samples. 
        - Defaults to 1000
    - data: _numpy.ndarray_
        - input timeseries data
        - Defaults to None
        - If provided, 'sigma' and 'n_samples' are ignored

    Output:
    - time_series: _numpy.ndarray_
        - Generated time series 
        - Shape: 1 x n_samples or the same shape as 'data'
    """
    if data==None:
        mean = 0
        std_dev = sigma
        data = np.random.normal(mean, std_dev, n_samples)
    time_series = lfilter([1], np.hstack(([1], a)), data)
    time_series=np.real(time_series)
    return time_series