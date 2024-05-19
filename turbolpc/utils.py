import numpy as np
from scipy import signal as sig

def arcoeff_to_cep(a, sigma_squared, N):
    """
    calculates Cepstral Coefficients from LPC (Linear Predictive Coding) coefficients. 

    Input: 
    - a: _numpy.ndarray_
        - LPC coefficients [a1, a2, ...] (without the leading 1).
    - sigma_squared: _float_
        - Square of Noise power
    - N: _int_
        - Specifies the desired length of cepstrum coefficients
    
    Output:
    - c: _numpy.ndarray_
        - First N Cepstral coefficients (c0,c1,...cN-1) where c0=log(sigma^2)
    """
    c = [0] * N
    c[0] = np.log(sigma_squared)
    c[1] = -a[0]        
    for n in range(2, N):
        if n <= len(a):
            c[n] = -a[n - 1] - sum((1 - m / n) * a[m - 1] * c[n - m] for m in range(1, n))
        else:
            c[n] = - sum((1 - m / n) * a[m - 1] * c[n - m] for m in range(1, len(a)+1))       
    return c

def cep_to_arcoeff(c, order):
    """
    calculates LPC (Linear Predictive Coding) coefficients from Cepstral Coefficients
    
    Input:
    - c: _numpy.ndarray_
        - Cepstrum coefficients [c0, c1, ...] where c0=log(sigma^2)
    - order: _int_
        - Order of LPC model    
    
    Output:
    - a: _numpy.ndarray_
        - LPC coefficients [a1, a2, ...] (without the leading 1).
        - Shape: 1 x order
    """
    a = [0] * order
    a[0] = -c[1]        
    for i in range(2, order+1):
            a[i-1] = -c[i] - sum((1 - m / i) * c[i - m] * a[m-1] for m in range(1, i))              
    return a

def freqz(a=1,sigma_squared=1, worN=1000, whole=False, fs=500):
    """
    Input:
    Calculates power spectrum from LPC coefficients
    
    Input:
    - a: numpy.ndarray
        - LPC coefficients [a1, a2, ...] (without the leading 1).
        - Default is 1
    - sigma_squared: float, optional
        - Square of Noise power.
        - Default is 1.
    - worN: int, optional
        - Specifies the number of points at which the frequency response will be computed.
        - Default is 1000.
    - whole: bool, optional
        - Determines whether the frequency range should be from 0 to π (if whole=True) or from 0 to the Nyquist frequency (if whole=False).
        - Default is False.
    - fs: float, optional
        - Sampling frequency in Hz.
        - Default is 500.

    Output:
    - w: numpy.ndarray
        - Array of frequencies in Hz at which the power spectrum is computed.
    - pwr: numpy.ndarray
        - Power at each frequency, expressed in dB.
    """
    a=np.insert(a, 0, 1)
    b=np.zeros_like(a)
    b[0]=1
    w, h = sig.freqz(b, a, worN=worN, whole=whole, fs=fs)
    pwr=20 * np.log10(abs(h)*np.sqrt(sigma_squared))
    return w,pwr 

def arcoeff_warp(a,warp_factor,task="warp"):
    """
    function recalculates LPC (Linear Predictive Coding) coefficients based on a specified frequency warping or unwarping.

    Input:
    - a: numpy.ndarray
        - LPC coefficients [a1, a2, ...] (without the leading 1).
    - warp_factor: float
        - Frequency warping factor
        - Value between -1 and 1
    - worN: int
        - Specifies the number of points at which the frequency response will be computed.
        - Default is 1000.
    - task: str
        - Determines the objective of the function. 
        - "warp" for warping from unwarped condition
        - "unwarp" for unwarping from warped condition
    
    Output:
    - a: numpy.ndarray
        - Warped/Un-warped LPC coefficients [a1, a2, ...] (without the leading 1).
    """

    if task=="warp":
        s=-1
    elif task=="unwarp":
        s=1
    else:
        raise ValueError("task must be warp or unwarp")
    a=np.insert(a, 0, 1)
    b=np.zeros_like(a)
    b[0]=1
    # get poles
    z, p, k = sig.tf2zpk(b, a)
    # pole conversion
    p_new=(p+s*warp_factor)/(1+s*warp_factor* p)
    b,a=sig.zpk2tf(z, p_new, k)
    a=a[1:] # exclude the first 1
    return a