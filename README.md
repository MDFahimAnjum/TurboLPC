![](https://github.com/MDFahimAnjum/TurboLPC/blob/master/media/logo.png?raw=true)

TurboLPC is a fast, simple yet powerful Python library that provides the functionality of Linear Predictive Coding for signals. This README will guide you through the setup and usage of TurboLPC.

## What is Linear Predictive Coding?
LPC is a method used in signal processing to estimate the stochastic random process of a signal. It models the spectral envelope of a signal and predicts future signal values based on past values, making it useful for speech and audio analysis, data compression, and feature extraction.
## Features of TurboLPC

- **Fast LPC Modeling:** Quickly calculate LPC coefficients for time series data using Burg's method.
- **Frequency-Warping in LPC Modeling:** Apply frequency-warping in LPC modeling for dynamic spectral resolution.
- **Cepstrum Coefficients:** Easily convert between LPC model coefficients and cepstral coefficients.
- **Time Series Generation:** Generate time series data from LPC model coefficients with ease.
- **Power Spectrum Density Estimation:** Obtain precise LPC model-based power spectrum density estimates for your time series data.

## Why Choose TurboLPC?

### Exceptional Speed
TurboLPC is engineered for performance, offering LPC modeling functions that are significantly faster—by orders of magnitude—compared to other libraries like Spectrum. This means you can handle larger datasets and more complex computations without compromising on speed.
![](https://github.com/MDFahimAnjum/TurboLPC/blob/master/media/execution_time_perf_plot.png?raw=true)
### Advanced Frequency-Warping Capabilities
TurboLPC goes beyond standard LPC with its frequency-warped LPC feature. This advanced variation estimates spectral powers with a non-uniform resolution, addressing the practical reality that useful information in time series data is often localized in higher or lower frequencies rather than being uniformly distributed. This capability allows for more accurate and insightful spectral analysis, making TurboLPC ideal for applications requiring dynamic frequency resolution.
![](https://github.com/MDFahimAnjum/TurboLPC/blob/master/media/psd_example.png?raw=true)

### Generate Signals from LPC Model
Choose TurboLPC for its powerful signal generation capabilities. Not only does TurboLPC excel in LPC modeling, but it also allows you to generate new signal from model parameters that has the same spectral properties, providing a comprehensive toolset for signal processing, analysis and data augmentation.
![](https://github.com/MDFahimAnjum/TurboLPC/blob/master/media/reconstruct_ts.png?raw=true)

## Installation

### Install from PyPI
You can install TurboLPC using pip:

```bash
pip install turbolpc
```
### Install from source
```bash
git clone git@github.com/MDFahimAnjum/TurboLPC.git
cd TurboLPC
python setup.py install
```

# Usage

## LPC Modeling 
We can calculate LPC coefficients for time series data:

```python
from turbolpc import analysis
import numpy as np

# generate sample data
signal=np.random.randn(1000) # vector of 1 timeseries
signal_matrix=np.tile(signal, (5, 1)).T # matrix of 5 identical timeseries
# fit LPC model 
lpc_order=2 # LPC order
a1,e1,r1=analysis.arburg_vector(x=signal,order=lpc_order)
print(f"LPC coefficients: \n {a1}\n Reflection coeff: \n {r1} \n Sigma:{e1}")
a2,e2,r2=analysis.arburg_matrix(X=signal_matrix,order=lpc_order)
print(f"LPC for 5-column matrix data\n LPC coefficients: \n {a2}\n Reflection coeff: \n {r2} \n Sigma:{e2}")
```

## Frequency-warped LPC Modeling 
We can calculate LPC coefficients for Frequency-warped case:

```python
from turbolpc import analysis
import numpy as np

# generate sample data
signal=np.random.randn(1000) # vector of 1 timeseries
signal_matrix=np.tile(signal, (5, 1)).T # matrix of 5 identical timeseries
# fit LPC model 
lpc_order=2 # LPC order
warp_factor=0.2 # Frequency warping factor
a1,e1,r1=analysis.arburg_warped_vector(x=signal,order=lpc_order,warp_factor=warp_factor)
print(f"LPC coefficients: \n {a1}\n Reflection coeff: \n {r1} \n Sigma:{e1}")
a2,e2,r2=analysis.arburg_warped_matrix(X=signal_matrix,order=lpc_order,warp_factor=warp_factor)
print(f"LPC for 5-column matrix data\n LPC coefficients: \n {a2}\n Reflection coeff: \n {r2} \n Sigma:{e2}")
```

## Generate time series from LPC model
We can generate time series using LPC model coefficients:

```python
from turbolpc import analysis, synthesis
import numpy as np
import matplotlib.pyplot as plt

# generate sample data
signal=np.random.randn(1000) # vector of 1 timeseries

# fit LPC model 
lpc_order=10 # LPC order
a1,e1,r1=analysis.arburg_vector(x=signal,order=lpc_order)

# generate time series
new_ts=synthesis.gen_ts(a=a1,sigma=np.sqrt(e1),n_samples=1000)

# plot and compare
plt.figure()
plt.plot(signal,label="Actual")
plt.plot(new_ts,label="Generated")
plt.legend()
```

## Calculate Cepstral Coefficient
We can calculate cepstral coefficients from LPC model coefficients and vice versa:

```python
from turbolpc import analysis, utils
import numpy as np

# generate sample data
signal=np.random.randn(1000) # vector of 1 timeseries

# fit LPC model 
lpc_order=2 # LPC order
a1,e1,r1=analysis.arburg_vector(x=signal,order=lpc_order)
print(f"LPC Coefficient: \n {a1}")

# LPC to Cepstral coef.
c1=utils.arcoeff_to_cep(a=a1,sigma_squared=e1,N=4)
print(f"Cepstral Coefficient upto 4 elements: \n {c1}")

# Cepstral coef. to LPC
ac1=utils.cep_to_arcoeff(c=c1,order=lpc_order)
print(f"LPC from Cepstral Coefficient: \n {ac1}")
```

## Calculate power spectrum
We can obtain power spectral density from LPC model coefficients:

```python
from turbolpc import analysis, utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

Fs=10000 # sampling freq
t_length=10 # total seconds in timeseries
num_samples = t_length*Fs  # Number of samples
frequency1 = 50       # Frequency of the sinusoid signal
frequency2 = frequency1*10       # Frequency of the sinusoid signal
amplitude = 1       # Amplitude of the sinusoid signal
noise_level = 0.2   # Level of random noise

# Generate time vector
t = np.linspace(0, t_length, num_samples)

# Generate sinusoidal signal
sinusoid_signal = amplitude * np.sin(2 * np.pi * frequency1 * t)
sinusoid_signal = sinusoid_signal+ amplitude * np.sin(2 * np.pi * frequency2 * t)

# Generate random noise
np.random.seed(42)
noise = noise_level * np.random.randn(num_samples)

# Add noise to the sinusoidal signal
signal = sinusoid_signal +  noise

# LPC parameters
lpc_order=20 # LPC order
freq_warping=0.5 # Frequency warping coefficient

# PSD from frequency-warped LPC model
a1,e1,r1=analysis.arburg_warped_vector(x=signal,order=lpc_order,warp_factor=freq_warping)
a1=utils.arcoeff_warp(a=a1,warp_factor=freq_warping,task="unwarp")
w1,pwr1=utils.freqz(a=a1,sigma_squared=e1,worN=2*Fs,fs=Fs)

# PSD from LPC model
a0,e0,r0=analysis.arburg_vector(x=signal,order=lpc_order)
w2,pwr2=utils.freqz(a=a0,sigma_squared=e0,worN=2*Fs,fs=Fs)

# plot
plt.figure()
plt.semilogx(w1,pwr1,label="Freq warped LPC")
plt.semilogx(w2,pwr2,label="LPC")
plt.title('PSD')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.show()
```

# Run Examples and Benchmarks

You can also run the Python jupyter notebook examples with benchmark results directly by running example_notebook.ipynb

# Documentation

## `arburg_vector`
The `arburg_vector` function calculates LPC (Linear Predictive Coding) coefficients from a vector of time series data.

### Syntex
```python
arburg_vector(x, order=1)
```
### Parameters
- x: _numpy.ndarray_
    - Vector of Time series data
- order: _int_
    - Specifies the LPC order. 
    - Has to be positive and integer.
    - Defaults to 1

### Returns
- a: _numpy.ndarray_
    - Vector of LPC coefficients [a1, a2, ...] (without the leading 1).
    - Shape: 1 x order
- E: _float_
    - Squared of error/Noise power (sigma^2)
- ref: _numpy.ndarray_
    - Reflection coefficients.
    - Shape: 1 x order

## `arburg_matrix`
The `arburg_matrix` function calculates LPC (Linear Predictive Coding) coefficients for multiple time series data.

### Syntex
```python
arburg_matrix(X, order=1)
```
### Parameters
- x: _numpy.ndarray_
    - Matrix of Time series data
    - Shape: n x obs 
    - Total number of time series is obs (total columns in the matrix)
    - Each column is a timeseries data of length n
- order: _int_
    - Specifies the LPC order. 
    - Has to be positive and integer.
    - Defaults to 1

### Returns
- a: _numpy.ndarray_
    - Matrix of LPC coefficients (without the leading 1).
    - Each column is a set of LPC coefficients
    - Shape: order x obs
- E: _numpy.ndarray_
    - Squared of error/Noise power (sigma^2)
    - Shape: 1 x obs
- ref: _numpy.ndarray_
    - Reflection coefficients.
    - Each column is a set of reflection coefficients
    - Shape: order x obs

## `arburg_warped_vector`
The `arburg_warped_vector` function calculates Frequency-warped LPC (Linear Predictive Coding) coefficients from a vector of time series data.

### Syntex
```python
arburg_warped_vector(x, order=1,warp_factor=0)
```
### Parameters
- x: _numpy.ndarray_
    - Vector of Time series data
- order: _int_
    - Specifies the LPC order. 
    - Has to be positive and integer.
    - Defaults to 1.
- warp_factor: _float_
    - Frequency warping factor
    - Value between -1 and 1
    - Defaults to 0 (Normal LPC)

### Returns
- a: _numpy.ndarray_
    - Vector of LPC coefficients [a1, a2, ...] (without the leading 1).
    - Shape: 1 x order
- E: _float_
    - Squared of error/Noise power (sigma^2)
- ref: _numpy.ndarray_
    - Reflection coefficients.

## `arburg_warped_matrix`
The `arburg_warped_matrix` function calculates Frequency-warped LPC (Linear Predictive Coding) coefficients for multiple time series data.

### Syntex
```python
arburg_warped_matrix(X, order=1,warp_factor=0)
```
### Parameters
- x: _numpy.ndarray_
    - Matrix of Time series data
    - Shape: n x obs 
    - Total number of time series is obs (total columns in the matrix)
    - Each column is a timeseries data of length n
- order: _int_
    - Specifies the LPC order. 
    - Has to be positive and integer.
    - Defaults to 1.
- warp_factor: _float_
    - Frequency warping factor
    - Value between -1 and 1
    - Defaults to 0 (Normal LPC)
### Returns
- a: _numpy.ndarray_
    - Matrix of LPC coefficients (without the leading 1).
    - Each column is a set of LPC coefficients
    - Shape: order x obs
- E: _numpy.ndarray_
    - Squared of error/Noise power (sigma^2)
    - Shape: 1 x obs
- ref: _numpy.ndarray_
    - Reflection coefficients.
    - Each column is a set of reflection coefficients
    - Shape: order x obs


## `gen_ts`
The `gen_ts` function calculates Cepstral Coefficients from LPC (Linear Predictive Coding) coefficients. 

### Syntex
```python
gen_ts(ar_coeffs,sigma=1,n_samples=1000,data=None)
```
### Parameters
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

### Returns
- time_series: _numpy.ndarray_
    - Generated time series 
    - Shape: 1 x n_samples or the same shape as 'data'


## `arcoeff_to_cep`

The `arcoeff_to_cep` function calculates Cepstral Coefficients from LPC (Linear Predictive Coding) coefficients. 

### Syntex
```python
arcoeff_to_cep(a, sigma_squared, N)
```
### Parameters
- a: _numpy.ndarray_
    - LPC coefficients [a1, a2, ...] (without the leading 1).
- sigma_squared: _float_
    - Square of Noise power
- N: _int_
    - Specifies the desired length of cepstrum coefficients

### Returns
- c: _numpy.ndarray_
    - First N Cepstral coefficients (c0,c1,...cN-1) where c0=log(sigma^2)

## `cep_to_arcoeff`
The `cep_to_arcoeff` function calculates LPC (Linear Predictive Coding) coefficients from Cepstral Coefficients. 

### Syntex
```python
cep_to_arcoeff(c, order)
```
### Parameters
- c: _numpy.ndarray_
    - Cepstrum coefficients [c0, c1, ...] where c0=log(sigma^2)
- order: _int_
    - Order of LPC model

### Returns
- a: _numpy.ndarray_
    - LPC coefficients [a1, a2, ...] (without the leading 1).
    - Shape: 1 x order


## `arcoeff_warp`

The `arcoeff_warp` function recalculates LPC (Linear Predictive Coding) coefficients based on a specified frequency warping or unwarping.

### Syntex
```python
arcoeff_warp(a, warp_factor, task="warp")
```
### Parameters
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

### Returns
- a: numpy.ndarray
    - Warped/Un-warped LPC coefficients [a1, a2, ...] (without the leading 1).

### Warning: Numerical accuracy of arcoeff_warp
arcoeff_warp uses scipy.signal.tf2zpk and scipy.signal.zpk2tf functions which are numerically inaccurate after 60th order. This can be observed by the following example:

```python
from turbolpc import analysis, utils
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

Fs=1000 # sampling freq
t_length=1 # total seconds in timeseries
num_samples = t_length*Fs  # Number of samples
frequency1 = 50       # Frequency of the sinusoid signal
frequency2 = frequency1*10       # Frequency of the sinusoid signal
amplitude = 1       # Amplitude of the sinusoid signal
noise_level = 0.2   # Level of random noise

# Generate time vector
t = np.linspace(0, t_length, num_samples)

# Generate sinusoidal signal
sinusoid_signal = amplitude * np.sin(2 * np.pi * frequency1 * t)
sinusoid_signal = sinusoid_signal+ amplitude * np.sin(2 * np.pi * frequency2 * t)
# Generate random noise
np.random.seed(42)
noise = noise_level * np.random.randn(num_samples)

# Add noise to the sinusoidal signal
signal = sinusoid_signal +  noise

# LPC parameters
freq_warping=0.0 # Frequency warping coefficient

lpc_orders=np.linspace(10,100,9)
error1=np.zeros_like(lpc_orders)
error2=np.zeros_like(lpc_orders)

for i, lpc_order in enumerate(lpc_orders):
    lpc_order=int(lpc_order)
    # PSD from frequency-warped LPC model
    a1,e1,r1=analysis.arburg_warped_vector(x=signal,order=lpc_order,warp_factor=freq_warping)
    a11=utils.arcoeff_warp(a=a1,warp_factor=freq_warping,task="unwarp")
    # PSD from LPC model
    a0,e0,r0=analysis.arburg_vector(x=signal,order=lpc_order)

    # show the problem in scipy.signal.tf2zpk and scipy.signal.zpk2tf functions
    a=np.insert(a11, 0, 1)
    b=np.zeros_like(a)
    b[0]=1
    b2,a2=sig.zpk2tf(*sig.tf2zpk(b, a))

    error1[i]=np.linalg.norm(a0-a11)
    error2[i]=np.linalg.norm(a-a2)


# plot
plt.figure(figsize=(11, 4))
plt.semilogy(lpc_orders,error1,label=r"$LPC(x)-LPC_{Freq. Warped}(x,warp=0)$", marker='d', linestyle='--')
plt.semilogy(lpc_orders,error1,label=r"$x-tf2zpk(zpk2tf(x))$", marker='.', linestyle=':')
plt.title('Numerical Error')
plt.grid(True, which='both', linestyle='--')  # Optional: add grid for better readability
plt.legend()
plt.xlabel('LPC order')
plt.ylabel('MSE Error')
plt.show()
```
Here we test two things:
- Standard LPC and Frequency-warped LPC with 0 warping coefficient should provide same power spectrum
- LPC coefficients going through tf2zpk and zpk2tf should give identical results (x=tf2zpk(zpk2tf(x)))
![](https://github.com/MDFahimAnjum/TurboLPC/blob/master/media/numerical_error.png?raw=true)

## `freqz`
The `freqz` function calculates the power spectrum from Linear Predictive Coding (LPC) coefficients. It provides an easy-to-use interface for analyzing the frequency response of a system defined by its LPC coefficients.

### Syntex
```python
freqz(a=1, sigma=1, worN=1000, whole=False, fs=500)
```
### Parameters
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

### Returns
- w: numpy.ndarray
    - Array of frequencies in Hz at which the power spectrum is computed.
- pwr: numpy.ndarray
    - Power at each frequency, expressed in dB.


# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Contributing

We welcome contributions! 

