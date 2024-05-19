import numpy as np

def arburg_vector(x, order=1):
    """
    Calculates LPC coefficients from time series data.
    
    Input:
    - x: _numpy.ndarray_
        - Vector of Time series data
    - order: _int_
        - Specifies the LPC order. 
        - Has to be positive and integer.
        - Defaults to 1
    
    Output:
    - a: _numpy.ndarray_
        - Vector of LPC coefficients [a1, a2, ...] (without the leading 1).
        - Shape: 1 x order
    - E: _float_
        - Squared of error/Noise power (sigma^2)
    - ref: _numpy.ndarray_
        - Reflection coefficients
        - Shape: 1 x order
    
    """
    x = np.array(x)
    N = len(x)

    if order == 0.: 
        raise ValueError("order must be > 0")

    # Initialization
    efp = x[1:]
    ebp = x[:-1]

    # Initial error
    E = np.dot(x, x) / N

    a = np.zeros(order+1)
    a[0] = 1

    ref = np.zeros(order)

    for m in range(order):
        # Calculate the next order reflection (parcor) coefficient
        k = (-2 * np.dot(ebp, efp)) / (np.dot(efp.T, efp) + np.dot(ebp, ebp))
        ref[m] = k
        
        # Update the forward and backward prediction errors
        ef = efp[1:] + k * ebp[1:]
        ebp = ebp[:-1] + np.conj(k) * efp[:-1]
        efp = ef

        # Update the AR coeff.
        if m==0:
            a[1] = a[1] + k * np.conj(a[0])
        else:
            a[1:m+2] = a[1:m+2] + k * np.conj(a[m::-1])

        # Update the prediction error
        E = (1 - np.conj(k) * k) * E

    a=a[1:]
    # Assign outputs
    return a, E, ref

def arburg_matrix(X, order=1):
    """
    calculates LPC (Linear Predictive Coding) coefficients for multiple time series data.

    Input:
    - x: _numpy.ndarray_
        - Matrix of Time series data
        - Shape: n x obs 
        - Total number of time series is obs (total columns in the matrix)
        - Each column is a timeseries data of length n
    - order: _int_
        - Specifies the LPC order. 
        - Has to be positive and integer.

    Output:
    - a: _numpy.ndarray_
        - Matrix of LPC coefficients (without the leading 1).
        - Shape: obs x order
    - E: _numpy.ndarray_
        - Squared of Noise power (sigma^2)
        - Shape: 1 x obs
    - ref: _numpy.ndarray_
        - Reflection coefficients.
        - Shape: obs x order
    """
    x = np.array(X)
    N,obs = x.shape
    

    if order == 0.: 
        raise ValueError("order must be > 0")

    # Initialization
    efp = x[1:,:]
    ebp = x[:-1,:]

    # Initial error
    """ approach 1: efficient for small data. 
        This creates an intermediate array for the element-wise product before summing, 
        which may consume additional memory and processing time for large arrays.
    """
    #E = np.sum(x**2, axis=0) / N
    """approach 2: more efficient"""
    E=np.einsum('ij,ij->j', x, x) / N

    a = np.zeros((order+1,obs))
    a[0,:] = 1

    ref = np.zeros((order,obs))

    for m in range(order):
        # Calculate the next order reflection (parcor) coefficient
        """ approach 1: very inefficient"""
        #k = (-2 * np.diag(np.dot(ebp.T, efp))) / (np.diag(np.dot(efp.T, efp)) + np.diag(np.dot(ebp.T, ebp)))
        """ approach 2: efficient for small data. 
            This creates an intermediate array for the element-wise product before summing, 
            which may consume additional memory and processing time for large arrays.
        """
        #k = (-2 * np.sum(ebp*efp,axis=0)) / (np.sum(efp*efp,axis=0) + np.sum(ebp*ebp,axis=0))
        """approach 3: more efficient"""
        k = (-2 *  np.einsum('ij,ij->j', ebp, efp)) / ( np.einsum('ij,ij->j', efp, efp) +  np.einsum('ij,ij->j', ebp, ebp))
        
        ref[m,:] = k
        
        # Update the forward and backward prediction errors
        ef = efp[1:,:] + k * ebp[1:,:]
        ebp = ebp[:-1,:] + np.conj(k) * efp[:-1,:]
        efp = ef

        # Update the AR coeff.
        if m==0:
            a[1,:] = a[1,:] + k * np.conj(a[0,:])
        else:
            a[1:m+2,:] = a[1:m+2,:] + k * np.conj(a[m::-1,:])

        # Update the prediction error
        E = (1 - np.conj(k) * k) * E

    a=a[1:,:]
    # Assign outputs
    return a, E, ref

def arburg_warped_vector( x, order=1, warp_factor=0):
    """
    calculates Frequency-warped LPC (Linear Predictive Coding) coefficients from a vector of time series data.
    
    Input:
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
    
    Output:
    - a: _numpy.ndarray_
        - Vector of LPC coefficients [a1, a2, ...] (without the leading 1).
        - Shape: 1 x order
    - E: _float_
        - Squared of error/Noise power (sigma^2)
    - ref: _numpy.ndarray_
        - Reflection coefficients.
    """

    N = len(x)

    #1.  Calculate reflection coefficient
    # initialize
    E = np.dot(x, x) / N
    ref = np.zeros((order))
    ebp = x.copy()
    efp = x.copy()   
    # recursion
    for index in range(1, order + 1):
        #bb = np.zeros((N - l, 1))
        bb = np.zeros(N - index)
        bb[0] = ebp[0] - warp_factor * ebp[1]
        for i in range(1, N - index):
            bb[i] = ebp[i] - warp_factor * (ebp[i + 1] - bb[i - 1])
        F = efp[1:]

        # Calculate the next order reflection coefficient
        k = (-2 * np.dot(F.T, bb)) / (np.dot(F.T, F) + np.dot(bb.T, bb))
        ref[index - 1] = k

        # Update the forward and backward prediction errors
        efp = F + k * bb
        ebp = bb + k * F
        
        # Update the prediction error
        E = (1 - np.conj(k) * k) * E

    #2. calculate LPC coefficient
    # Initialize
    a = np.array([1])
    # Recursion
    for index in range(order):
        aa = np.concatenate((a, [0]))
        #J = np.fliplr(np.diag(np.ones(len(aa))))
        J = np.eye(len(aa))[::-1] 
        a = aa + ref[index] * np.dot(J, aa)
    a=a[1:] # remove 1st 1
    return a, E, ref

def arburg_warped_matrix(X, order=1, warp_factor=0):
    """
    calculates Frequency-warped LPC (Linear Predictive Coding) coefficients for multiple time series data.

    Input:
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

    Output:
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

    """
    N,m=X.shape
    
    #1.  Calculate reflection coefficient
    # initialize
    E=np.einsum('ij,ij->j', X, X) / N
    ref = np.zeros((order,m)) # each colum is one obs
    ebp = X.copy()
    efp = X.copy()
    # recursion
    for index in range(1, order + 1):
        bb = np.zeros((N - index,m))
        bb[0] = ebp[0] - warp_factor * ebp[1]
        for i in range(1, N - index):
            bb[i] = ebp[i] - warp_factor * (ebp[i + 1] - bb[i - 1])
        F = efp[1:]

        # Calculate the next order reflection coefficient
        k = (-2 * np.einsum('ij,ij->j', F, bb)) / (np.einsum('ij,ij->j', F, F) + np.einsum('ij,ij->j', bb, bb))
        ref[index - 1] = k

        # Update the forward and backward prediction errors
        efp = F + k * bb
        ebp = bb + k * F
        
        # Update the prediction error
        E = (1 - np.conj(k) * k) * E

    #2. calculate LPC coefficient
    lpc_matrix=np.zeros((order+1,m)) # each colum is one obs
    """
    # calculate lpc for each obs
    for obs in range(m):
        curr_k= ref[:,obs]
        # Initialize
        a = np.array([1])
        # Recursion
        for index in range(order):
            aa = np.concatenate((a, [0]))
            #J = np.fliplr(np.diag(np.ones(len(aa))))
            J = np.eye(len(aa))[::-1] 
            a = aa + curr_k[index] * np.dot(J, aa)
        lpc_matrix[:,obs]=a
    """
    #"""
    # do above without outer for loop
    a=np.ones((1, m))
    for index in range(order):
            aa = np.row_stack((a, np.zeros((1, m))))
            J = np.eye(aa.shape[0])[::-1] 
            a = aa + ref[index] * np.dot(J, aa)
    lpc_matrix=a
    #"""

    lpc_matrix=lpc_matrix[1:,:] # remove 1st 1
    return lpc_matrix, E, ref
