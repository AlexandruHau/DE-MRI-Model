# Import the required libraries numpy, scipy and matplotlib
# for generating the input data using the Tofts Model
import numpy as np
from scipy import interpolate
import scipy.integrate
import matplotlib.pyplot as plt
import os
import sys

# Create the Tofts Model function which takes as input the
# four parameters with the time and the Arterial Input Function
# numpy arrays

# Declare a global calibration constant for the K-value - this
# will be heavily used in the neural network as well

K_calibration = 600

def ToftsModel(params, t, AIF):

    # Assign parameter names.  Ktrans is in ml/ml Tissue/second, Tg is in s, vp is between 0 and 1 and Delta is in s
    Ktrans, Tg, vp, Delta = params

    #Interpolate the AIF to upsample by a factor of 10 to avoid errors
    f = interpolate.interp1d(t, AIF, kind='linear', bounds_error=False, fill_value=0)
    tnew = np.linspace(0, t[-1], len(t)*10)
    AIFnew = f(tnew)

    #Calculate CpKid - this is the convolution of the AIF with the VIRF g(t) which we assume is an exponential with time constant Tg, shifted by toff, with g(t<Delta)=0
    g = (tnew >= Delta) * (1/Tg) * np.exp(-1 * (tnew - Delta) / Tg)

    # Convolve impulse response with AIF, accounting for temporal resolution
    CpKid = np.convolve(AIFnew, g)[0:len(tnew)] * tnew[1] 

    vdCd = Ktrans * scipy.integrate.cumtrapz(CpKid, x=tnew, initial=0)


    #Calculate Ct, this is the sum of vdCd and vpCpKid
    Ct = vdCd + (vp * CpKid)

    plt.plot(tnew, Ct)

    return Ct

# Create now function for initializng the parameters
def InitializeParameters():

    # Initialize the volume transfer constant: Ktrans is set
    # as 0.25 in units of 1 / min converted into 1 / s
    Ktrans = 4.1e-3

    # The time offset delta is set to 2.25 s
    delta = 2.25

    # Initialize the plasma volume as well with mm^3 units
    vp = 0.5 

    # Initialize the MRT
    MRT = 5.5

    # Initialize the Tg period
    Tg = MRT - delta

    # Initialize the volume fraction 
    return Ktrans, Tg, vp, delta

def Initialize_InputParams():
    # Initialize the Ktrans parameter array of size 1000 with
    # mean value of 0.25 and standard deviation of 0.1 and 
    # convert the whole values from min^(-1) to s^(-1) and take the
    # absolute value
    Ktrans = np.abs( np.random.normal(0.25, 0.1, 100) / 60 )

    # Initialize the plasma volume in units of mm^3 
    vp = np.random.uniform(0.2, 0.8, 100)

    new_arr = np.array(np.meshgrid(Ktrans, vp)).T.reshape(-1, 2)
    # Take now the uniform random distribution for the 
    # delta variable from 1 to 3.5
    delta = np.random.uniform(1, 3.5, 10000)

    # Analyse now the distribution of the mean residence 
    # time MRT value - normal distribution of mean value 5.5
    # and standard deviation of 0.7
    MRT = np.random.normal(5.5, 0.7, 10000)

    # Now calculate the exponential decay time constant Tg
    Tg = MRT - delta 

    # NEW IMPLEMENTATION
    # Declare a calibration factor for the K parameter ->
    # switch from units of 1e-3 to units of 1
    # Return the following values: the Ktrans parameter, 
    # the exponential time decay constant Tg, the plasma 
    # volume, and the delta offset time
    params = np.zeros((4, 10000))
    params[0] = new_arr[:,0] * K_calibration
    params[1] = Tg
    params[2] = new_arr[:,1] 
    params[3] = delta 
    return params.T

def main():
    
    # Load the two numpy arrays
    t = np.load('t.npy')
    AIF = np.load('AIF.npy')

    # f = interpolate.interp1d(t, AIF, kind='linear', bounds_error=False, fill_value = 0)
    tnew = np.linspace(0, t[-1], len(t)*10)
    # AIFnew = f(tnew)
    np.save('tnew.npy', tnew)

    # Initialize the array of (1000, 4) size with
    # the parameters to work out all the convoluted curves
    params = InitializeParameters()
    curve = ToftsModel(params, t, AIF)

    print(curve.shape)
    print(tnew.shape)

    # plt.plot(AIF, color='b')
    plt.plot(tnew, curve, color='red')
    plt.show()

    params = Initialize_InputParams()

    # Now initialize an empty array of size (1000, 150) and
    # each line corresponds to a different TOFTS curve of 
    # parameters suggested by the previous params array
    G = np.zeros((10000, 1500))
    for i in range(10000):
        
        # Calibrate back to the original scale for the K-value
        params[i][0] = params[i][0] / K_calibration

        # Now perform the Tofts Model curve
        G[i] = ToftsModel(params[i], t, AIF)

        # Go back to the new scale
        params[i][0] = params[i][0] * K_calibration

    # Save now the two arrays in .npy format
    np.save("data/synthetic/synthetic_curves.npy", G)
    np.save("data/synthetic/synthetic_params.npy", params)

main()