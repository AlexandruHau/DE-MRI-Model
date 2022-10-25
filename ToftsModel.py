# Import the required libraries numpy, scipy and matplotlib
# for generating the input data using the Tofts Model
import numpy as np
from scipy import interpolate
import scipy.integrate
import matplotlib.pyplot as plt

# Create the Tofts Model function which takes as input the
# four parameters with the time and the Arterial Input Function
# numpy arrays
def ToftsModel(params, t, AIF):

    # Assign the parameter names: 
    Ktrans, Tg, vp, delta = params 

    # Create the vasculat impulse response function taking
    # the form of an exponential where the argument of the 
    # exponential is given by the time shift (t - delta)
    imp = (1 / Tg) * np.exp(-1 * (t - delta) / Tg)
    imp[np.where(t < delta)] = 0
    
    #Interpolate the AIF to upsample by a factor of 10 to avoid errors
    '''
    f = interpolate.interp1d(t, AIF, kind='linear', bounds_error=False, fill_value=0)
    tnew = np.linspace(0, t[-1], len(t))
    AIF = f(tnew)
    '''

    # Perform now the convolution between the Arterial Input Function
    # and the vascular impulse response
    convolution = np.convolve(AIF, imp)

    # Work out the Cd term
    Cd = Ktrans * scipy.integrate.cumtrapz(convolution[0:len(t)], x=t, initial=0)

    # Adjust for the temporal resolution and add the second term.
    # Return the overall array
    # G = convolution[0:len(t)] * t[1] + vp * AIFnew
    G = convolution[0:len(t)] * t[1] + vp * Cd
    return G

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
    ve = Ktrans * Tg 
    return Ktrans, Tg, vp, delta

def Initialize_InputParams():
    # Initialize the Ktrans parameter array of size 1000 with
    # mean value of 0.25 and standard deviation of 0.1 and 
    # convert the whole values from min^(-1) to s^(-1) and take the
    # absolute value
    Ktrans = np.abs( np.random.normal(0.25, 0.1, 1000) / 60 )
    
    # Take now the uniform random distribution for the 
    # delta variable from 1 to 3.5
    delta = np.random.uniform(1, 3.5, 1000)

    # Analyse now the distribution of the mean residence 
    # time MRT value - normal distribution of mean value 5.5
    # and standard deviation of 0.7
    MRT = np.random.normal(5.5, 0.7, 1000)

    # Now calculate the exponential decay time constant Tg
    Tg = MRT - delta 

    # Initialize the plasma volume in units of mm^3 
    vp = np.random.uniform(0.2, 0.8, 1000)

    # Return the following values: the Ktrans parameter, 
    # the exponential time decay constant Tg, the plasma 
    # volume, and the delta offset time
    params = np.zeros((4, 1000))
    params[0] = Ktrans
    params[1] = Tg
    params[2] = vp 
    params[3] = delta
    return params.T

def main():
    
    # Load the two numpy arrays
    t = np.load('t.npy')
    AIF = np.load('AIF.npy')

    # Initialize the array of (1000, 4) size with
    # the parameters to work out all the convoluted curves
    params = Initialize_InputParams()

    # Now initialize an empty array of size (1000, 150) and
    # each line corresponds to a different TOFTS curve of 
    # parameters suggested by the previous params array
    G = np.zeros((1000, 150))
    for i in range(1000):
        
        G[i] = ToftsModel(params[i], t, AIF)

    # Save now the two arrays in .npy format
    np.save("data/synthetic/synthetic_curves.npy", G)
    np.save("data/synthetic/synthetic_params.npy", params)

main()