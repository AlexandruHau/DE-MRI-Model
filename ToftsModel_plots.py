from seaborn import set_theme
import matplotlib.pyplot as plt
import numpy as np
from ToftsModel import *

# Choose one set of parameters common for all the plots - all these are
# taken from the main Tofts Model article
K_trans = 0.25 / 60 
delta = 2.25
MRT = 5.5  
vp = 0.5  
Tg = MRT - delta  

# Load the .npy files corresponding to the time and AIF samples
t = np.load("t.npy")
AIF = np.load("AIF.npy")
tnew = np.load("tnew.npy")

# Now work on the filtration rate parameter k - choose spacings
# of 0.5 and divide it to 60 to work on the same time scale
delta_k = np.arange(5) * 0.005 / 60 - 2 * 0.005 / 60

# Keep all the other three parameters and tune only the filtration
# rate. Perform the plots to show the variation of the Tofts model
# as function of the filtration value k. Round all the values 
# to 4 digits
set_theme()
for dk in delta_k:
    K = K_trans + dk 
    G = ToftsModel(np.array([K, Tg, vp, delta]), t, AIF)
    plt.plot(tnew, G, label=f'dK: {round(dk, 4)}, K: {round(K, 4)}')

plt.title(f"Tofts Model for K: {round(K_trans, 4)}, Tg: {Tg}, vp: {vp}, delta: {delta}")
plt.legend()
plt.show()

# Work with the delta value
del_delta = np.arange(5) * 1 - 2 * 1
for d_delta in del_delta:
    D = delta + d_delta
    G = ToftsModel(np.array([K_trans, Tg, vp, D]), t, AIF)
    plt.plot(tnew, G, label=f'D: {D}')

plt.title(f"Tofts Model for K: {round(K_trans, 4)}, Tg: {Tg}, vp: {vp}, delta: {delta}")
plt.legend()
plt.show()

# Work with the Tg value
delta_Tg = np.arange(5) * 0.5 - 2 * 0.5
for d_Tg in delta_Tg:
    new_Tg = Tg + d_Tg
    G = ToftsModel(np.array([K_trans, new_Tg, vp, delta]), t, AIF)
    plt.plot(tnew, G, label=f'Tg: {new_Tg}')

plt.title(f"Tofts Model for K: {round(K_trans, 4)}, Tg: {Tg}, vp: {vp}, delta: {delta}")
plt.legend()
plt.show()

# Work with the vp value
delta_Vp = np.arange(5) * 0.2 - 2 * 0.2
for d_Vp in delta_Vp:
    new_Vp = vp + d_Vp
    G = ToftsModel(np.array([K_trans, Tg, new_Vp, delta]), t, AIF)
    plt.plot(tnew, G, label=f'vp: {round(new_Vp, 2)}')

plt.title(f"Tofts Model for K: {round(K_trans, 4)}, Tg: {Tg}, vp: {vp}, delta: {delta}")
plt.legend()
plt.show()