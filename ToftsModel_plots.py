from seaborn import set_theme
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ToftsModel import *
from sklearn.metrics import r2_score

# Import the .csv file
df = pd.read_csv("Parameters_GFR.csv")
input_params = df["Input params"].to_numpy().reshape(3000, 4)
predicted_params = df["Predicted params"].to_numpy().reshape(3000, 4)

# Calculate the correlation coefficients r2 test scores
r1 = r2_score(input_params[:,0], predicted_params[:,0])
r2 = r2_score(input_params[:,1], predicted_params[:,1])
r3 = r2_score(input_params[:,2], predicted_params[:,2])
r4 = r2_score(input_params[:,3], predicted_params[:,3])

set_theme()
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(input_params[:,0], predicted_params[:,0])
axs[0, 0].set_title(f'Filtration rate K (1/s) recalibrated of r$^2$ score {np.round(r1, 3)}')
axs[0, 1].scatter(input_params[:,1], predicted_params[:,1])
axs[0, 1].set_title(f'Time decay constant $T_g$ (s) of r$^2$ score {np.round(r2, 3)}')
axs[1, 0].scatter(input_params[:,2], predicted_params[:,2])
axs[1, 0].set_title(f'Plasma volume fraction $v_p$ of r$^2$ score {np.round(r3, 3)}')
axs[1, 1].scatter(input_params[:,3], predicted_params[:,3])
axs[1, 1].set_title(f'Time offset $\delta$ (s) of r$^2$ score {np.round(r4, 3)}')
fig.tight_layout()
plt.show()

ToftsParams = InitializeParameters()
t = np.load('t.npy')
AIF = np.load('AIF.npy')
Ct = ToftsModel(ToftsParams, t, AIF)

f = interpolate.interp1d(t, AIF, kind='linear', bounds_error=False, fill_value=0)
tnew = np.linspace(0, t[-1], len(t)*10)
AIFnew = f(tnew)

plt.plot(tnew, AIFnew, label='CA conecntration in the artery AIF')
plt.plot(tnew, Ct, label='CA concentration in the kidney $C_t$')
plt.title(f'AIF vs CA concentrations for $(K^t (1/s), T_g (s), v_p, \Delta (s))$ = {ToftsParams}')
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.legend()
plt.show()