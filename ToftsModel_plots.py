from seaborn import set_theme
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ToftsModel import *
from sklearn.metrics import r2_score

# Correlation coefficients for the cross-validation

input_params = np.zeros((10000, 4))
predicted_params = np.zeros((10000, 4))
unit_division = 2500
for fold in range(4):
    input_params[fold * unit_division : (fold + 1) * unit_division] = np.loadtxt("CrossValidationEvaluation_Params_%d_Input.txt" % fold, delimiter = ',')
    predicted_params[fold * unit_division : (fold + 1) * unit_division] = np.loadtxt("CrossValidationEvaluation_Params_%d_Predict.txt" % fold, delimiter = ',')

# Correlation coefficients for the direct test
'''
df = pd.read_csv("Parameters_GFR.csv")
input_params = df["Input params"].to_numpy().reshape(3000, 4)
predicted_params = df["Predicted params"].to_numpy().reshape(3000, 4)
'''

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

t = np.load('t.npy')
tnew = np.linspace(0, t[-1], len(t)*10)

def DoCurves():
    # Now work out the input and prediction curves from the 
    # given sets of parameters
    ToftsModel_InputCurves = np.zeros((10000, 1500))
    ToftsModel_PredictionCurves = np.zeros((10000, 1500))

    # Perform the non-linear relationship between the AIF and
    # the Ct kidney concentration - perform the linear interpolation
    # in order to upsample the time and concentration spaces
    t = np.load('t.npy')
    AIF = np.load('AIF.npy')
    f = interpolate.interp1d(t, AIF, kind='linear', bounds_error=False, fill_value=0)
    tnew = np.linspace(0, t[-1], len(t)*10)
    AIFnew = f(tnew)

    print(AIFnew.shape)
    print(tnew.shape)
    print(ToftsModel_InputCurves.shape)
    print(ToftsModel_PredictionCurves.shape)

    # Recalibration for the Ktrans parameter
    input_params[:,0] = input_params[:,0] / K_calibration
    predicted_params[:,0] = predicted_params[:,0] / K_calibration

    # Work out the concentrations now
    for i in range(10000):

        print(i)

        # Perform now the Extended Tofts Model operator
        ToftsModel_InputCurves[i] = ToftsModel(input_params[i], t, AIF)
        ToftsModel_PredictionCurves[i] = ToftsModel(predicted_params[i], t, AIF)

        if(np.isnan(np.sum(ToftsModel_PredictionCurves[i]))):
             ToftsModel_InputCurves[i] = np.zeros(1500)
             ToftsModel_PredictionCurves[i] = np.zeros(1500)

    df = pd.DataFrame({"Input curves" : ToftsModel_InputCurves.flatten(), "Predicted curves" : ToftsModel_PredictionCurves.flatten()}) 
    df.to_csv("ToftsCurves.csv")

def AnalyzeCurves():

    # Read the pandas dataframe and extract the input and the
    # prediction curves
    df = pd.read_csv("ToftsCurves.csv")
    input_curves = df["Input curves"].to_numpy().reshape(10000, 1500)
    predicted_curves = df["Predicted curves"].to_numpy().reshape(10000, 1500)

    # Now plot a histogram for the chi-squared test of the curves
    # Note: We only do subtraction, not division
    error = np.zeros(10000)
    count = 0
    for i in range(10000):
        error[i] = np.sqrt(np.sum((input_curves[i] - predicted_curves[i])**2))

        if(error[i] > 1e+6):
            error[i] = 1e+4

        if(error[i] > 50):
            print(error[i])
            print(count)
            count += 1

    print(np.max(error))
    plt.hist(np.sort(error)[:9996], bins = 100)
    plt.suptitle("Cross-validation MSE Test")
    plt.title("Distribution of losses between input and prediction")
    plt.xlabel("Loss values")
    plt.ylabel(f"Counts - {count} tests yield loss greater than 50")
    plt.show()

    # Do a graphic of 6 subplots regarding the input vs predicted curve
    fig, axis = plt.subplots(nrows = 2, ncols = 3, constrained_layout = True)

    for i in range(2):
        for j in range(3):

            index = np.random.randint(10000)
            test_loss = np.sqrt(np.sum((input_curves[index] - predicted_curves[index])**2))
            axis[i][j].plot(tnew, input_curves[index], label=f"T: {np.round(input_params[index], 2)}")
            axis[i][j].plot(tnew, predicted_curves[index], label=f"P: {np.round(predicted_params[index], 2)}")
            axis[i][j].set_title(f"P vs T curves of curve loss: {round(test_loss, 3)}")
            axis[i][j].legend()

    plt.show()

# Make a separate function required for plotting the AIF
# as well as the kidney concentration Ct (input and output
# functions) as functions of time
def Plot_AIF_Ct():
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

AnalyzeCurves()