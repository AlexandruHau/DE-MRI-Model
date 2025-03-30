# File for conducting the Bland-Altman analysis
# as an additional evaluator apart from the correlation test

# Import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# Open the file with the coefficients from the cross validation
df = pd.read_csv("CrossValidation_FinalPredictions.csv")
input_params = df["Input parameters"].to_numpy().reshape(10000, 4)
predicted_params = df["Predicted parameters"].to_numpy().reshape(10000, 4)

def Bland_Altman(x_vals, y_vals):

    # Calculate the average measurements as well as the differences
    # between the two entities
    avg_meas = 0.5 * (x_vals + y_vals)
    diff_meas = x_vals - y_vals

    # Calculate the average difference
    avg_diff = np.mean(diff_meas)
    print(f"Average difference: {avg_diff}")

    # Calculate the confidence interval for the differences - take 
    # afterwards the upper and lower bounds
    confidence = np.std(diff_meas)
    stat_factor = 1.95
    upper_bound = avg_diff + stat_factor * confidence
    lower_bound = avg_diff - stat_factor * confidence

    # Return the final values
    return avg_meas, diff_meas, avg_diff, upper_bound, lower_bound

# Declare the constants - These will be filled with values
# from the Bland-Altman plots
avg_meas = np.zeros((4, 10000))
diff_meas = np.zeros((4, 10000))
avg_diff = np.zeros(4)
upper_bound = np.zeros(4)
lower_bound = np.zeros(4) 

for i in range (4):

    # Read from the input .csv file on the results
    # from the cross-validation
    x_vals = input_params[:, i]
    y_vals = predicted_params[:, i]

    # Perform the Bland-Altman function and update the above
    # mentioned constants
    a_m, d_m, a_d, u_b, l_b = Bland_Altman(x_vals, y_vals)
    avg_meas[i] = a_m
    diff_meas[i] = d_m 
    avg_diff[i] = a_d
    upper_bound[i] = u_b 
    lower_bound[i] = l_b 

# Implement the seaborn theme style and plot the
# average measurements on x-axis and difference measurement
# on the y-axis

plt.hist(diff_meas[3], 100)
plt.show()

sns.set_theme()

fig, axs = plt.subplots(2, 2)
plt.title("Difference histograms")
axs[0, 0].hist(diff_meas[0], 100)
axs[0, 0].set_title("Differences for $ K^{trans} $")
axs[0, 0].set_ylabel("Counts")

axs[0, 1].hist(diff_meas[1], 100)
axs[0, 1].set_title("Differences for $ T_{g} $")

axs[1, 0].hist(diff_meas[2], 100)
axs[1, 0].set_title("Differences for $ v_{p} $")
axs[1, 0].set_ylabel("Counts")
axs[1, 0].set_xlabel("$ x_{p} - x_{i} $")

axs[1, 1].hist(diff_meas[3], 100)
axs[1, 1].set_title("Differences for $ \Delta $")
axs[1, 1].set_xlabel("$ x_{p} - x_{i} $")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(avg_meas[0], diff_meas[0])
axs[0, 0].set_title(f"CI for $ K_{{trans}} (s^{{-1}}) $: {np.round(avg_diff[0], 3)} $ \pm $ {np.round(1.95 * np.std(diff_meas[0]), 3)}")
axs[0, 0].axhline(y = avg_diff[0], color = 'r', label='Mean difference')
axs[0, 0].axhline(y = upper_bound[0], color = 'g', linestyle = 'dashed', label='Upper / Lower bound')
axs[0, 0].axhline(y = lower_bound[0], color = 'g', linestyle = 'dashed')
axs[0, 0].set_ylabel("Difference")
axs[0, 0].legend()

axs[0, 1].scatter(avg_meas[1], diff_meas[1])
axs[0, 1].set_title(f"CI for $ T_{{g}} (s)$: {np.round(avg_diff[1], 3)} $ \pm $ {np.round(1.95 * np.std(diff_meas[1]), 3)}")
axs[0, 1].axhline(y = avg_diff[1], color = 'r', label='Mean difference')
axs[0, 1].axhline(y = upper_bound[1], color = 'g', linestyle = 'dashed', label=f'Upper / Lower bound')
axs[0, 1].axhline(y = lower_bound[1], color = 'g', linestyle = 'dashed')
axs[0, 1].legend()

axs[1, 0].scatter(avg_meas[2], diff_meas[2])
axs[1, 0].set_title(f"CI for $ v_{{p}} $: {np.round(avg_diff[2], 3)} $ \pm $ {np.round(1.95 * np.std(diff_meas[2]), 3)}")
axs[1, 0].axhline(y = avg_diff[2], color = 'r', label='Mean difference')
axs[1, 0].axhline(y = upper_bound[2], color = 'g', linestyle = 'dashed', label=f'Upper / Lower bound')
axs[1, 0].axhline(y = lower_bound[2], color = 'g', linestyle = 'dashed')
axs[1, 0].set_ylabel("Difference")
axs[1, 0].set_xlabel("Mean value")
axs[1, 0].legend()

axs[1, 1].scatter(avg_meas[3], diff_meas[3])
axs[1, 1].set_title(f"CI for $ \Delta (s) $: {np.round(avg_diff[3], 3)} $ \pm $ {np.round(1.95 * np.std(diff_meas[3]), 3)}")
axs[1, 1].axhline(y = avg_diff[3], color = 'r', label='Mean difference')
axs[1, 1].axhline(y = upper_bound[3], color = 'g', linestyle = 'dashed', label=f'Upper / Lower bound')
axs[1, 1].axhline(y = lower_bound[3], color = 'g', linestyle = 'dashed')
axs[1, 1].set_xlabel("Mean value")
axs[1, 1].legend()
# plt.scatter(avg_meas, diff_meas)

# Plot the horizontal line
'''
plt.axhline(y = avg_diff, color = 'r', alpha = 0.9, linestyle = '-', label='Mean difference')
plt.axhline(y = upper_bound, color = 'g', alpha = 0.9, linestyle = '-', label='Upper bound')
plt.axhline(y = lower_bound, color = 'g', alpha = 0.9, linestyle = '-', label='Lower bound')
plt.legend()
'''
plt.legend()
plt.tight_layout()
plt.show()
