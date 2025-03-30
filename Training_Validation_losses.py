# Import the following libraries: numpy, matplotlib, seaborn
# and pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

# Use pandas in order to read the data from the loss .csv files
df = pd.read_csv("Losses_Adam_0002.csv")
training = df["Training"].to_numpy()
validation = df["Validation"].to_numpy()
epochs = np.arange(3000)

# Plot the required curves - set the seaborn
# style and theme firstly
sns.set_theme()
sns.set_style()

# Now prepare two subplots - one for the whole range epoch
# and one for the loss values after epoch 200
fig, axs = plt.subplots(nrows = 2, ncols = 1)
'''
axs[0].plot(epochs, training, label="Training")
axs[0].plot(epochs, validation, label="Validation")
axs[0].set_ylabel("Losses")
axs[0].legend()
'''

axs[0].plot(epochs, training, label="Training")
axs[0].plot(epochs, validation, label="Validation")
axs[0].set_title("T vs V losses for Adam at $ \eta $ = 0.001")
axs[0].set_ylabel("Losses")
axs[0].legend()

axs[1].plot(epochs[200:500], training[200:500], label="Training")
axs[1].plot(epochs[200:500], validation[200:500], label="Validation")
axs[1].set_title("T vs V losses for Adam at $ \eta $ = 0.002 after epochs 200")
axs[1].set_ylabel("Losses")
axs[1].set_xlabel("Epochs")
axs[1].legend()

plt.tight_layout()
plt.show()