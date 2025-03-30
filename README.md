# DE-MRI-Model
## Alexandru-Mihai Hau

The project approaches the Dynamic Contrast-Enhanced Magnetic Resonance Imaging methods for estimating the vascular system of the region surrounding the kidney. The Extended Tofts Model is used as the pharmacokinetic model. The patient is administered intravenously a Contrast Agent Gd concentration which represents the Arterial Input Function. However, the CA concentration in the kidney is different, given by the Extended Tofts Model Equation. This project approaches the estimation of the following parameters: the transfer rate of CA from the plasma to the kidney, the plasma volume fraction, the time decay constant of the CA sample and the time offset. This has been done by implementing the Multi-Layer Perceptron MLP Architecture for an Artifical Neural Network.

### Instructions to run the code

To generate the 10,000 CA functions and the 1,500 parameter sets - all saved in the two files data/synthetic/synthetic_curves.npy and data/synthetic/synthetic_params.npy, run:
```
python3 ToftsModel.py
```
Next, the following file takes the file index number and retains the folder name in a .txt file - this is done for the training, validation and test files, as they will all be called in PyTorch:

```
python3 sort_data.py data/synthetic/synthetic_curves.npy data/synthetic/synthetic_params.py synthetic
```

The actual training process starts here:
```
python3 GFR_Kidneys_DL.py
```

python3 test_model.py
```

For checking the correlation plots between the input ground truth and the predicted parameters, run the following:

```
python3 ToftsModel_plots.py
```
