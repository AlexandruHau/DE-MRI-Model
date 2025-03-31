# DE-MRI-Model
## Alexandru-Mihai Hau

The project approaches the Dynamic Contrast-Enhanced Magnetic Resonance Imaging methods for estimating the vascular system of the region surrounding the kidney. The Extended Tofts Model is used as the pharmacokinetic model. The patient is administered intravenously a Contrast Agent Gd concentration which represents the Arterial Input Function. However, the CA concentration in the kidney is different, given by the Extended Tofts Model Equation. This project approaches the estimation of the following parameters: the transfer rate of CA from the plasma to the kidney, the plasma volume fraction, the time decay constant of the CA sample and the time offset. This has been done by implementing the Multi-Layer Perceptron MLP Architecture for an Artifical Neural Network.

### Libraries needed to run the code

```
torch 1.10.1
numpy 1.18.5
yaml 0.2.5
scikit-learn 0.23.1 
```

### Instructions to run the code

#### Data pre-processing & Input

To generate the 10,000 CA functions and parameter sets, run:
```
python3 ToftsModel.py
```

The 10,000 CA functions each of 1,500 elements and the 10,000 parameter sets are saved in:
```
data/synthetic/synthetic_curves.npy
data/synthetic/synthetic_params.npy
```
Next, the following file takes the file index number and retains the folder name in a .txt file - this is done for the training, validation and test files, as they will all be called in PyTorch:

```
python3 sort_data.py data/synthetic/synthetic_curves.npy data/synthetic/synthetic_params.npy synthetic
```

For each fold from the cross-validation, the following folders are created:

```
datasplits/synthetic/fold/test
datasplits/synthetic/fold/train
datasplits/synthetic/fold/validate
```
Each of the above folders contains a filepath to the .npy curve, hence implementing the cross-validation data division into training, testing and validation datasets. The file paths are called into batches in the DataLoader method from PyTorch during the training process.

#### Training
The actual training process starts here - the Model.py file is called in, where the MLP model is designed. The cross-validation method is also incorporated. 
```
python3 GFR_Kidneys_DL.py
```

#### Testing

For testing the network, run:
```
python3 TestModel_CrossValidation.py
```
