# DE-MRI-Model
## Alexandru-Mihai Hau

The project approaches the estimation of the Glomerular Flow Rate (GFR) parameter of the kidney using the Multiperceptron architecture for a neural network.

### Instructions to run the code

```
python3 ToftsModel.py
python3 sort_data.py data/synthetic/synthetic_curves.npy data/synthetic/synthetic_params.py synthetic
python3 GFR_Kidneys_DL.py
python3 test_model.py
```

For checking the correlation plots between the input ground truth and the predicted parameters, run the following:

```
python3 ToftsModel_plots.py
```
