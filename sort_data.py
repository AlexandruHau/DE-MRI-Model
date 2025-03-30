import numpy as np
from sklearn.model_selection import KFold
import os
import sys

def main(args, curves_path, params_path):
    curves = np.load(args[1])
    params = np.load(args[2])
    
    dataset_length = curves.shape[0]
    training_and_validation_samples = int(dataset_length * 0.7)

    # Define now the length of the training dataset
    training_samples = int(training_and_validation_samples * 0.7)

    # Save the paths corresponding to the training datasets
    for i in range(0, training_samples):
        
        np.save(os.path.join(curves_path, str(i)), curves[i])
        np.save(os.path.join(params_path, str(i)), params[i])

        f = open("datasplits/" + args[3] + "/train/curves.txt", "a")
        f.write(os.path.join(curves_path, str(i) + ".npy") + "\n")
        f.close()

        f = open("datasplits/" + args[3] + "/train/params.txt", "a")
        f.write(os.path.join(params_path, str(i) + ".npy") + "\n")
        f.close()

    # Save the paths corresponding to the validation datasets
    for i in range(training_samples, training_and_validation_samples):
        
        np.save(os.path.join(curves_path, str(i)), curves[i])
        np.save(os.path.join(params_path, str(i)), params[i])

        f = open("datasplits/" + args[3] + "/validate/curves.txt", "a")
        f.write(os.path.join(curves_path, str(i) + ".npy") + "\n")
        f.close()

        f = open("datasplits/" + args[3] + "/validate/params.txt", "a")
        f.write(os.path.join(params_path, str(i) + ".npy") + "\n")
        f.close()

    # Save the paths corresponding to the testing datasets
    for i in range(training_and_validation_samples, dataset_length):
        
        np.save(os.path.join(curves_path, str(i)), curves[i])
        np.save(os.path.join(params_path, str(i)), params[i])

        f = open("datasplits/" + args[3] + "/test/curves.txt", "a")
        f.write(os.path.join(curves_path, str(i) + ".npy") + "\n")
        f.close()

        f = open("datasplits/" + args[3] + "/test/params.txt", "a")
        f.write(os.path.join(params_path, str(i) + ".npy") + "\n")
        f.close()

    # Now work on the splitting of the datasets for the cross-validation
    # Choose 4 splits and therefore a (75-25)% split of the training and
    # testing datasets
    splits = KFold(n_splits = 4, shuffle = True, random_state = 42)
    
    for fold, (train_idx, test_idx) in enumerate (splits.split(np.arange(10000))):
        
        # Initializa a counter - this will be good for setting up
        # the training and validation path files
        count_fold = 0
        for train_element in train_idx:
            # Save the corresponding numpy files
            np.save(os.path.join(curves_path, str(train_element)), curves[train_element])
            np.save(os.path.join(params_path, str(train_element)), params[train_element])

            if(count_fold < 0.75 * len(train_idx)):

                # Save the corresponding numpy path file for the curves datasets
                f = open("datasplits/" + args[3] + "/fold%d/train/curves.txt" % fold, "a")
                f.write(os.path.join(curves_path, str(train_element) + ".npy") + "\n")
                f.close()

                # Save the corresponding numpy path file for the parameters datasets
                f = open("datasplits/" + args[3] + "/fold%d/train/params.txt" % fold, "a")
                f.write(os.path.join(params_path, str(train_element) + ".npy") + "\n")
                f.close()

            # Now set up the path files for the validation datasets
            else:

                # Save the corresponding numpy path file for the curves datasets
                f = open("datasplits/" + args[3] + "/fold%d/validate/curves.txt" % fold, "a")
                f.write(os.path.join(curves_path, str(train_element) + ".npy") + "\n")
                f.close()

                # Save the corresponding numpy path file for the parameters datasets
                f = open("datasplits/" + args[3] + "/fold%d/validate/params.txt" % fold, "a")
                f.write(os.path.join(params_path, str(train_element) + ".npy") + "\n")
                f.close()

            count_fold += 1

        for test_element in test_idx:
            # Save the corresponding numpy files
            np.save(os.path.join(curves_path, str(test_element)), curves[test_element])
            np.save(os.path.join(params_path, str(test_element)), params[test_element])

            # Save the corresponding numpy path file for the curves datasets
            f = open("datasplits/" + args[3] + "/fold%d/test/curves.txt" % fold, "a")
            f.write(os.path.join(curves_path, str(test_element) + ".npy") + "\n")
            f.close()

            # Save the corresponding numpy path file for the parameters datasets
            f = open("datasplits/" + args[3] + "/fold%d/test/params.txt" % fold, "a")
            f.write(os.path.join(params_path, str(test_element) + ".npy") + "\n")
            f.close()

    # Now save all the data in separate directory
    for i in range(dataset_length):

        np.save(os.path.join(curves_path, str(i)), curves[i])
        np.save(os.path.join(params_path, str(i)), params[i])

        f = open("datasplits/data/curves.txt", "a")
        f.write(os.path.join(curves_path, str(i) + ".npy") + "\n")
        f.close()

        f = open("datasplits/data/params.txt", "a")
        f.write(os.path.join(params_path, str(i) + ".npy") + "\n")
        f.close()

if __name__ == "__main__":
    os.makedirs("datasplits/" + sys.argv[3] + "/train", exist_ok=True)
    os.makedirs("datasplits/" + sys.argv[3] + "/validate", exist_ok=True)
    os.makedirs("datasplits/" + sys.argv[3] + "/test", exist_ok=True)
    os.makedirs("datasplits/data/", exist_ok=True)

    # Create directories for the five folds
    # used for the cross-validation procedure
    for fold in range(5):
        os.makedirs("datasplits/" + sys.argv[3] + "/fold%d/train" % fold, exist_ok=True)
        os.makedirs("datasplits/" + sys.argv[3] + "/fold%d/test" % fold, exist_ok=True)
        os.makedirs("datasplits/" + sys.argv[3] + "/fold%d/validate" % fold, exist_ok=True)

    curves_path = "data/" + sys.argv[3] + "/curves"
    params_path = "data/" + sys.argv[3] + "/params"
    os.makedirs(curves_path, exist_ok=True)
    os.makedirs(params_path, exist_ok=True)
    main(sys.argv, curves_path, params_path)