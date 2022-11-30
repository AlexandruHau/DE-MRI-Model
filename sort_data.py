import numpy as np
import os
import sys

def main(args, curves_path, params_path):
    curves = np.load(args[1])
    params = np.load(args[2])
    
    dataset_length = curves.shape[0]
    training_samples = int(dataset_length * 0.8)

    for i in range(0, training_samples):
        
        np.save(os.path.join(curves_path, str(i)), curves[i])
        np.save(os.path.join(params_path, str(i)), params[i])

        f = open("datasplits/" + args[3] + "/train/curves.txt", "a")
        f.write(os.path.join(curves_path, str(i) + ".npy") + "\n")
        f.close()

        f = open("datasplits/" + args[3] + "/train/params.txt", "a")
        f.write(os.path.join(params_path, str(i) + ".npy") + "\n")
        f.close()

    for i in range(training_samples, dataset_length):
        
        np.save(os.path.join(curves_path, str(i)), curves[i])
        np.save(os.path.join(params_path, str(i)), params[i])

        f = open("datasplits/" + args[3] + "/test/curves.txt", "a")
        f.write(os.path.join(curves_path, str(i) + ".npy") + "\n")
        f.close()

        f = open("datasplits/" + args[3] + "/test/params.txt", "a")
        f.write(os.path.join(params_path, str(i) + ".npy") + "\n")
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
    os.makedirs("datasplits/" + sys.argv[3] + "/test", exist_ok=True)
    os.makedirs("datasplits/data/", exist_ok=True)
    curves_path = "data/" + sys.argv[3] + "/curves"
    params_path = "data/" + sys.argv[3] + "/params"
    os.makedirs(curves_path, exist_ok=True)
    os.makedirs(params_path, exist_ok=True)
    main(sys.argv, curves_path, params_path)