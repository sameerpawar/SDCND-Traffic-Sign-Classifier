import numpy as np
import matplotlib.pyplot as plt

def mean_variance_of_data(x_data = None):    
    results = {}
    if x_data is not None:
        mean_sample = np.mean(x_data, axis = 0)
        std_sample  = np.std(x_data, axis = 0)
        results['mean'] = mean_sample
        results['std']  = std_sample
    return results



def data_stats(x, y):
    results = {}
    
    # Number of samples
    n_samples = x.shape[0]
    results['num_samples'] = n_samples

    # shape of samples
    shape_samples = []
    for i in range(len(x.shape)):
        shape_samples.append(x.shape[i])
    shape_samples = np.array(shape_samples)
    results['shape_samples'] = shape_samples

    # unique classes/labels in a dataset.
    results['num_classes'] = len(np.unique(y))

    # bin count of different classes
    results['hist_classes'] = np.bincount(y)

    return results
 

