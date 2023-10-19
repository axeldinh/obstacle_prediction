
import numpy as np

def flatten_data(data):

    data = np.reshape(data, (len(data), -1))

    return data