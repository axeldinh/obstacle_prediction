import os
import numpy as np

def load_data(type='train', make_val=True):

    os.makedirs('data', exist_ok=True)

    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, 'data/data.npz')

    if not os.path.exists(data_path):
        raise NotImplementedError('No function to load the data')
    
    data = np.load(data_path, allow_pickle=True)

    if type == 'train':
        data = data['train']
        labels = data[:, 1].astype(int)
        data = data[:, 0]

    elif type == 'test':
        data = data['test']
        labels = None

    data = np.stack(data)

    return data, labels
