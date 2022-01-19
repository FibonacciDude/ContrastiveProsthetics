import scipy.io as sio
from sklearn.linear_model import LinearRegression as lr
import numpy as np
from tqdm import tqdm

for person in tqdm(range(27, 77)):
    mat=sio.loadmat("/home/breezy/Downloads/s_%s_angles/S%s_E1_A1" % (str(person+1), str(person+1)))
    stim=mat['restimulus']
    print(stim.min(), stim.max())
    print(mat['angles'][0])
    mat=sio.loadmat("/home/breezy/Downloads/s_%s_angles/S%s_E2_A1" % (str(person+1), str(person+1)))
    stim=mat['restimulus']
    print(stim.min(), stim.max())

    """
    dat=mat['angles'][:, 5]
    print(dat.shape[0], dat.itemsize*dat.size)
    if not np.isnan(dat).any():
        print("Clean")
    """
