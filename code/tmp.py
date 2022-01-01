import scipy.io as sio
import numpy as np

subject = 77
mat = sio.loadmat("/home/breezy/Downloads/s_%s_angles/S%s_E2_A1"%(str(subject), str(subject)))
glove=mat['glove']
angles=mat['angles']
i=1000
print(glove[i])
print(angles[i])


