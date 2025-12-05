import numpy as np
import numpy.random as rnd
import random
A=np.random.random((5,10))
print(A)
print(A.shape)
A[3,5]=np.nan
print(A)

filter_nans_y = np.all(np.isfinite(A), axis=1).ravel()
print(filter_nans_y)

A[np.isnan(A)] = 0
print(A)

