from scipy.stats import hmean
import numpy as np
adapt = np.array([58.07, 60.32])
general = np.array([24.1, 27.3])
print(np.mean(adapt))
print(np.mean(general))
print(hmean([0.1515498305046235, 6.131719635310903e-05]))