from __future__ import annotations
from scipy.stats import hmean
import numpy as np
adapt = np.array([58.07, 60.32])
general = np.array([24.1, 27.3])
print(np.mean(adapt))
print(np.mean(general))
print(hmean([0.10096474678486393, 0.1448128797240894, 0.19]))

annotation_frames = {}
annotation_frames['scene0329_00'] = [1297]
annotation_frames['scene0565_00'] = [119, 229, 470] # 119 need to be disscussed
annotation_frames['scene0644_00'] = [330, 968, 1045, 1330]
annotation_frames['scene0207_00'] = [527, 867, 1209]