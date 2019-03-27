# AdjustedRandCI
Compute Adjusted Rand Index and Confidence Interval. The confidence interval is computed using central limit theorem and the exact variance of the Adjusted Rand Index.

# Dependencies 

`numpy,scipy,sklearn`

# Example
```
import numpy as np
from ari_ci import ari_ci

clusters = np.array([0,1,1,1,1,1,1,1,1,0,0,0,0])
trueclusters = np.array([0,1,1,1,1,0,0,1,1,1,0,0,0])

print(ari_ci(clusters,trueclusters,alpha = 0.05))
```
