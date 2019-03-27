# AdjustedRandCI
Compute Adjusted Rand Index and Confidence Interval. The confidence interval is an approximate normal confidence interval using the exact variance of the Adjusted Rand Index. 

For details see: Steinley, Douglas, Michael J. Brusco, and Lawrence Hubert. "The variance of the adjusted Rand index." Psychological methods 21.2 (2016): 261.

# Dependencies 

`numpy,scipy,sklearn`

# Example
```
import numpy as np
from ari_ci import ari_ci

clusters = np.array([0,1,1,1,1,1,1,1,1,0,0,0,0])
trueclusters = np.array([0,1,1,1,1,0,0,1,1,1,0,0,0])

# print ARI, variance of ARI, lower CI, upper CI
print(ari_ci(clusters,trueclusters,alpha = 0.05))
```
