import numpy as np
diag = np.diag([1,2,3,4,5])
left = np.random.uniform(size=(3,5))
print(left @ diag @ left.T)