import numpy as np
np.random.seed(100)
def rdm_rtn_data(num_assets:int, back_window_size:int):
    return np.random.uniform(low=-10, high=10, size=(num_assets, back_window_size))
