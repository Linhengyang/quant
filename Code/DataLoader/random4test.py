import numpy as np
np.random.seed(100)












# return rate data loader
# must return : 
# 1 np.array type data with shape (num_assets, trade_days)
# 2 list type indices indicating the indices of assets

def rdm_rtn_data(num_assets:int, back_window_size:int):
    rtn_data = np.random.uniform(low=-10, high=10, size=(num_assets, back_window_size)) # size=(num_assets, trade_days)
    assets_inds = [str(i) for i in range(num_assets)]
    return rtn_data, assets_inds
