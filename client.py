import requests
import numpy as np


# user_info = {'name':['lhy', 'linhengyang'], 'password':'123'} # 表单数据 form data
# json_data = {'a':1, 'b':2} # json数据 json data
# r = requests.post('http://127.0.0.1:8000/register', data=user_info) # post data
# print(r.text)



inputs = {
          'num_assets':5, 'back_window_size':180,
          'expt_rtn_rate':0.05,
          'expt_var':5.5,
          'view_pick_mat':[[1, -1, 0, 0, 0],
                           [0, 0, 1, -0.5, -0.5],
                           [0, 0, 0, 1, 0]],
          'view_rtn_vec': [0.01, 0.03, 0.08],
          'risk_avers_factor':0.3,
          'equi_wght_vec':[0.1, 0.3, 0.2, 0.15, 0.25],
          'tau':0.05,
          'low_constraints':[0.1, 0, 0.1, 0, 0.05], 'high_constraints':[0.2, 0.4, 1.0, 0.2, 0.2],
          'category_mat':[[1,1,0,0,0],
                          [0,0,1,1,0],
                          [0,0,0,0,1]],
          'tgt_contrib_ratio':[0.5, 0.3, 0.2],
          'assets_idx':['000001.SH', '000016.SH', '000002.SH', '000009.SH', '000010.SH'],
          'startdate':'20230201',
          'enddate':'20230301',
          'rtn_dilate':100
          }
r = requests.post("http://127.0.0.1:8000/mvopt_var_from_r", json=inputs) # post data
print(r.text)
