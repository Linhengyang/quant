import requests
import numpy as np


# user_info = {"name":["lhy", "linhengyang"], "password":"123"} # 表单数据 form data
# json_data = {"a":1, "b":2} # json数据 json data
# r = requests.post("http://127.0.0.1:8000/register", data=user_info) # post data
# print(r.text)



inputs = {
        #   "num_assets":5, "back_window_size":180,
        #   "view_pick_mat":[[1, -1, 0, 0, 0],
        #                    [0, 0, 1, -0.5, -0.5],
        #                    [0, 0, 0, 1, 0]],
        #   "view_rtn_vec": [0.01, 0.03, 0.08],
        #   "risk_avers_factor":0.3,
        #   "equi_wght_vec":[0.1, 0.3, 0.2, 0.15, 0.25],
        #   "tau":0.05,
          "mvo_target":"sharpe", # minWave: get min var from given r; maxReturn: get max r from given var; sharpe: max shap ratio
          "expt_tgt_value":0.4,
          "rtn_dilate":100,
          "begindate":"20230301",
          "termidate":"20230725",
          "gapday":20,
          "back_window_size":30,
          "benchmark":"CSI800",
          "assets_info":[
              {
                "id":"000001.SH",
                "lower_bound":'0.1',
                "upper_bound":'',
                "category":"index",
                "asset_risk_ratio":0.4,
                "fixed_wght":0.2
              },
              {
                "id":"000016.SH",
                "lower_bound":'',
                "upper_bound":'',
                "category":"index",
                "asset_risk_ratio":0.1,
                "fixed_wght":0.3
              },
              {
                "id":"000002.SH",
                "lower_bound":'',
                "upper_bound":'',
                "category":"index",
                "asset_risk_ratio":0.2,
                "fixed_wght":0.15
              },
              {
                "id":"000009.SH",
                "lower_bound":'',
                "upper_bound":'',
                "category":"index",
                "asset_risk_ratio":0.1,
                "fixed_wght":0.25
              },
              {
                "id":"000010.SH",
                "lower_bound":'',
                "upper_bound":'',
                "category":"index",
                "asset_risk_ratio":0.2,
                "fixed_wght":0.1
              },
          ]
          }
r = requests.post("http://127.0.0.1:33000/asset_allocate/mean_var_opt", json=inputs) # post data
# r = requests.post("http://127.0.0.1:33000/asset_allocate/risk_manage", json=inputs) # post data
# r = requests.post("http://127.0.0.1:33000/asset_allocate/fixed_combination", json=inputs) # post data
print(r.text)
