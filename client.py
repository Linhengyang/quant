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
          # "low_constraints":[-10, -10, -10, -10, -10], "high_constraints":[20, 20, 20, 20, 20],
        #   "category_mat":[[1,1,0,0,0],
        #                   [0,0,1,1,0],
        #                   [0,0,0,0,1]],
          # "tgt_contrib_ratio":[0.5, 0.3, 0.2],
          # "assets_idx":["000001.SH", "000016.SH", "000002.SH", "000009.SH", "000010.SH"],
          "mvo_target":"minWave", # minWave: get min var from given r; maxReturn: get max r from given var; sharp: max shap ratio
          "expt_tgt_value":0.3,
          "rtn_dilate":100,
          "begindate":"20230301",
          "termidate":"20230309",
          "gapday":2,
          "back_window_size":30,
          "benchmark":"CSI800",
          "assets_info":[
              {
                "id":"000001.SH",
                "lower_bound":-10,
                "upper_bound":20,
                "category":"index",
                "asset_risk_ratio":0.4
              },
              {
                "id":"000016.SH",
                "lower_bound":-10,
                "upper_bound":20,
                "category":"index",
                "asset_risk_ratio":0.1
              },
              {
                "id":"000002.SH",
                "lower_bound":-10,
                "upper_bound":20,
                "category":"index",
                "asset_risk_ratio":0.2
              },
              {
                "id":"000009.SH",
                "lower_bound":-10,
                "upper_bound":20,
                "category":"index",
                "asset_risk_ratio":0.1
              },
              {
                "id":"000010.SH",
                "lower_bound":-10,
                "upper_bound":20,
                "category":"index",
                "asset_risk_ratio":0.2
              },
          ]
          }
# r = requests.post("http://127.0.0.1:8000/asset_allocate/BT_mvopt_var_from_r", json=inputs) # post data
r = requests.post("http://127.0.0.1:8000/asset_allocate/risk_manage", json=inputs) # post data
print(r.text)
