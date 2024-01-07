from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
import json
import re
import warnings
from datetime import datetime
from markupsafe import escape
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True

from Code.projs.asset_allocate.functions import *
from Code.DataLoader.random4test import rdm_rtn_data
from Code.DataLoader.remoteDB import db_rtn_data

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)


@app.route('/mvopt_var_from_r', methods=['POST'])
def application_mvopt_var_from_r():
    print('/mvopt_var_from_r')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'low_constraints' in inputs:
        low_constraints = inputs['low_constraints']
    else:
        low_constraints = None
    if 'high_constraints' in inputs:
        high_constraints = inputs['high_constraints']
    else:
        high_constraints = None
    res = mvopt_portf_var_from_r(inputs['expt_rtn_rate'], low_constraints, high_constraints,
                                 rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                 startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate'])
    print('output: ')
    pprint(res)
    return res


# @app.route('/BT_mvopt_var_from_r', methods=['POST'])
# def BT_mvopt_var_from_r():
#     print('[{}]'.format(datetime.now()))
#     inputs = request.json
#     print('Back Test for mean-variance-optimal strategy from {startdate} to {enddate} trading in every {gapday} upon assets {assets}'.format(
#         startdate=inputs['startdate'], enddate=inputs['enddate'], gapday=inputs['gapday'], assets=inputs['assets_ids']
#     ))
#     print('input params: ', inputs)
#     if 'low_constraints' in inputs:
#         low_constraints = inputs['low_constraints']
#     else:
#         low_constraints = None
#     if 'high_constraints' in inputs:
#         high_constraints = inputs['high_constraints']
#     else:
#         high_constraints = None
    
#     portf_w_list, period_rtn_mat_list = [], []
    




@app.route('/mvopt_r_from_var', methods=['POST'])
def application_mvopt_r_from_var():
    print('/mvopt_r_from_var')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'low_constraints' in inputs:
        low_constraints = inputs['low_constraints']
    else:
        low_constraints = None
    if 'high_constraints' in inputs:
        high_constraints = inputs['high_constraints']
    else:
        high_constraints = None
    res = mvopt_portf_r_from_var(inputs['expt_var'], low_constraints, high_constraints,
                                 rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                 startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate'])
    print('output: ')
    pprint(res)
    return res



@app.route('/blkltm_var_from_r', methods=['POST'])
def application_blkltm_var_from_r():
    print('/blkltm_var_from_r')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'low_constraints' in inputs:
        low_constraints = inputs['low_constraints']
    else:
        low_constraints = None
    if 'high_constraints' in inputs:
        high_constraints = inputs['high_constraints']
    else:
        high_constraints = None
    res = blkltm_portf_var_from_r(inputs['expt_rtn_rate'],
                                  inputs['view_pick_mat'], inputs['view_rtn_vec'], inputs,
                                  low_constraints, high_constraints,
                                  rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                  startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate'])
    print('output: ')
    pprint(res)
    return res



@app.route('/blkltm_r_from_var', methods=['POST'])
def application_blkltm_r_from_var():
    print('/blkltm_r_from_var')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'low_constraints' in inputs:
        low_constraints = inputs['low_constraints']
    else:
        low_constraints = None
    if 'high_constraints' in inputs:
        high_constraints = inputs['high_constraints']
    else:
        high_constraints = None
    res = blkltm_portf_r_from_var(inputs['expt_var'], 
                                  inputs['view_pick_mat'], inputs['view_rtn_vec'], inputs,
                                  low_constraints, high_constraints,
                                  rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                  startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate'])
    print('output: ')
    pprint(res)
    return res


@app.route('/riskparity', methods=['POST'])
def application_riskparity():
    print('/riskparity')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'category_mat' in inputs:
        category_mat = inputs['category_mat']
    else:
        category_mat = None
    try:
        res = get_riskparity(category_mat,
                             #   rtn_data_loader=rdm_rtn_data,
                             #   num_assets=inputs['num_assets'], back_window_size=inputs['back_window_size']
                             rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                             startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate']
                             )
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    print('output: ')
    pprint(res)
    return res


@app.route('/riskbudget', methods=['POST'])
def application_riskbudget():
    print('/riskbudget')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'category_mat' in inputs:
        category_mat = inputs['category_mat']
    else:
        category_mat = None
    try:
        res = get_riskbudget(category_mat, inputs['tgt_contrib_ratio'],
                             #   rtn_data_loader=rdm_rtn_data,
                             #   num_assets=inputs['num_assets'], back_window_size=inputs['back_window_size']
                             rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                             startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate']
                             )
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    print('output: ')
    pprint(res)
    return res









if __name__ == "__main__":
    app.run(port=8000, debug=True)