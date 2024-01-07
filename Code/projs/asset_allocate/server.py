from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
import json
import re
import warnings
from datetime import datetime, timedelta
from markupsafe import escape
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True

from Code.projs.asset_allocate.functions import *
from Code.DataLoader.random4test import rdm_rtn_data
from Code.DataLoader.remoteDB import db_rtn_data
from Code.Utils.DateTime import yld_series_dates

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

asset_allocate_app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)


@asset_allocate_app.route('/mvopt_var_from_r', methods=['POST'])
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


@asset_allocate_app.route('/BT_mvopt_var_from_r', methods=['POST'])
def BT_mvopt_var_from_r():
    print('[{}]'.format(datetime.now()))
    inputs = request.json
    print('Back Test for mean-variance-optimal strategy from {begindate} to {termidate} trading in every {gapday} upon \
          assets {assets}'.format(
        begindate=inputs['begindate'], termidate=inputs['termidate'], gapday=inputs['gapday'], assets=inputs['assets_idx']
    ))
    print('input params: ', inputs)
    if 'low_constraints' in inputs:
        low_constraints = inputs['low_constraints']
    else:
        low_constraints = None
    if 'high_constraints' in inputs:
        high_constraints = inputs['high_constraints']
    else:
        high_constraints = None
    
    portf_w_list, period_rtn_mat_list = [], []
    res_list = []
    begindate, termidate, gapday, back_window_size = inputs['begindate'], inputs['termidate'], \
        int(inputs['gapday']), inputs['back_window_size']
    for trade_dt in yld_series_dates(begindate, termidate, gapday):
        # trade_dt是begindate开始，到termidate为止或之前，每gapday的日期。即交易日期。
        startdate = datetime.strptime(trade_dt, "%Y%m%d") + timedelta(days= -back_window_size)
        enddate = datetime.strptime(trade_dt, "%Y%m%d") + timedelta(days= -1)
        res = mvopt_portf_var_from_r(inputs['expt_rtn_rate'], low_constraints, high_constraints, 
                                     rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                     startdate=startdate.strftime("%Y%m%d"), enddate=enddate.strftime("%Y%m%d"),
                                     rtn_dilate=inputs['rtn_dilate'])
        res_list.append(res)
    return res_list




@asset_allocate_app.route('/mvopt_r_from_var', methods=['POST'])
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



@asset_allocate_app.route('/blkltm_var_from_r', methods=['POST'])
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



@asset_allocate_app.route('/blkltm_r_from_var', methods=['POST'])
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


@asset_allocate_app.route('/riskparity', methods=['POST'])
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


@asset_allocate_app.route('/riskbudget', methods=['POST'])
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