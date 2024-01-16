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
from Code.DataLoader.remoteDB import db_rtn_data, db_date_data
from Code.Utils.DateTime import yld_natural_dates
from Code.Utils.Sequence import strided_slicing_w_residual, strided_indexing_w_residual
from Code.BackTester.BT_AssetAllocate import rtn_multi_periods

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


@asset_allocate_app.route('/asset_allocate/BT_mvopt_var_from_r', methods=['POST'])
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
    
    begindate, termidate, gapday, back_window_size, num_assets = inputs['begindate'], inputs['termidate'], \
        int(inputs['gapday']), inputs['back_window_size'], len(inputs['assets_idx'])
    portf_w_list, res_list = [[1/num_assets,]*num_assets, ], []
    # rtn_data shape: (num_assets, num_trade_days)
    rtn_data, _ = db_rtn_data(assets=inputs['assets_idx'], startdate=begindate, enddate=termidate, rtn_dilate=inputs['rtn_dilate'])
    # creating return rate matrices for assets from begin_date to termi_date on every gap_days
    strided_slices, rsd_slices = strided_slicing_w_residual(rtn_data.shape[1], gapday, gapday)
    period_rtn_mat_list = list(rtn_data.T[strided_slices].transpose(0,2,1))
    if rsd_slices is not None:
        period_rtn_mat_list.append( rtn_data.T[rsd_slices].T )
    # all market-open dates
    all_mkt_dates = db_date_data(inputs['assets_idx'], begindate, termidate).to_numpy()
    strided_inds, rsd_ind = strided_indexing_w_residual(len(all_mkt_dates), gapday, gapday)
    if rsd_ind is not None:
        strided_inds.append(rsd_ind)

    for trade_dt in all_mkt_dates[strided_inds]:
        # trade_dt是begindate开始，到termidate为止或之前，每gapday的日期。即交易日期。
        startdate = datetime.strptime(str(trade_dt), "%Y%m%d") + timedelta(days= -back_window_size)
        enddate = datetime.strptime(str(trade_dt), "%Y%m%d") + timedelta(days= -1)
        res = mvopt_portf_var_from_r(inputs['expt_rtn_rate'], low_constraints, high_constraints, 
                                     rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                     startdate=startdate.strftime("%Y%m%d"), enddate=enddate.strftime("%Y%m%d"),
                                     rtn_dilate=inputs['rtn_dilate'])
        # 求解失败: {"err_msg":str(e), 'status':'fail'}
        # 无约束求解: {'portf_w':list, 'portf_var':float, 'portf_r':float, 'assets_inds':list}
        # 带约束求解:  {'portf_w':list, 'portf_var':float, 'portf_r':float,"qp_status":'optimal'/'unknown','assets_inds':list}
        if 'portf_w' in res:
            # dilate修正
            res['portf_std'] = np.sqrt(res['portf_var'])
            res['portf_var'] = res['portf_var'] / np.power( ( int(inputs['rtn_dilate']) ), 2)
            res['portf_std'] = res['portf_std'] / int(inputs['rtn_dilate'])
            res['portf_r'] = res['portf_r'] / int(inputs['rtn_dilate'])
        res_list.append(res)
        if ('portf_w' in res): # 当有解
            if 'qp_status' not in res: # 若非二次规划
                portf_w_list.append(res['portf_w'])
            elif res['qp_status'] == 'optimal': # 若二次规划最优
                portf_w_list.append(res['portf_w'])
            else:
                portf_w_list.append(portf_w_list[-1])
        else:
            # 否则，延用上一期的配置
            portf_w_list.append(portf_w_list[-1])
    BT_res = rtn_multi_periods(portf_w_list[1:], period_rtn_mat_list)
    # {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float}
    # 修正
    BT_res['rtn'] = BT_res['rtn']/int(inputs['rtn_dilate'])
    BT_res['gross_rtn'] = BT_res['gross_rtn']/int(inputs['rtn_dilate'])
    return {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:]}




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