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
from Code.projs.asset_allocate.dataload import db_rtn_data, db_date_data
from Code.Utils.Sequence import strided_slicing_w_residual
from Code.BackTester.BT_AssetAllocate import rtn_multi_periods

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

asset_allocate_app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)


# mean-variance optimal
def load_data_mvopt(low_constraints, high_constraints, rtn_data_loader:Callable, **data_kwargs):
    rtn_data, assets_inds = rtn_data_loader(**data_kwargs)
    cov_mat = np.cov(rtn_data)
    rtn_rates = rtn_data.mean(axis=1)
    if low_constraints is not None:
        low_constraints = np.array(low_constraints)
    if high_constraints is not None:
        high_constraints = np.array(high_constraints)
    finmodel = MeanVarOpt(rtn_rates, cov_mat, low_constraints, high_constraints, assets_inds)
    return finmodel

def get_mvopt(low_constraints, high_constraints, rtn_data, assets_inds):
    cov_mat = np.cov(rtn_data)
    rtn_rates = rtn_data.mean(axis=1)
    if low_constraints is not None:
        low_constraints = np.array(low_constraints)
    if high_constraints is not None:
        high_constraints = np.array(high_constraints)
    finmodel = MeanVarOpt(rtn_rates, cov_mat, low_constraints, high_constraints, assets_inds)
    return finmodel

def mvopt_portf_var_from_r(r:np.float32, low_constraints, high_constraints, rtn_data, assets_inds):
    try:
        mvopt = get_mvopt(low_constraints, high_constraints, rtn_data, assets_inds)
        if low_constraints is None and high_constraints is None:
            var_star, weights_star = mvopt.get_portf_var_from_r(r)
            res = {'portf_w':list(weights_star), 'portf_var':var_star, 'portf_r':r,
                   'assets_inds':mvopt.assets_inds}
        else:
            # {"portf_w":np_array, "portf_var":float,
            #  "portf_r":float,    "qp_status":string}
            solution = mvopt.solve_constrained_qp_from_r(r)
            res = {'portf_w':list(solution['portf_w']), 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
                   "qp_status":solution['qp_status'], 'assets_inds':mvopt.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    return res



def mvopt_portf_r_from_var(var:np.float32, low_constraints, high_constraints, rtn_data, assets_inds):
    try:
        mvopt = get_mvopt(low_constraints, high_constraints, rtn_data, assets_inds)
        if low_constraints is None and high_constraints is None:
            r_star, weights_star = mvopt.get_portf_r_from_var(var)
            res = {'portf_w':list(weights_star), 'portf_var':var, 'portf_r':r_star, 
                   'assets_inds':mvopt.assets_inds}
        else:
            # {"portf_w":np_array, "portf_var":float,
            #  "portf_r":float,    "qp_status":string}
            solution = mvopt.solve_constrained_qp_from_var(var)
            res = {'portf_w':list(solution['portf_w']), 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
                   "qp_status":solution['qp_status'], 'assets_inds':mvopt.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
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
        int(inputs['gapday']), int(inputs['back_window_size']), len(inputs['assets_idx'])
    
    # 取数据，一次io解决
    # 取出2000-01-01至终止日, 所有的交易日期, 以及对应的row_indx
    all_mkt_dates = db_date_data('20000101', termidate) # 返回numpy of int
    # 涉及到的最早的date，是begindate往前数 back_window_size 个交易日的日期。因为begindate当日早上完成调仓，需要前一天至前back_window_size天
    begindate_idx = np.where(all_mkt_dates==int(begindate))[0].item()
    earlistdate_idx = begindate_idx-back_window_size
    if earlistdate_idx < 0:
        raise ValueError('Traceback earlier than 2000-01-01 from {begindate} going back with {back_window_size} days'\
                         .format(begindate=begindate, back_window_size=back_window_size))
    # 裁剪掉前面无用的日期
    all_mkt_dates = all_mkt_dates[earlistdate_idx:]
    earlistdate = all_mkt_dates[0] # 最早天数是第一天
    begindate_idx -= earlistdate_idx #
    earlistdate_idx = 0
    # all_rtn_data shape: (num_assets, begindate - back_window_size to begindate to termidate)
    # which is back_window_size + num_period_days_from_begin_to_termi
    all_rtn_data, assets_inds = db_rtn_data(assets=inputs['assets_idx'], startdate=str(earlistdate),\
                                            enddate=termidate, rtn_dilate=inputs['rtn_dilate'])
    assert all_rtn_data.shape[1] == len(all_mkt_dates),\
        'Trade dates with length {mkt_len} and Index market return data {tr_len} mismatch'.format(mkt_len=len(all_mkt_dates),tr_len=rtn_data.shape[1])
    rtn_data = all_rtn_data[:, begindate_idx:] # 从 all_rtn_data 中，取出 begindate到termidate的列
    portf_w_list, res_list = [[1/num_assets,]*num_assets, ], []
    # 每一期持仓起始，往后持仓gapday天
    strided_slices, rsd_slices = strided_slicing_w_residual(rtn_data.shape[1], gapday, gapday)
    hold_rtn_mat_list = list(rtn_data.T[strided_slices].transpose(0,2,1))
    if rsd_slices is not None:
        hold_rtn_mat_list.append( rtn_data.T[rsd_slices].T )
    # 每一期调仓日期起始，往前回溯back_window_size天。调仓日期在持仓日之前
    strided_slices, _ = strided_slicing_w_residual(all_rtn_data.shape[1]-1, back_window_size, gapday)
    train_rtn_mat_list = list(all_rtn_data.T[strided_slices].transpose(0,2,1))
    assert len(train_rtn_mat_list) == len(hold_rtn_mat_list), 'train & hold period mismatch error. Please check code'
    for train_rtn_mat in train_rtn_mat_list:
        # train_rtn_mat shape: (num_assets, back_window_size)
        res = mvopt_portf_var_from_r(inputs['expt_rtn_rate'], low_constraints, high_constraints, train_rtn_mat, assets_inds)
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
    BT_res = rtn_multi_periods(portf_w_list[1:], hold_rtn_mat_list)
    # {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float}
    # 修正
    BT_res['rtn'] = BT_res['rtn']/int(inputs['rtn_dilate'])
    BT_res['gross_rtn'] = BT_res['gross_rtn']/int(inputs['rtn_dilate'])
    return {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:]}