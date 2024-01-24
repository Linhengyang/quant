from flask import request, Blueprint
import warnings
import numpy as np
from datetime import datetime
from markupsafe import escape
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True
# from werkzeug.middleware.proxy_fix import ProxyFix
# import json
# import re
from typing import Callable
from Code.Allocator.MeanVarOptimal import MeanVarOpt
from Code.projs.asset_allocate.dataload import db_rtn_data, db_date_data
from Code.Utils.Sequence import strided_slicing_w_residual
from Code.BackTester.BT_AssetAllocate import rtn_multi_periods, modify_BackTestResult
from Code.projs.asset_allocate.benchmark import get_benchmark_rtn_data, BackTest_benchmark, parse_benchmark

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

# asset_allocate_app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)
mvopt_api = Blueprint('mean_var', __name__)

# mean-variance optimal

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
            res = {'portf_w':solution['portf_w'], 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
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
            res = {'portf_w':solution['portf_w'], 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
                   "qp_status":solution['qp_status'], 'assets_inds':mvopt.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    return res


def get_train_rtn_data(begindate, termidate, gapday, back_window_size, dilate, assets_inds):
    # 取数据，一次io解决
    # 取出2000-01-01至终止日, 所有的交易日期，已排序
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
    all_rtn_data, assets_inds = db_rtn_data(assets=assets_inds, startdate=str(earlistdate),enddate=termidate, rtn_dilate=dilate)
    assert all_rtn_data.shape[1] == len(all_mkt_dates),\
        'market dates with length {mkt_len} and Index return dates {index_len} mismatch'.\
            format(mkt_len=len(all_mkt_dates),index_len=all_rtn_data.shape[1])
    rtn_data = all_rtn_data[:, begindate_idx:] # 从 all_rtn_data 中，取出 begindate到termidate的列
    # 每一期持仓起始，往后持仓gapday天或最后一天
    strided_slices, _, last_range = strided_slicing_w_residual(rtn_data.shape[1], gapday, gapday)
    hold_rtn_mat_list = list(rtn_data.T[strided_slices].transpose(0,2,1))
    if list(last_range): # rsd_range不为空
        hold_rtn_mat_list.append( rtn_data.T[last_range].T )
    # 每一期调仓日期起始，往前回溯back_window_size天。调仓日期在持仓日之前
    strided_slices, _, _ = strided_slicing_w_residual(all_rtn_data.shape[1]-1, back_window_size, gapday)
    train_rtn_mat_list = list(all_rtn_data.T[strided_slices].transpose(0,2,1))
    assert len(train_rtn_mat_list) == len(hold_rtn_mat_list), 'train & hold period mismatch error. Please check code'
    return train_rtn_mat_list, hold_rtn_mat_list, assets_inds



def BackTest_mvopt(expt_tgt_value, solver_func:Callable, dilate, begindate, termidate,  
                   train_rtn_mat_list, hold_rtn_mat_list, assets_inds, low_constraints, high_constraints):
    num_assets = len(assets_inds)
    portf_w_list, res_list = [ np.array([1/num_assets,]*num_assets), ], [] # 初始化为平均分配
    for train_rtn_mat in train_rtn_mat_list:
        # train_rtn_mat shape: (num_assets, back_window_size)
        # solver_func pair with expt_tgt_value: mvopt_portf_var_from_r with expt_r, mvopt_portf_r_from_var with expt_var
        res = solver_func(expt_tgt_value, low_constraints, high_constraints, train_rtn_mat, assets_inds)
        # 求解失败: {"err_msg":str(e), 'status':'fail'}
        # 无约束求解: {'portf_w':np_array, 'portf_var':float, 'portf_r':float, 'assets_inds':list}
        # 带约束求解:  {'portf_w':np_array, 'portf_var':float, 'portf_r':float,"qp_status":'optimal'/'unknown','assets_inds':list}
        if 'portf_w' in res:
            # dilate修正
            res['portf_std'] = np.sqrt(res['portf_var'])
            res['portf_var'] = res['portf_var'] / np.power(dilate, 2)
            res['portf_std'] = res['portf_std'] / dilate
            res['portf_r'] = res['portf_r'] / dilate
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
    # 回测
    BT_res = rtn_multi_periods(portf_w_list[1:], hold_rtn_mat_list)
    # BT_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float, 'annual_rtn':float}
    # 回测结果dilate修正
    BT_res = modify_BackTestResult(BT_res, dilate, begindate, termidate)
    return {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:], 'assets_id':assets_inds}







@mvopt_api.route('/asset_allocate/mean_var_opt', methods=['POST'])
def mvopt():
    inputs = request.json
    assets_info = inputs["assets_info"] # assets_info
    num_assets = len(assets_info)
    # 资产信息: id, 下限，上限，类别
    assets_inds, low_constraints, high_constraints, assets_categs = [], [], [], []
    for asset in assets_info:
        assets_inds.append(asset['id'])
        if 'category' in asset and asset['category'] != '':
            assets_categs.append(asset['category'])
        else:
            assets_categs.append(None)
        if asset["lower_bound"] != '':
            low_constraints.append(asset['lower_bound'])
        else:
            low_constraints.append(None)
        if asset['upper_bound'] != '':
            high_constraints.append(asset['upper_bound'])
        else:
            high_constraints.append(None)
    # 持仓起始日，持仓终结日，调仓频率，回看窗口天数，膨胀系数
    begindate, termidate, gapday, back_window_size, dilate = inputs['begindate'], inputs['termidate'],\
        int(inputs['gapday']), int(inputs['back_window_size']), int(inputs['rtn_dilate'])
    print('Back Test for mean-variance-optimal strategy from {begindate} to {termidate} trading in every {gapday} upon \
          assets {assets}'.format(begindate=begindate, termidate=termidate, gapday=gapday, assets=assets_inds))
    # 下限/上限
    if low_constraints == [None] * num_assets: # 假如 low_constraints 全部是 None即完全空输入
        low_constraints = None
    else: # 将个别空输入设为下限-1000
        low_constraints = [lower_bound if lower_bound is not None else -1000 for lower_bound in low_constraints]

    if high_constraints == [None] * num_assets: # 假如 high_constraints 全部是 None即完全空输入
        high_constraints = None
    else: # 将个别空输入设为上限 1000 
        high_constraints = [upper_bound if upper_bound is not None else 1000 for upper_bound in high_constraints]
    # 获取训练和持仓数据
    train_rtn_mat_list, hold_rtn_mat_list, assets_inds = get_train_rtn_data(begindate, termidate,\
                                                                            gapday, back_window_size, dilate, assets_inds)
    # 根据目标求解
    mvo_target = inputs['mvo_target']
    expt_tgt_value = inputs['expt_tgt_value'] # expt_tgt_value 可以是预期收益率也可以是预期方差
    if mvo_target == "minWave": # 当目标是给定预期收益率，最小化波动时
        solver_func = mvopt_portf_var_from_r
    elif mvo_target == "maxReturn": # 当目标是给定预期方差，最大化收益率时
        solver_func = mvopt_portf_r_from_var
    elif mvo_target == "sharp": # 当目标是最大化sharp ratio时
        raise NotImplementedError('sharp ratio maximized not implemented yet')
    else:
        raise ValueError("wrong target code. must be one of minWave, maxReturn, sharp")
    # mvopt_res = {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:], 'assets_id':assets_inds}
    mvopt_res = BackTest_mvopt(expt_tgt_value, solver_func, dilate, begindate, termidate,
                               train_rtn_mat_list, hold_rtn_mat_list, assets_inds, low_constraints, high_constraints)
    
    # benchmark求解
    bm_asset_inds, bm_weights, bm_rebal_gapday = parse_benchmark(inputs['benchmark'])
    bm_hold_rtn_mat_list, bm_asset_inds = get_benchmark_rtn_data(begindate, termidate, bm_asset_inds, dilate, bm_rebal_gapday)
    bm_bt_res = BackTest_benchmark(begindate, termidate, bm_hold_rtn_mat_list, dilate, bm_asset_inds, bm_weights)
    # bm_bt_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float, 'annual_rtn':float}

    mvopt_res['benchmark'] = bm_bt_res
    mvopt_res['excess'] = {'rtn':mvopt_res['backtest']['rtn'] - bm_bt_res['rtn'],
                           'annual_rtn':mvopt_res['backtest']['annual_rtn'] - bm_bt_res['annual_rtn']}
    
    # output mvopt_res =
    # {
    # 'details':res_list, 'weights':portf_w_list, 'assets_id':assets_inds,
    # 'backtest':{'rtn':, 'trade_days':, 'total_cost':, 'gross_rtn':, 'annual_rtn':},
    # 'benchmark':{'rtn':, 'trade_days':, 'total_cost':, 'gross_rtn':, 'annual_rtn':},
    # 'excess':{'rtn':, 'annual_rtn':}
    # }
    return mvopt_res