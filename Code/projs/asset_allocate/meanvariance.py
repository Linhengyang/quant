from flask import request, Blueprint
import warnings
import numpy as np
from datetime import datetime
from markupsafe import escape
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True
from typing import Callable
from Code.Allocator.MeanVarOptimal import MeanVarOpt
from Code.projs.asset_allocate.dataload import (
    get_train_rtn_data, 
    _REMOTE_DB, 
    _LOCAL_DB
)
from Code.BackTester.BT_AssetAllocate import (
    rtn_multi_periods,
    modify_BackTestResult
    )
from Code.projs.asset_allocate.benchmark import (
    get_benchmark_rtn_data,
    BackTest_benchmark,
    parse_benchmark,
    _BENCHMARK_WEIGHTS
    )

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

mvopt_api = Blueprint('mean_var', __name__)

_DB = _LOCAL_DB

# mean-variance optimal

def get_mvopt(
        low_constraints,
        high_constraints,
        rtn_data,
        assets_idlst
        ):
    cov_mat = np.cov(rtn_data)
    rtn_rates = rtn_data.mean(axis=1)

    if low_constraints is not None:
        low_constraints = np.array(low_constraints)
    if high_constraints is not None:
        high_constraints = np.array(high_constraints)
    
    finmodel = MeanVarOpt(rtn_rates, cov_mat, low_constraints, high_constraints, assets_idlst)

    return finmodel


def mvopt_portf_var_from_r(r:np.float32, low_constraints, high_constraints, rtn_data, assets_idlst):
    try:
        mvopt = get_mvopt(low_constraints, high_constraints, rtn_data, assets_idlst)
        if low_constraints is None and high_constraints is None:
            var_star, weights_star = mvopt.get_portf_var_from_r(r) # return float, np.array
            res = {'err_msg':'', 'status':'success', 'solve_status':"direct",
                   'portf_w':weights_star, 'portf_var':var_star, 'portf_rtn':r, 'portf_std':np.sqrt(var_star),
                   'assets_ids':mvopt.assets_idlst}
        else:
            solution = mvopt.solve_constrained_qp_from_r(r)
            res = {'err_msg':'', 'status':'success', 'solve_status':"qp_"+solution['qp_status'],
                   'portf_w':solution['portf_w'], 'portf_var':solution['portf_var'],
                   'portf_rtn':solution['portf_rtn'], 'portf_std':np.sqrt(solution['portf_var']),
                   'assets_ids':mvopt.assets_idlst}
    except Exception as e:
        traceback.print_exc()
        res = {'err_msg':str(e), 'status':'fail', 'solve_status':'',
               'portf_w':np.array([]), 'portf_var':-1, 'portf_rtn':0, 'portf_std':-1,
               'assets_ids':assets_idlst}
    return res



def mvopt_portf_r_from_var(var:np.float32, low_constraints, high_constraints, rtn_data, assets_idlst):
    try:
        mvopt = get_mvopt(low_constraints, high_constraints, rtn_data, assets_idlst)
        if low_constraints is None and high_constraints is None:
            r_star, weights_star = mvopt.get_portf_r_from_var(var) # return float, np.array
            res = {'err_msg':'', 'status':'success', 'solve_status':"direct",
                   'portf_w':weights_star,'portf_var':var, 'portf_rtn':r_star, 'portf_std':np.sqrt(var),
                   'assets_ids':mvopt.assets_idlst}
        else:
            solution = mvopt.solve_constrained_qp_from_var(var)
            res = {'err_msg':'', 'status':'success', 'solve_status':"qp_"+solution['qp_status'],
                   'portf_w':solution['portf_w'], 'portf_var':solution['portf_var'],
                   'portf_rtn':solution['portf_rtn'],  'portf_std':np.sqrt(solution['portf_var']),
                   'assets_ids':mvopt.assets_idlst}
    except Exception as e:
        traceback.print_exc()
        res = {'err_msg':str(e), 'status':'fail', 'solve_status':'',
               'portf_w':np.array([]), 'portf_var':-1, 'portf_rtn':0, 'portf_std':-1,
               'assets_ids':assets_idlst}
    return res




def modify_SolveResult(solve_res:dict, dilate:int):
    # input solve_res = {'err_msg':str(e), 'status':'fail', 'solve_status':'',
    #                    'portf_w':np.array([]), 'portf_var':-1, 'portf_rtn':0,
    #                    'assets_ids':mvopt.assets_ids}
    if solve_res['status'] == 'success': # 当运行成功时
        solve_res['portf_rtn'] /= dilate # rtn 膨胀系数修正
        solve_res['portf_std'] /= dilate # std 膨胀系数修正
        solve_res['portf_var'] /= (dilate*dilate) # var膨胀系数修正
    solve_res['portf_w'] = list(solve_res['portf_w']) # 将np.array转化为list，以输出json
    # output solve_res = {'err_msg':str(e), 'status':str, 'solve_status':str,
    #                     'portf_w':list, 'portf_var':float, 'portf_rtn':float, 'portf_std':float, 'portf_ann_rtn':float,
    #                     'assets_ids':list}
    return solve_res



def BackTest_mvopt(expt_tgt_value, solver_func:Callable, dilate, begindate, termidate,  
                   train_rtn_mat_list, hold_rtn_mat_list, assets_idlst, low_constraints, high_constraints):
    num_assets = len(assets_idlst)
    portf_w_list, res_list = [ np.array([1/num_assets,]*num_assets), ], [] # 初始化为平均分配
    for train_rtn_mat in train_rtn_mat_list:
        # train_rtn_mat shape: (num_assets, back_window_size)
        # solver_func pair with expt_tgt_value: mvopt_portf_var_from_r with expt_r, mvopt_portf_r_from_var with expt_var
        res = solver_func(expt_tgt_value, low_constraints, high_constraints, train_rtn_mat, assets_idlst)
        # res: {'err_msg':str, 'status':str, 'solve_status':str, 'portf_w':np.array, 'portf_var':float, 'portf_r':float, 'assets_ids':list}
        if res['status'] == 'success' and  res['solve_status'] in ('direct', 'qp_optimal'):
            # 只有当运行求解成功，且求解模式满足direct和qp_optimal时，才记录解
            portf_w_list.append(res['portf_w'])
        else: # 除此之外，延用上一期的配置
            portf_w_list.append(portf_w_list[-1])
        
        res_list.append( modify_SolveResult(res, dilate) )
    # 回测 并作 dilate修正
    BT_res = rtn_multi_periods(portf_w_list[1:], hold_rtn_mat_list)
    BT_res = modify_BackTestResult(BT_res, dilate, begindate, termidate)
    # BT_res = {'rtn':float, 'gross_rtn':float, 'annual_rtn':float, 'trade_days':int, 'total_cost':float}
    portf_w_list = [list(portf_w) for portf_w in portf_w_list] # 将np.array转化为list，以输出json
    return {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:], 'assets_ids':assets_idlst}



@mvopt_api.route('/asset_allocate/mean_var_opt', methods=['POST'])
def mvopt():
    inputs = request.json
    assets_info = inputs["assets_info"] # assets_info
    num_assets = len(assets_info)
    # 资产信息: id, 下限，上限，类别
    assets_ids, low_constraints, high_constraints, assets_categs = [], [], [], []
    for asset in assets_info:
        assets_ids.append(asset['id'])
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
          assets {assets}'.format(begindate=begindate, termidate=termidate, gapday=gapday, assets=assets_ids))
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
    train_rtn_mat_list, hold_rtn_mat_list, assets_idlst = get_train_rtn_data(begindate,
                                                                             termidate,
                                                                             gapday,
                                                                             back_window_size,
                                                                             dilate,
                                                                             assets_ids,
                                                                             'aidx_eod_prices',
                                                                             _DB
                                                                             )
    # 根据目标求解
    mvo_target = inputs['mvo_target']
    expt_tgt_value = inputs['expt_tgt_value'] # expt_tgt_value 可以是预期收益率也可以是预期方差
    if mvo_target == "minWave": # 当目标是给定预期收益率，最小化波动时
        solver_func = mvopt_portf_var_from_r
    elif mvo_target == "maxReturn": # 当目标是给定预期方差，最大化收益率时
        solver_func = mvopt_portf_r_from_var
    elif mvo_target == "sharp": # 当目标是最大化sharp ratio时
        raise NotImplementedError(
            'sharp ratio maximized not implemented yet'
            )
    else:
        raise ValueError(
            "wrong target code. must be one of minWave, maxReturn, sharp"
            )
    # mvopt_res = {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:], 'assets_id':assets_idlst}
    mvopt_res = BackTest_mvopt(expt_tgt_value,
                               solver_func,
                               dilate,
                               begindate,
                               termidate,
                               train_rtn_mat_list,
                               hold_rtn_mat_list,
                               assets_idlst,
                               low_constraints,
                               high_constraints
                               )
    
    # benchmark求解
    bm_id = inputs['benchmark']
    bm_assets_ids, bm_tbl_names, bm_rebal_gapday = parse_benchmark(bm_id)
    bm_hold_rtn_mat_list, bm_assets_idlst = get_benchmark_rtn_data(begindate,
                                                                   termidate,
                                                                   bm_assets_ids,
                                                                   bm_tbl_names,
                                                                   dilate,
                                                                   bm_rebal_gapday,
                                                                   _DB
                                                                   )
    bm_weights = [ _BENCHMARK_WEIGHTS[bm_id][asset] for asset in bm_assets_idlst ]

    bm_bt_res = BackTest_benchmark(begindate,
                                   termidate,
                                   bm_hold_rtn_mat_list,
                                   dilate,
                                   bm_weights
                                   )
    # bm_bt_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float, 'annual_rtn':float}

    mvopt_res['benchmark'] = bm_bt_res
    mvopt_res['excess'] = {'rtn':mvopt_res['backtest']['rtn'] - bm_bt_res['rtn'],
                           'annual_rtn':mvopt_res['backtest']['annual_rtn'] - bm_bt_res['annual_rtn']
                           }
    '''
    output mvopt_res =
    {
        'details':res_list,
        'weights':portf_w_list,
        'assets_id':assets_idlst,
        'backtest':
            {'rtn':, 'trade_days':, 'total_cost':, 'gross_rtn':, 'annual_rtn':},
        'benchmark':
            {'rtn':, 'trade_days':, 'total_cost':, 'gross_rtn':, 'annual_rtn':},
        'excess':
            {'rtn':, 'annual_rtn':}
    }
    '''
    return mvopt_res