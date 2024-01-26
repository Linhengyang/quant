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
from Code.Allocator.RiskParity import RiskParity
from Code.projs.asset_allocate.dataload import get_train_rtn_data
from Code.Utils.Sequence import strided_slicing_w_residual
from Code.BackTester.BT_AssetAllocate import rtn_multi_periods

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

# asset_allocate_app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)
risk_manage_api = Blueprint('risk_manage', __name__)

# risk manage
# tgt_contrib_ratio = None，为风险平价; 不为None时，为风险预算
def riskmng_portf(category_mat, tgt_contrib_ratio, rtn_data, assets_idlst):
    if category_mat is not None:
        category_mat = np.array(category_mat)
    if tgt_contrib_ratio is not None:
        tgt_contrib_ratio = np.array(tgt_contrib_ratio)
    try:
        fin = RiskParity(rtn_data, category_mat, tgt_contrib_ratio, assets_idlst)
        res = fin.optimal_solver()
        res['err_msg'] = ''
        res['status'] = 'success'
        res['assets_ids'] = fin.assets_idlst
        # res = {"portf_w": np.array, "portf_var": float, "portf_rtn": float, "risk_contribs": np.array, "solve_status": string
        #        "err_msg": string, 'status':string, 'assets_ids':list}
    except Exception as e:
        traceback.print_exc()
        res = {'err_msg':str(e), 'status':'fail', 'solve_status':'',
               'portf_w':np.array([]), 'portf_var':-1, 'portf_rtn':0,
               'assets_ids':assets_idlst}
    return res


def modify_SolveResult(solve_res:dict, dilate:int, begindate:str, termidate:str):
    # input solve_res = {'err_msg':str(e), 'status':'fail', 'solve_status':'',
    #                    'portf_w':np.array([]), 'portf_var':-1, 'portf_rtn':0,
    #                    'assets_ids':mvopt.assets_ids}
    if solve_res['status'] == 'success': # 当运行成功时
        solve_res['portf_rtn'] /= dilate # rtn 膨胀系数修正
        solve_res['portf_std'] = np.sqrt(solve_res['portf_var']) / dilate # 标准差计算
        solve_res['portf_var'] /= (dilate*dilate) # var膨胀系数修正
        delta_year = ( datetime.strptime(termidate, '%Y%m%d') - datetime.strptime(begindate, '%Y%m%d') ).days / 365
        solve_res['portf_ann_rtn'] = np.power( 1 + solve_res['portf_rtn'], 1/delta_year) - 1 # 年化利率计算
    else: # 当运行失败
        solve_res['portf_std'] = -1 # 标准差
        solve_res['portf_ann_rtn'] = 0 # 年化利率
    
    solve_res['portf_w'] = list(solve_res['portf_w']) # 将np.array转化为list，以输出json
    # output solve_res = {'err_msg':str(e), 'status':str, 'solve_status':str,
    #                     'portf_w':list, 'portf_var':float, 'portf_rtn':float, 'portf_std':float, 'portf_ann_rtn':float,
    #                     'assets_ids':list}
    return solve_res




# risk manage
# tgt_contrib_ratio = None，为风险平价; 不为None时，为风险预算
@risk_manage_api.route('/asset_allocate/risk_manage', methods=['POST'])
def riskmng():
    inputs = request.json
    assets_info = inputs["assets_info"] # assets_info
    num_assets = len(assets_info)
    # 资产信息: id, 类别
    assets_ids, assets_categs, categs, tgt_risk_ratio = [], [], [], []
    for asset in assets_info:
        assets_ids.append(asset['id'])
        if 'category' in asset:
            next_categ = asset['category'] # 如果该资产输入了类别, 那么该类别被记录下来
        else:
            next_categ = "Null" # 如果该资产没有输入类别，那么记录 Null
        assets_categs.append(next_categ) # 记录资产类别
        if next_categ not in categs: # 记录去重后的所有类别
            categs.append(next_categ)
        if 'asset_risk_ratio' in asset: # 如果该资产输入了贡献风险比例
            tgt_risk_ratio.append(asset['asset_risk_ratio']) # 记录该比例
        else: # 如果该资产没有贡献风险比例
            tgt_risk_ratio.append(None) # 记录None
    # 制作 tgt_contrib_ratio.
    if tgt_risk_ratio == [None]*num_assets: # 如果所有都是None，说明用所有都风险平价来计算
        tgt_contrib_ratio = None
    elif None not in tgt_risk_ratio: # 所有都是数值
        tgt_contrib_ratio = np.array(tgt_risk_ratio)
    else: # 有数值且存在None, 默认None的资产平均分担风险
        cur_risk = filter(None, tgt_risk_ratio)
        none_num = len(tgt_risk_ratio) - len(cur_risk)
        if cur_risk > 1:
            raise ValueError('existing risk contribution ratios sum up than 1')
        tgt_contrib_ratio = np.array([float(x) if x is not None else (1-cur_risk)/none_num for x in tgt_risk_ratio])
    # 制作category_mat. 当前暂不考虑按大资产类别作风险管理, 只支持资产纬度的风险管理
    if len(categs) == 1 and categs[0] == 'Null': # 如果全部资产都没有输入category
        category_mat = None
    else:
        category_mat = None
        # category_mat = np.array( [list(assets_categs == categ) for categ in categs] ).astype(int)
    # 持仓起始日，持仓终结日，调仓频率，回看窗口天数，膨胀系数
    begindate, termidate, gapday, back_window_size, dilate = inputs['begindate'], inputs['termidate'],\
        int(inputs['gapday']), int(inputs['back_window_size']), int(inputs['rtn_dilate'])
    if tgt_contrib_ratio is None:
        strategy = "parity"
    else:
        strategy = "budget"
    print('BackTest for risk {strtg} from {begindate} to {termidate} trading in every {gapday} upon assets {assets}'.\
          format(strtg=strategy, begindate=begindate, termidate=termidate,gapday=gapday, assets=assets_ids))
    
    # 获取训练和持仓数据
    train_rtn_mat_list, hold_rtn_mat_list, assets_idlst = get_train_rtn_data(begindate, termidate, gapday, back_window_size,\
                                                                             dilate, assets_ids, 'aidx_eod_prices')
    portf_w_list, res_list = [[1/num_assets]*num_assets,], []
    for train_rtn_mat in train_rtn_mat_list:
        # train_rtn_mat shape: (num_assets, back_window_size)
        res = riskmng_portf(category_mat, tgt_contrib_ratio, train_rtn_mat, assets_idlst)
        # 求解失败: {"err_msg":str(e), 'status':'fail'}
        # 求解成功: {'portf_w':list, 'portf_var':float, 'portf_r':float, 'risk_contribs':list, 'assets_inds':list}
        if 'portf_w' in res: # 当有解
            # dilate修正
            res['portf_std'] = np.sqrt(res['portf_var'])
            res['portf_var'] = res['portf_var'] / np.power(dilate, 2)
            res['portf_std'] = res['portf_std'] / dilate
            res['portf_r'] = res['portf_r'] / dilate
        res_list.append(res) # 记录求解原生结果
        if 'portf_w' in res: # 当有解
            portf_w_list.append(res['portf_w']) # 记录权重
        else: # 求解失败
            portf_w_list.append(portf_w_list[-1]) # 延用上一期结果
    # 回测
    BT_res = rtn_multi_periods(portf_w_list[1:], hold_rtn_mat_list)
    # BT_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float, 'annual_rtn':float}
    # 回测结果dilate修正
    BT_res['rtn'] = BT_res['rtn']/dilate
    BT_res['gross_rtn'] = BT_res['gross_rtn']/dilate
    delta_year = ( datetime.strptime(termidate, '%Y%m%d') - datetime.strptime(begindate, '%Y%m%d') ).days / 365
    BT_res['annual_rtn'] = np.power( 1 + BT_res['rtn'], 1/delta_year) - 1
    return {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:], 'assets_id':assets_idlst}