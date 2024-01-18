from flask import Flask, request, Blueprint
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

# asset_allocate_app = Flask(app_name, static_folder=static_folder, template_folder=template_folder)
risk_manage_api = Blueprint('risk_manage', __name__)

# risk manage
# tgt_contrib_ratio = None，为风险平价; 不为None时，为风险预算
def get_risk_manage(category_mat, tgt_contrib_ratio, rtn_data, assets_inds):
    if category_mat is not None:
        category_mat = np.array(category_mat)
    if tgt_contrib_ratio is not None:
        tgt_contrib_ratio = np.array(tgt_contrib_ratio)
    try:
        fin = RiskParity(rtn_data, category_mat, tgt_contrib_ratio, assets_inds=assets_inds)
        portf_w = fin.optimal_solver()
        res = {'portf_w':list(portf_w), 'portf_var':fin.risk_contribs.sum(), 'portf_r':fin.portf_return,\
               'risk_contribs':list(fin.risk_contribs),'assets_inds':fin.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    return res



# risk manage
# tgt_contrib_ratio = None，为风险平价; 不为None时，为风险预算
@risk_manage_api.route('/asset_allocate/risk_manage', methods=['POST'])
def application_riskmanage():
    inputs = request.json
    assets_info = inputs["assets_info"] # assets_info
    num_assets = len(assets_info)
    # 资产信息: id, 类别
    assets_inds, assets_categs, categs, tgt_risk_ratio = [], [], [], []
    for asset in assets_info:
        assets_inds.append(asset['id'])
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
          format(strtg=strategy, begindate=begindate, termidate=termidate,gapday=gapday, assets=assets_inds))

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
        res = get_risk_manage(category_mat, tgt_contrib_ratio, train_rtn_mat, assets_inds)
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
    return {'details':res_list, 'backtest':BT_res ,'weights':portf_w_list[1:], 'assets_id':assets_inds}