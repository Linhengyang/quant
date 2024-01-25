from Code.projs.asset_allocate.dataload import db_rtn_data, db_date_data
from Code.Utils.Sequence import strided_slicing_w_residual
from Code.BackTester.BT_AssetAllocate import rtn_multi_periods, modify_BackTestResult


import numpy as np
from datetime import datetime
import sys
sys.dont_write_bytecode = True


def get_benchmark_rtn_data(begindate, termidate, assets_inds:list, dilate, rebal_gapday=None):
    # 取数据，一次io解决
    # 取出 begindate (包括) 至 termidate (包括）, 所有的交易日期，已排序
    mkt_dates = db_date_data(begindate, termidate) # 返回numpy of int
    # rtn_data: (num_assets, mkt_days)的numpy matrix, assets_inds: 长度为num_assets的list
    rtn_data, assets_inds = db_rtn_data(assets_inds, begindate, termidate, dilate, "aidx_eod_prices")
    # 每gapday持仓
    # 当前，月度持仓精简为每20天持仓
    if isinstance(rebal_gapday, int): # 若参数输入了rebal_gapday, 那么以 rebal_gapday作为gapday调仓
        strided_slices, _, last_range = strided_slicing_w_residual(rtn_data.shape[1], rebal_gapday, rebal_gapday)
        hold_rtn_mat_list = list(rtn_data.T[strided_slices].transpose(0,2,1))
        if list(last_range): # rsd_range不为空
            hold_rtn_mat_list.append( rtn_data.T[last_range].T )
    else: # benchmark不需要调仓
        hold_rtn_mat_list = [rtn_data]
    return hold_rtn_mat_list, assets_inds



def BackTest_benchmark(begindate, termidate, hold_rtn_mat_list, dilate, assets_inds:list, weights:list=[]):
    assert weights == [] or len(assets_inds) == len(weights), "benchmark asset lists length must match with weights"
    num_assets = len(assets_inds) # 资产个数
    num_hold_periods = len(hold_rtn_mat_list) # 持仓期数
    if weights == [] : # 如果 没有输入 weights, 默认benchmark里所有资产平均分配
        portf_w_list = [ np.array([1/num_assets,]*num_assets), ] * num_hold_periods
    else:
        portf_w_list = [np.array(weights),] * num_hold_periods
    # 回测
    BT_res = rtn_multi_periods(portf_w_list, hold_rtn_mat_list)
    # 回测结果dilate修正
    BT_res = modify_BackTestResult(BT_res, dilate, begindate, termidate)
    # BT_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float, 'annual_rtn':float}
    return BT_res


def parse_benchmark(benchmark:str):
    if benchmark == 'CSI800':
        assets_inds = ["h00906.CSI"]
        rebal_gapday = None
        weights = []
    elif benchmark == "CDCSI":
        assets_inds = ["CBA00101.CS"]
        rebal_gapday = None
        weights = []
    elif benchmark == "S2D8_Mon":
        assets_inds = ["h00906.CSI", "CBA00101.CS"]
        rebal_gapday = 20
        weights = [0.2, 0.8]
    elif benchmark == "S4D6_Mon":
        assets_inds = ["h00906.CSI", "CBA00101.CS"]
        rebal_gapday = 20
        weights = [0.4, 0.6]
    elif benchmark == "S5D5_Mon":
        assets_inds = ["h00906.CSI", "CBA00101.CS"]
        rebal_gapday = 20
        weights = [0.5, 0.5]
    elif benchmark == "S8D2_Mon":
        assets_inds = ["h00906.CSI", "CBA00101.CS"]
        rebal_gapday = 20
        weights = [0.8, 0.2]
    else:
        raise ValueError("wrong code {benchmark} for benchmark".format(benchmark=benchmark))
    return assets_inds, weights, rebal_gapday