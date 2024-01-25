from Code.projs.asset_allocate.dataload import db_rtn_data, db_rtn_data_multi_tbl
from Code.Utils.Sequence import strided_slicing_w_residual
from Code.BackTester.BT_AssetAllocate import rtn_multi_periods, modify_BackTestResult


import numpy as np
from datetime import datetime
import sys
sys.dont_write_bytecode = True


def get_benchmark_rtn_data(begindate, termidate, assets_ids:list, tbl_names, dilate, rebal_gapday):
    '''
    assets_ids & tbl_names:
    1. if all assets come from 1 table, then arg {tbl_names} is the string of the table, assets_ids is a list of asset id codes
        e.g, assets_ids = ['000001.SH', '000002.SH'], tbl_names = 'aidx_eod_prices'
    2. if assets come from multiple tables, then arg {tbl_names} is the string of tables, assets_ids is a list of lists of asset id code
    which come from corresponding table name by order.
        e.g, assets_ids = [['000001.SH', '000002.SH'], ['CBA0001.CBI']], tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
    '''
    # 取数据，一次io解决
    # rtn_data: (num_assets, mkt_days)的numpy matrix, assets_idlst: 长度为num_assets的list
    if isinstance(assets_ids[0], list): # assets_ids = [['000001.SH', '000002.SH'], ['CBA0001.CB' ,'CBA0002.CB']]
        assert isinstance(tbl_names, list), "arg tbl_names must be a list for different tables"
        rtn_data, assets_idlst = db_rtn_data_multi_tbl(assets_ids, begindate, termidate, dilate, tbl_names)
    else: # assets_ids = ['000001.SH', '000002.SH']
        assert isinstance(tbl_names, str), "arg tbl_names must be string"
        rtn_data, assets_idlst = db_rtn_data(assets_ids, begindate, termidate, dilate, tbl_names)
    # 每gapday持仓
    # 当前，月度持仓精简为每20天持仓
    if isinstance(rebal_gapday, int): # 若参数输入了rebal_gapday, 那么以 rebal_gapday作为gapday调仓
        strided_slices, _, last_range = strided_slicing_w_residual(rtn_data.shape[1], rebal_gapday, rebal_gapday)
        hold_rtn_mat_list = list(rtn_data.T[strided_slices].transpose(0,2,1))
        if list(last_range): # rsd_range不为空
            hold_rtn_mat_list.append( rtn_data.T[last_range].T )
    else: # benchmark不需要调仓
        hold_rtn_mat_list = [rtn_data]
    return hold_rtn_mat_list, assets_idlst



def BackTest_benchmark(begindate, termidate, hold_rtn_mat_list, dilate, weights:list=[]):
    num_assets = hold_rtn_mat_list[0].shape[0]
    assert weights == [] or num_assets == len(weights), "hold_rtn_mat axis 0 lenght must match with weights"
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


benchmark_weights = {
    "CSI800":{
        "h00906.CSI":1
    },
    "CDCSI":{
        "CBA00101.CS":1
    },
    "S2D8_Mon":{
        "h00906.CSI":0.2,
        "CBA00101.CS":0.8
    },
    "S4D6_Mon":{
        "h00906.CSI":0.4,
        "CBA00101.CS":0.6
    },
    "S5D5_Mon":{
        "h00906.CSI":0.5,
        "CBA00101.CS":0.5
    },
    "S8D2_Mon":{
        "h00906.CSI":0.8,
        "CBA00101.CS":0.2
    }
}

def parse_benchmark(benchmark:str):
    if benchmark == 'CSI800':
        assets_ids = ["h00906.CSI"]
        rebal_gapday = None
        tbl_names = 'aidx_eod_prices'
    elif benchmark == "CDCSI":
        assets_ids = ["CBA00101.CS"]
        rebal_gapday = None
        tbl_names = 'cbidx_eod_prices'
    elif benchmark == "S2D8_Mon":
        assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
        rebal_gapday = 20
        tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
    elif benchmark == "S4D6_Mon":
        assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
        rebal_gapday = 20
        tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
    elif benchmark == "S5D5_Mon":
        assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
        rebal_gapday = 20
        tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
    elif benchmark == "S8D2_Mon":
        assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
        rebal_gapday = 20
        tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
    else:
        raise ValueError("wrong code {benchmark} for benchmark".format(benchmark=benchmark))
    return assets_ids, tbl_names, rebal_gapday
