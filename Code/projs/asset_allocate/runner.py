import warnings
from operator import itemgetter
from flask import request, Blueprint
from typing import Any
from Code.Utils.Decorator import serialize
from Code.projs.asset_allocate.functions import (
    cal_rtn_intervals,
    cal_rtn_years
)

import numpy as np

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

mvopt_api = Blueprint('mean_var', __name__)
riskmng_api = Blueprint('risk_manage', __name__)
fxdcomb_api = Blueprint('fixed_comb', __name__)


__all__ = [
    "begindate",
    "termidate",
    "dilate",
    "back_window_size",
    "gapday",
    "benchmark",
]


test_mode = False

def release2global(inputs: Any) -> None:

    # 持仓起始日，持仓终结日，膨胀系数, 调仓周期, 基准代号, 回看窗口天数
    global begindate, termidate, dilate, gapday, benchmark, back_window_size

    begindate, termidate, dilate, gapday, benchmark, back_window_size = \
        itemgetter(
            "begindate",
            "termidate",
            "rtn_dilate",
            "gapday",
            "benchmark",
            "back_window_size",
        )(inputs)
    
    print(f'release begindate {begindate}, termidate {termidate}, dilate {dilate},\
            gapday {gapday}, benchmark {benchmark}, back_window_size {back_window_size}\
            to global')

    # 类型转换
    gapday, back_window_size, dilate = int(gapday), int(back_window_size), int(dilate)





@mvopt_api.route('/asset_allocate/mean_var_opt', methods=['POST'])
def mvopt():
    '''
    output:
    {
        'details': [res1 = {'portf_w':[], 'portf_rtn': ,...}, res2],
        'weights': [portf_w1 = [], portf_w2 = [], ...],
        'assets_id': ['id1', 'id2'],
        'backtest':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':, 
            'gross_rtn':, 'annual_rtn':},
        'benchmark':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':,
            'gross_rtn':, 'annual_rtn':},
    }
    '''
    if test_mode:
        from Test.test_result import test_res
        return test_res
    
    inputs = request.json

    release2global(inputs)


    from Code.projs.asset_allocate.benchmark import benchmarkStrat
    from Code.Strategy.LowFrequency.meanvarOpt import meanvarOptStrat

    stratg = meanvarOptStrat(inputs)
    BT_stratg = stratg.backtest()

    bchmk = benchmarkStrat(benchmark)
    BT_bchmk = bchmk.backtest()


    details, hold_dates_lst, portf_rtn_series_lst, bchmk_rtn_series_lst = [], [], [], []
    for strat_detail, bchmk_detail in zip(stratg.details, bchmk.details):
        assert strat_detail['position_no'] == bchmk_detail['position_no'], \
            f"position number not match on strategy {strat_detail['position_no']}"

        strat_detail['bchmk_rtn'] = bchmk_detail['bchmk_rtn']
        strat_detail['bchmk_rtn_series'] = bchmk_detail['bchmk_rtn_series']
        strat_detail['bchmk_std'] = bchmk_detail['bchmk_std']
        strat_detail['bchmk_var'] = bchmk_detail['bchmk_var']
        strat_detail['execs_rtn'] = strat_detail['portf_rtn'] - bchmk_detail['bchmk_rtn']

        details.append( strat_detail )

        hold_dates_lst.append( strat_detail['hold_dates'] )
        portf_rtn_series_lst.append( strat_detail['portf_rtn_series'] )
        bchmk_rtn_series_lst.append( strat_detail['bchmk_rtn_series'] )


    hold_dates_arr = np.concatenate(hold_dates_lst)
    portf_rtn_arr = np.concatenate(portf_rtn_series_lst)
    bchmk_rtn_arr = np.concatenate(bchmk_rtn_series_lst)

    result = {
        'details': details,
        'weights': stratg.weights,
        'assets_id': stratg.assets_idlst,
        'backtest': BT_stratg,
        'benchmark': BT_bchmk,
        'intervals': cal_rtn_intervals(hold_dates_arr, portf_rtn_arr, bchmk_rtn_arr, termidate),
        'years':cal_rtn_years(hold_dates_arr, portf_rtn_arr, bchmk_rtn_arr)
    }

    
    return serialize(result)





@riskmng_api.route('/asset_allocate/risk_manage', methods=['POST'])
def riskmanage():
    '''
    output:
    {
        'details': [res1 = {'portf_w':[], 'portf_rtn': ,...}, res2],
        'weights': [portf_w1 = [], portf_w2 = [], ...],
        'assets_id': ['id1', 'id2'],
        'backtest':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':, 
            'gross_rtn':, 'annual_rtn':},
        'benchmark':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':,
            'gross_rtn':, 'annual_rtn':},
    }
    '''
    if test_mode:
        from Test.test_result import test_res
        return test_res
    
    inputs = request.json

    release2global(inputs)


    from Code.projs.asset_allocate.benchmark import benchmarkStrat
    from Code.Strategy.LowFrequency.riskMng import riskMngStrat

    stratg = riskMngStrat(inputs)
    BT_stratg = stratg.backtest()

    bchmk = benchmarkStrat(benchmark)
    BT_bchmk = bchmk.backtest()


    details, hold_dates_lst, portf_rtn_series_lst, bchmk_rtn_series_lst = [], [], [], []
    for strat_detail, bchmk_detail in zip(stratg.details, bchmk.details):
        assert strat_detail['position_no'] == bchmk_detail['position_no'], \
            f"position number not match on strategy {strat_detail['position_no']}"

        strat_detail['bchmk_rtn'] = bchmk_detail['bchmk_rtn']
        strat_detail['bchmk_rtn_series'] = bchmk_detail['bchmk_rtn_series']
        strat_detail['bchmk_std'] = bchmk_detail['bchmk_std']
        strat_detail['bchmk_var'] = bchmk_detail['bchmk_var']
        strat_detail['execs_rtn'] = strat_detail['portf_rtn'] - bchmk_detail['bchmk_rtn']

        details.append( strat_detail )

        hold_dates_lst.append( strat_detail['hold_dates'] )
        portf_rtn_series_lst.append( strat_detail['portf_rtn_series'] )
        bchmk_rtn_series_lst.append( strat_detail['bchmk_rtn_series'] )


    hold_dates_arr = np.concatenate(hold_dates_lst)
    portf_rtn_arr = np.concatenate(portf_rtn_series_lst)
    bchmk_rtn_arr = np.concatenate(bchmk_rtn_series_lst)

    result = {
        'details': details,
        'weights': stratg.weights,
        'assets_id': stratg.assets_idlst,
        'backtest': BT_stratg,
        'benchmark': BT_bchmk,
        'intervals': cal_rtn_intervals(hold_dates_arr, portf_rtn_arr, bchmk_rtn_arr, termidate),
        'years':cal_rtn_years(hold_dates_arr, portf_rtn_arr, bchmk_rtn_arr)
    }

    
    return serialize(result)




@fxdcomb_api.route('/asset_allocate/fixed_combination', methods=['POST'])
def fixedcomb():
    '''
    output:
    {
        'details': [res1 = {'portf_w':[], 'portf_rtn': ,...}, res2],
        'weights': [portf_w1 = [], portf_w2 = [], ...],
        'assets_id': ['id1', 'id2'],
        'backtest':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':, 
            'gross_rtn':, 'annual_rtn':},
        'benchmark':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':,
            'gross_rtn':, 'annual_rtn':},
    }
    '''
    if test_mode:
        from Test.test_result import test_res
        return test_res
    
    inputs = request.json

    release2global(inputs)


    from Code.projs.asset_allocate.benchmark import benchmarkStrat
    from Code.Strategy.LowFrequency.fixedComb import FxdCombStrat

    stratg = FxdCombStrat(inputs)
    BT_stratg = stratg.backtest()

    bchmk = benchmarkStrat(benchmark)
    BT_bchmk = bchmk.backtest()


    details, hold_dates_lst, portf_rtn_series_lst, bchmk_rtn_series_lst = [], [], [], []
    for strat_detail, bchmk_detail in zip(stratg.details, bchmk.details):
        assert strat_detail['position_no'] == bchmk_detail['position_no'], \
            f"position number not match on strategy {strat_detail['position_no']}"

        strat_detail['bchmk_rtn'] = bchmk_detail['bchmk_rtn']
        strat_detail['bchmk_rtn_series'] = bchmk_detail['bchmk_rtn_series']
        strat_detail['bchmk_std'] = bchmk_detail['bchmk_std']
        strat_detail['bchmk_var'] = bchmk_detail['bchmk_var']
        strat_detail['execs_rtn'] = strat_detail['portf_rtn'] - bchmk_detail['bchmk_rtn']

        details.append( strat_detail )

        hold_dates_lst.append( strat_detail['hold_dates'] )
        portf_rtn_series_lst.append( strat_detail['portf_rtn_series'] )
        bchmk_rtn_series_lst.append( strat_detail['bchmk_rtn_series'] )


    hold_dates_arr = np.concatenate(hold_dates_lst)
    portf_rtn_arr = np.concatenate(portf_rtn_series_lst)
    bchmk_rtn_arr = np.concatenate(bchmk_rtn_series_lst)

    result = {
        'details': details,
        'weights': stratg.weights,
        'assets_id': stratg.assets_idlst,
        'backtest': BT_stratg,
        'benchmark': BT_bchmk,
        'intervals': cal_rtn_intervals(hold_dates_arr, portf_rtn_arr, bchmk_rtn_arr, termidate),
        'years':cal_rtn_years(hold_dates_arr, portf_rtn_arr, bchmk_rtn_arr)
    }

    
    return serialize(result)