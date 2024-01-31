from flask import request, Blueprint
import warnings
import numpy as np
from datetime import datetime
from markupsafe import escape
from pprint import pprint
import traceback
import sys
sys.dont_write_bytecode = True
import typing as t
from Code.Allocator.MeanVarOptimal import MeanVarOpt
from Code.projs.asset_allocate.dataLoad import (
    get_train_rtn_data, 
    _DB
)
from Code.BackTester.BT_AssetAllocate import (
    rtn_multi_periods,
    dltDecorBT
)
from Code.projs.asset_allocate.benchmark import benchmarkBT
from Code.projs.asset_allocate.inputParser import *
from Code.projs.asset_allocate.inputParser import parseInput

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

mvopt_api = Blueprint('mean_var', __name__)




@mvopt_api.route('/asset_allocate/mean_var_opt', methods=['POST'])
def application():
    inputs = request.json
    train_rtn_mat_list, hold_rtn_mat_list, assets_idlst = parseInput(inputs)

    num_assets = len(assets_idlst)
    portf_w_list, res_list = [ np.array([1/num_assets,]*num_assets), ], [] # 初始化为平均分配

    for train_rtn_mat in train_rtn_mat_list:

        cov_mat = np.cov(train_rtn_mat)
        rtn_rates = train_rtn_mat.mean(axis=1)
        try:
            fin = MeanVarOpt(rtn_rates, cov_mat, constraints, assets_idlst)

            res = fin(expt_tgt_value, mvo_target)
            # {'portf_w': np.array, 'portf_rtn': np.float32,
            #  'portf_var': np.float32, 'solve_status': str}
        except Exception as e:
            traceback.print_exc()
            res = {'err_msg':str(e), 'status':'fail', 'solve_status':'',
                   'portf_w':np.array([]), 'portf_var':-1, 'portf_rtn':0, 'portf_std':-1,
                   'assets_ids':assets_idlst}
            
        if res['solve_status'] in ('direct', 'qp_optimal'):
            portf_w_list.append( res['portf_w'] )
        else:
            portf_w_list.append( portf_w_list[-1] )
        
        res_list.append(res)

    BT_mvopt = rtn_multi_periods(portf_w_list[1:], hold_rtn_mat_list)
    # {
    #     'rtn': ( invest_amount * gross_rtn - total_cost ) / invest_amount,
    #     'trade_days': trade_days,
    #     'total_cost': total_cost,
    #     'gross_rtn': gross_rtn
    #     }

    fin_bchmk = benchmarkBT(benchmark)
    BT_bchmak = fin_bchmk.backtest()
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
    return {
        'details': res_list,
        'weights': portf_w_list[1:],
        'assets_id': assets_idlst,
        'backtest': BT_mvopt,
        'benchmark': BT_bchmak,
        'excess': {}
    }

