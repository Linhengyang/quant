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
from typing import Any, Callable
from Code.Utils.Type import (
    basicPortfSolveRes,
    basicBackTestRes
)
from operator import itemgetter

from Code.Allocator.MeanVarOptimal import MeanVarOpt
from Code.projs.asset_allocate.dataLoad import (
    get_train_hold_rtn_data, 
    _DB
)
from Code.BackTester.BT_AssetAllocate import (
    basicBT_multiPeriods
)
from Code.projs.asset_allocate.benchmark import benchmarkBT
from Code.projs.asset_allocate.inputParser import *
from Code.projs.asset_allocate.inputParser import (
    getAssetInfo_rls2glob,
    get_constraints
    )
from Code.Utils.Decorator import deDilate, addAnnual, addSTD


warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

mvopt_api = Blueprint('mean_var', __name__)


def get_meanvar_data_params(
        inputs: Any
        ) -> Any:
    
    # global begindate, termidate, dilate, gapday, benchmark, back_window_size
    assets_dict, src_tbl_dict = getAssetInfo_rls2glob(inputs)

    mvo_target, expt_tgt_value = \
        itemgetter("mvo_target", "expt_tgt_value")(inputs)
    
    print(
        f'Back Test for mean-variance-optimal {mvo_target} strategy \
          from {begindate} to {termidate} trading on every {gapday} \
          upon {len(assets_dict)} assets')

    expt_tgt_value = np.float32(expt_tgt_value)

    tbl_names = list( src_tbl_dict.keys() ) # list of str
    assets_ids = [ src_tbl_dict[tbl] for tbl in tbl_names] # list of lists

    train_rtn_mat_list, hold_rtn_mat_list, assets_idlst = \
        get_train_hold_rtn_data(
            begindate,
            termidate,
            gapday,
            back_window_size,
            dilate,
            assets_ids,
            tbl_names,
            _DB
        )
    
    constraints = get_constraints( assets_dict, assets_idlst )

    return train_rtn_mat_list, hold_rtn_mat_list, assets_idlst, \
           constraints, mvo_target, expt_tgt_value




@addSTD('portf_var')
@deDilate(dilate)
def solve_mvopt(
    train_rtn_mat,
    assets_idlst,
    constraints,
    mvo_target,
    expt_tgt_value,
    ) -> basicPortfSolveRes:

    cov_mat = np.cov(train_rtn_mat)
    rtn_rates = train_rtn_mat.mean(axis=1)

    try:
        fin = MeanVarOpt(rtn_rates, cov_mat,
                         constraints, assets_idlst)
        
        res = fin(expt_tgt_value, mvo_target)
        
    except Exception as e:
        traceback.print_exc()

        res = {
            'portf_w': np.array([]),
            'portf_rtn': 0,
            'portf_var': -1,
            'solve_status': 'FAIL_' + str(e),
            'assets_idlst': assets_idlst
            }
    
    return res



@addAnnual('rtn', begindate, termidate)
@deDilate(dilate)
@addSTD('var')
def mvoptStrat_backtest(
        portf_w_list: t.List[np.array],
        period_rtn_mat_list: t.List[np.array]
        ) -> Any:
    '''
    de-dilated
        'rtn': np.float32
        'var': np.float32,
        'std': np.float32
        'trade_days': int,
        'total_cost': float,
        'gross_rtn': np.float32
        'annual_rtn': np.float32
    '''

    return basicBT_multiPeriods(
                portf_w_list,
                period_rtn_mat_list)




@mvopt_api.route('/asset_allocate/mean_var_opt', methods=['POST'])
def application():
    '''
    output:
    {
        'details': [res1, res2],
        'weights': [portf_w1, portf_w2],
        'assets_id': ['id1', 'id2'],
        'backtest':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':, 
            'gross_rtn':, 'annual_rtn':},
        'benchmark':
            {'rtn', 'var', 'std', 'trade_days':, 'total_cost':,
            'gross_rtn':, 'annual_rtn':},
        'excess':
            {'rtn':, 'annual_rtn':}
    }
    '''

    inputs = request.json

    train_rtn_mat_list, hold_rtn_mat_list, assets_idlst, constraints,\
        mvo_target, expt_tgt_value = get_meanvar_data_params(inputs)
    
    num_assets = len(assets_idlst)

    # 初始化为平均分配
    portf_w_list, res_list = [np.repeat(1/num_assets, num_assets), ], []

    for train_rtn_mat in train_rtn_mat_list:

        cur_res = solve_mvopt(
            train_rtn_mat,
            constraints,
            assets_idlst,
            expt_tgt_value,
            mvo_target
            )

        if cur_res['solve_status'] in ('direct', 'qp_optimal'):
            portf_w_list.append( cur_res['portf_w'] )
        else:
            portf_w_list.append( portf_w_list[-1] )
        
        res_list.append(cur_res)

    BT_mvopt = mvoptStrat_backtest(portf_w_list[1:], hold_rtn_mat_list)

    BT_bchmak = benchmarkBT(benchmark).backtest()

    return {
        'details': res_list,
        'weights': portf_w_list[1:],
        'assets_id': assets_idlst,
        'backtest': BT_mvopt,
        'benchmark': BT_bchmak,
        'excess': {}
    }

