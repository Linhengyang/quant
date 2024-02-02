import warnings
from operator import itemgetter
from flask import request, Blueprint
from typing import Any
from Code.Utils.Decorator import serialize

warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'

mvopt_api = Blueprint('mean_var', __name__)


__all__ = [
    "begindate",
    "termidate",
    "dilate",
    "back_window_size",
    "gapday",
    "benchmark",
]


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
def runner():
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
        'excess':
            {'rtn':, 'annual_rtn':}
    }
    '''

    inputs = request.json

    release2global(inputs)


    from Code.projs.asset_allocate.benchmark import benchmarkStrat
    from Code.projs.asset_allocate.meanvarOpt import meanvarOptStrat

    mvopt_strat = meanvarOptStrat(inputs)
    BT_mvopt = mvopt_strat.backtest()

    bchmk_strat = benchmarkStrat(benchmark)
    BT_bchmak = bchmk_strat.backtest()


    result = {
        'details': mvopt_strat.detail_solve_results,
        'weights': mvopt_strat.portf_w_list,
        'assets_id': mvopt_strat.assets_idlst,
        'backtest': BT_mvopt,
        'benchmark': BT_bchmak
    }

    # mvopt_strat = meanvarOptStrat(inputs)
    # result = mvopt_strat.detail_window(12)
    
    return serialize(result)

