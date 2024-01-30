import numpy as np
import functools
from typing import Any, Callable
import typing as t
from datetime import datetime
from functools import wraps


def rtn_period(
        portf_w:np.array,
        period_rtn_mat:np.array
        ) -> Any:
    
    assert len(portf_w) == len(period_rtn_mat),\
        f'asset weight array with size {len(portf_w)} must match \
          rtn matrix in period with asset size {len(period_rtn_mat)}'
    
    rtn_array = np.prod((1 + period_rtn_mat), axis=1) - 1

    return (portf_w @ rtn_array)




def reallocate_cost(
        portf_w_list: list,
        period_rtn_mat_list: list,
        invest_amount: float
        ) -> t.List:
    '''
                                     cost1                                          cost2
    portf_w1  with invest_amount A  ------->  portf_w2 with invest_amount A(1+r1)  -------> portf_w3 with invest_amount A(1+r2)
    
     cost_clear
    ------------>   0 asset with cash
    '''
    return [0]*len(portf_w_list)



def rtn_multi_periods(
        portf_w_list: t.List[np.array],
        period_rtn_mat_list: t.List[np.array],
        trade_cost_list: t.List[np.float32] = [], 
        invest_amount: bool = False
        ) -> dict:
    '''
    begin with A = invest_amount

                       [[],
                        [],                                rtn_period()
    portf_w1  ----->    .                                   ----->               r1 as return rate, end with A(1+r1)
                        .
                        []] (assets in days_1 period return mat)   
         (cost_1)                        
                       [[],
                        [],                                rtn_period()
    portf_w2  ----->    .                                   ----->               r2 as return rate, end with A(1+r1)(1+r2)
                        .
                        []] (assets in days_2 period return mat)   
         (cost_2)                        
                       [[],
                        [],                                rtn_period()
    portf_w3  ----->    .                                   ----->               r3 as return rate, end with A(1+r1)(1+r2)(1+r3)
                        .
                        []] (assets in days_3 period return mat)   
         (cost for clear)
    END with A(1+r1)(1+r2)(1+r3) - (cost_1 + cost_2 + cost_clear)

    complex return rate =
                            A[(1+r1)(1+r2)(1+r3) - 1] - (cost_1 + cost_2 + cost_clear)
                                                    over
                                                    A
                            in days_1 + days_2 + days_3 period
    '''

    assert len(portf_w_list) == len(period_rtn_mat_list),\
        f'portfolio weight allocations {len(portf_w_list)} not match actual rtn\
          matrix records {len(period_rtn_mat_list)}'
    
    rtn_list, trade_days = [], 0

    for portf_w, period_rtn_mat in zip(portf_w_list, period_rtn_mat_list):

        rtn_list.append(
            rtn_period(portf_w, period_rtn_mat)
            )
        
        trade_days += period_rtn_mat.shape[1]
    
    rtn_array = np.array(rtn_list)

    if not invest_amount:
        invest_amount = 10000
    
    total_cost = sum(trade_cost_list)
    gross_rtn = np.prod(1+rtn_array) - 1

    res = {
        'rtn': ( invest_amount * gross_rtn - total_cost ) / invest_amount,
        'trade_days': trade_days,
        'total_cost': total_cost,
        'gross_rtn': gross_rtn
        }
    
    return res




# def modify_BackTestResult(BT_res:dict, dilate:int, begindate:str, termidate:str):
#     # input BT_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float}
#     # rtn 膨胀系数修正 和 年化利率计算
#     BT_res['rtn'] = BT_res['rtn']/dilate
#     BT_res['gross_rtn'] = BT_res['gross_rtn']/dilate
#     delta_year = ( datetime.strptime(termidate, '%Y%m%d') - datetime.strptime(begindate, '%Y%m%d') ).days / 365
#     BT_res['annual_rtn'] = np.power( 1 + BT_res['rtn'], 1/delta_year) - 1
#     # output BT_res = {'rtn': float, 'gross_rtn': float, 'annual_rtn':float,
#     #                  'trade_days': int, 'total_cost': float}
#     return BT_res




class dltDecorBT:
    '''
    decorator for BackTest result
    BackTest Result must be a dict. 
    This decorator modify all values with key ending with "rtn|std|var" by dilate
    and add 'annual_rtn'
    '''
    def __init__(self,
                 rtn_dilate: int,
                 begindate: str,
                 termidate: str
                 ) -> None:
        
        self.__rtn_dilate = rtn_dilate
        self.__delta_year = ( datetime.strptime(termidate, '%Y%m%d') - \
                              datetime.strptime(begindate, '%Y%m%d')
                            ).days / 365


    def __call__(self,
                 backtest_fc: Callable
                 ) -> Callable:
        
        @wraps(backtest_fc)
        def wrapper(*args, **kwargs):
            bt_res = backtest_fc(*args, **kwargs)
            keys = bt_res.keys()
            for key in keys:
                if key.endswith('rtn'): # 所有rtn结尾的，都要 /dilate
                    bt_res[key] /= self.__rtn_dilate
                elif key.endswith('std'): # 所有std结尾的，都要 / dilate
                    bt_res[key] /= self.__rtn_dilate
                elif key.endswith('var'): # 所有var结尾的，都要 / dilate^2
                    bt_res[key] /= np.power(self.__rtn_dilate, 2)
            # 添加 annual_rtn
            bt_res['annual_rtn'] = np.power( 1 + bt_res['rtn'], 1/self.__delta_year) - 1
            return bt_res
        
        return wrapper




#  class Decorator:
#      def __init__(self, arg1, arg2):
#          print('执行类Decorator的__init__()方法')
#          self.arg1 = arg1
#          self.arg2 = arg2
         
#      def __call__(self, f):
#          print('执行类Decorator的__call__()方法')
#          def wrap(*args):
#              print('执行wrap()')
#              print('装饰器参数：', self.arg1, self.arg2)
#              print('执行' + f.__name__ + '()')
#              f(*args)
#              print(f.__name__ + '()执行完毕')
#          return wrap
