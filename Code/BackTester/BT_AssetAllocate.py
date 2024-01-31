import numpy as np
import functools
from typing import Any, Callable
import typing as t
from datetime import datetime
from functools import wraps
from Code.Utils.Type import basicBackTestRes


def rtn_period(
        portf_w: np.ndarray,
        period_rtn_mat: np.ndarray
        ) -> Any:
    
    assert len(portf_w) == period_rtn_mat.shape[0],\
        f'asset weight array with size {len(portf_w)} must match \
          rtn matrix in period with asset size {period_rtn_mat.shape[0]}'
    
    rtn_array = np.prod((1 + period_rtn_mat), axis=1) - 1

    return (portf_w @ rtn_array)


def day_portfw_on_period(
        init_portf_w: np.ndarray,
        day_rtn_on_period: np.ndarray) -> np.ndarray:
    
    '''
    return:
        day_porftw_mat: shape (num_assets, num_days_period)

    every column is the weight allocation on every day of the portfolio
    args:
        init_portf_w: shape (num_assets, )
        day_rtn_on_period: shape (num_assets, num_days_period)
    '''

    assert len(init_portf_w) == day_rtn_on_period.shape[0],\
        f'asset weight array with size {len(init_portf_w) } must match \
          day rtn matrix in period with asset size {day_rtn_on_period.shape[0]}'
    
    day_porftw_mat = np.zeros_like(day_rtn_on_period)
    days = day_porftw_mat.shape[1]

    day_porftw_mat[:,0] = init_portf_w # 第一列是初始权重

    for i in range(1, days):

        hadmud = day_porftw_mat[:, i-1] * day_rtn_on_period[:, i-1]

        day_porftw_mat[:, i] = ( day_porftw_mat[:, i-1] + hadmud ) / (1 + sum(hadmud))

    return day_porftw_mat

    



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



def basicBT_multiPeriods(
        portf_w_list: t.List[np.ndarray],
        period_rtn_mat_list: t.List[np.ndarray],
        trade_cost_list: t.List[np.float32] = [], 
        invest_amount: t.Union[bool, float] = False
        ) -> basicBackTestRes:
    '''
    return:
    {
        'rtn': np.float32
        'var': np.float32,
        'trade_days': int,
        'total_cost': np.float32,
        'gross_rtn': np.float32
    }
    '''

    assert len(portf_w_list) == len(period_rtn_mat_list),\
        f'portfolio weight allocations {len(portf_w_list)} not match actual rtn\
          matrix records {len(period_rtn_mat_list)}'
    
    day_rtn_lst, trade_days = [], 0

    for portf_w, period_rtn_mat in zip(portf_w_list, period_rtn_mat_list):

        porftw_mat = day_portfw_on_period(portf_w, period_rtn_mat)

        portf_rtn_arr = (porftw_mat * period_rtn_mat).sum(axis=0)

        day_rtn_lst.extend( list(portf_rtn_arr) )

    portf_rtn_arr = np.array( day_rtn_lst )

    if not invest_amount:
        invest_amount = 10000
    
    total_cost = sum(trade_cost_list)
    gross_rtn = np.prod(1+portf_rtn_arr) - 1

    res = {
        'rtn': ( invest_amount * gross_rtn - total_cost ) / invest_amount,
        'var': np.var(portf_rtn_arr),
        'trade_days': trade_days,
        'total_cost': total_cost,
        'gross_rtn': gross_rtn
        }
    
    return res

