import numpy as np
from typing import Any
import typing as t
from Code.Utils.Type import basicBackTestRes
from Code.Utils.Statistic import maxdrawdown


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
        day_rtn_on_period: np.ndarray
        ) -> np.ndarray:
    
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

        day_porftw_mat[:, i] = ( day_porftw_mat[:, i-1] + hadmud )/ \
                                    (1 + sum(hadmud))

    return day_porftw_mat

    



def reallocate_cost(
        portf_w_list: t.List[np.ndarray],
        hold_rtn_mat_list: t.List[np.ndarray],
        invest_amount: float
        ) -> t.List:
    # TODO
    '''
                                     cost1                                        
    portf_w1  with invest_amount A  ------->  portf_w2 with invest_amount A(1+r1) 
    cost2                                            cost_clear
    -------> portf_w3 with invest_amount A(1+r2)   ------------>   0 asset with cash
    '''
    return [0]*len(portf_w_list)








def basicBT_multiPeriods(
        portf_w_list: t.List[np.ndarray],
        hold_rtn_mat_list: t.List[np.ndarray],
        trade_cost: Any = None, 
        invest_amount: t.Union[bool, float] = False
        ) -> basicBackTestRes:
    '''
    return rates are meaningless after invest amount goes negative
    basicBT_multiPeriods assumes no further reaction after invest
    amount goes non-positive, and the BackTest stops.

    intput:
        portf_w_list: list of weight array
        period_rtn_mat_list: list of ndarray
    return:
    {
        'rtn': np.float32,
        'var': np.float32,
        'trade_days': int,
        'total_cost': np.float32,
        'gross_rtn': np.float32,
        'maxdd': np.float32
    }
    '''
    
    if not invest_amount:
        invest_amount = 10000.0

    assert len(portf_w_list) == len(hold_rtn_mat_list),\
        f'number of periods of portfolio weights {len(portf_w_list)} \
          not match with actual rtn matrix records {len(hold_rtn_mat_list)}'
    
    day_rtn_lst, cost_lst = [], []

    for portf_w, period_rtn_mat in zip(portf_w_list, hold_rtn_mat_list):

        porftw_mat = day_portfw_on_period(portf_w, period_rtn_mat)

        portf_rtn_arr = (porftw_mat * period_rtn_mat).sum(axis=0)

        # 第一个 小于等于 -1.0 的 portf_rtn, 亏掉所有deposit, 无论前面还剩多少
        # 所以整个回测停止于此

        # np.where returns a tuple of coords
        check_ = np.where(portf_rtn_arr <= -1.0)[0]

        if check_.__len__() > 0: # if -1 rtn happens
            stop_ind = check_[0] # backtest stops here
            portf_rtn_arr = portf_rtn_arr[:stop_ind+1]
            day_rtn_lst.extend( list(portf_rtn_arr) )
            cost_lst.append(0)

            break
        else: # if -1 rtn not happens
            day_rtn_lst.extend( list(portf_rtn_arr) )
            cost_lst.append(0)

    portf_rtn_arr = np.array(day_rtn_lst)
    trade_days = len(day_rtn_lst)
    total_cost = sum(cost_lst)
    gross_rtn = np.prod(1+portf_rtn_arr) - 1
    maxdd = maxdrawdown(portf_rtn_arr, mode='rtnrate')

    res = {
        'rtn': ( invest_amount * gross_rtn - total_cost )/ \
                            invest_amount,
        'var': np.var(portf_rtn_arr),
        'trade_days': trade_days,
        'total_cost': total_cost,
        'gross_rtn': gross_rtn,
        'maxdd': maxdd
        }
    
    return res

