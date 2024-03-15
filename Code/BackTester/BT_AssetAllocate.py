import numpy as np
from typing import Any
import typing as t
from Code.Utils.Type import(
    basicPortfSolveRes,
    basicBackTestRes
)
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
    args:
        初始的各资产权重\n.
        init_portf_w: shape (num_assets, )\n.
        各资产在单个持仓期内的每日收益率\n.
        day_rtn_on_period: shape (num_assets, num_days_period)
    
    return:
        各资产在单个持仓期内的每日权重分配\n.
        day_porftw_mat: shape (num_assets, num_days_period)

    every column is the weight allocation on every day of the portfolio
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



def basicBT_rtnarr_1prd(
        portf_w: np.ndarray,
        hold_rtn_mat: np.ndarray,
        cost: Any = None
        ) -> t.List[Any]:
    '''
    return rates are meaningless after invest amount goes negative

    basicBackTest assumes no further reaction after invest
    amount goes non-positive, and the BackTest stops.

    intput:
        portf_w: weight array
        hold_rtn_mat: ndarray of (num_assets, num_days_in_1_period)
        cost: Any
    return:
        portf_rtn_arr: \n.
        if not early_stop:
            ndarray of portfolio rtn rates in 1 period (num_days_in_1_period, )\n.
        if early_stop:\n.
            ( <= num_days_in_1_period, )\n.
        early_stop: bool to show if the backtest stops(-1 rtn happen) in this period\n.
        cost_arr: ndarray of portfolio cost in 1 period
    '''

    assert len(portf_w) == hold_rtn_mat.shape[0],\
        f'length of portfolio weights {len(portf_w)} \
          not match with actual return rates in hold {hold_rtn_mat.shape[0]}'
    
    # portf_w 是 各资产初始的权重, hold_rtn_mat是持仓期内各资产每天的收益率
    # portfw_mat 是持仓期内各资产每天的权重
    porftw_mat = day_portfw_on_period(portf_w, hold_rtn_mat)

    # portf_rtn_arr是 portfolio在持仓期内的每日收益率
    portf_rtn_arr = (porftw_mat * hold_rtn_mat).sum(axis=0)

    # 第一个 小于等于 -1.0 的 portf_rtn, 亏掉所有deposit, 无论前面还剩多少
    # 所以整个回测停止于此

    # np.where returns a tuple of coords
    check_ = np.where(portf_rtn_arr <= -1.0)[0]

    # early stop sign
    early_stop = False

    if check_.__len__() > 0: # if -1 rtn happens
        stop_ind = check_[0] # backtest stops here
        # -1 rtn day is included
        portf_rtn_arr = portf_rtn_arr[:stop_ind+1]
        early_stop = True
    
    cost_arr = np.array([])

    return [portf_rtn_arr, early_stop, cost_arr]



def BTeval_on_portfrtn(
        portf_rtn_arr: np.ndarray,
        costs: t.Union[np.ndarray, t.List[np.ndarray], None] = None,
        invest_amount: t.Union[bool, float] = False
        ) -> basicBackTestRes:
    '''
    return \n.
    {
        'rtn': rtn,
        'var': np.var(portf_rtn_arr),
        'trade_days': trade_days,
        'total_cost': total_cost,
        'gross_rtn': gross_rtn,
        'drawdown': maxdd,
        'rtn_series': np.ndarray
        }
    '''
    trade_days = len(portf_rtn_arr)

    if isinstance(costs, np.ndarray):
        total_cost = costs.sum()
    elif isinstance(costs, list):
        total_cost = np.concatenate(costs).sum()
    elif costs is None:
        total_cost = 0.0
    else:
        raise TypeError(
            f'costs shall be one of ndarray, list of ndarray, None'
        )
    
    if not invest_amount:
        invest_amount = 10000.0

    gross_rtn = np.prod(1+portf_rtn_arr) - 1

    rtn = ( invest_amount * gross_rtn - total_cost )/ \
                    invest_amount
    
    maxdd = maxdrawdown(portf_rtn_arr, mode='rtnrate')

    trade_days = len(portf_rtn_arr)

    return {
        'rtn': rtn,
        'var': np.var(portf_rtn_arr),
        'trade_days': trade_days,
        'total_cost': total_cost,
        'gross_rtn': gross_rtn,
        'drawdown': maxdd,
        'rtn_series': portf_rtn_arr
        }



def basicBT_eval_Nprd(
        portf_w_lst: t.List[np.ndarray],
        hold_rtn_mat_lst: t.List[np.ndarray],
        cost_lst: t.List[Any] = None,
        invest_amount: t.Union[bool, float] = False
        ) -> basicBackTestRes:
    '''
    return rates are meaningless after invest amount goes negative
    basicBT assumes no further reaction after invest
    amount goes non-positive, and the BackTest stops.

    intput:
        portf_w_lst: list of weight array
        hold_rtn_mat_lst: list of ndarray
        trade_cost_lst: list of Any
    return:
        basicBackTestRes
            rtn: np.floating
            var: np.floating
            trade_days: int
            total_cost: np.floating
            gross_rtn: np.floating
            drawdown: np.floating
            rtn_series: np.ndarray
    '''

    assert len(portf_w_lst) == len(hold_rtn_mat_lst),\
        f'number of periods of portfolio weights {len(portf_w_lst)} \
          not match with actual rtn matrix records {len(hold_rtn_mat_lst)}'
    
    portf_rtn_arr_lst, cost_arr_lst = [], []


    for portf_w, hold_rtn_mat, _ in zip(portf_w_lst, hold_rtn_mat_lst, cost_lst):

        cur_portf_rtns, early_stop, cur_hold_costs = \
            basicBT_rtnarr_1prd(portf_w, hold_rtn_mat, None)

        # 记录当前持仓期的portf_return_rate 和 持仓cost
        portf_rtn_arr_lst.append( cur_portf_rtns )
        cost_arr_lst.append( cur_hold_costs )

        # if -1 rtn happens in this hold
        if early_stop:
            break
    
    portf_rtn_arr = np.concatenate(portf_rtn_arr_lst)

    return BTeval_on_portfrtn(
                portf_rtn_arr,
                cost_arr_lst,
                invest_amount)
    