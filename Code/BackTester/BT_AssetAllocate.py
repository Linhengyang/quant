import numpy as np
from typing import Callable
from datetime import datetime

def rtn_period(portf_w:np.array, period_rtn_mat:np.array):
    assert len(portf_w) == len(period_rtn_mat),\
        'asset weight array with size {w_size} must match return matrix in period with asset size {rtn_size}'.format(
            w_size=len(portf_w), rtn_size=len(period_rtn_mat)
            )
    rtn_array = np.prod((1 + period_rtn_mat), axis=1) - 1
    return (portf_w @ rtn_array)




def reallocate_cost(portf_w_list:list, period_rtn_mat_list:list, invest_amount):
    '''
                                     cost1                                          cost2
    portf_w1  with invest_amount A  ------->  portf_w2 with invest_amount A(1+r1)  -------> portf_w3 with invest_amount A(1+r2)
    
     cost_clear
    ------------>   0 asset with cash
    '''
    return [0]*len(portf_w_list)





def rtn_multi_periods(portf_w_list:list, period_rtn_mat_list:list, trade_cost_list:list=[], invest_amount=False):
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
        'portfolio weight allocations {w} not match actual return matrix records {r}'.format(w=len(portf_w_list), r=len(period_rtn_mat_list))
    rtn_list = []
    trade_days = 0
    for portf_w, period_rtn_mat in zip(portf_w_list, period_rtn_mat_list):
        rtn_list.append( rtn_period(portf_w, period_rtn_mat) )
        trade_days += period_rtn_mat.shape[1]
    
    rtn_array = np.array(rtn_list)
    if not invest_amount:
        invest_amount = 10000
    
    total_cost = sum(trade_cost_list)
    gross_rtn = np.prod(1+rtn_array) - 1

    res = {'rtn': ( invest_amount * gross_rtn - total_cost ) / invest_amount,
           'trade_days': trade_days,
           'total_cost': total_cost,
           'gross_rtn': gross_rtn
           }
    
    return res




def modify_BackTestResult(BT_res, dilate, begindate:str, termidate:str):
    # input BT_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float}
    # rtn 膨胀系数修正 和 年化利率计算
    BT_res['rtn'] = BT_res['rtn']/dilate
    BT_res['gross_rtn'] = BT_res['gross_rtn']/dilate
    delta_year = ( datetime.strptime(termidate, '%Y%m%d') - datetime.strptime(begindate, '%Y%m%d') ).days / 365
    BT_res['annual_rtn'] = np.power( 1 + BT_res['rtn'], 1/delta_year) - 1
    # output BT_res = {'rtn': float, 'trade_days': int,'total_cost': float, 'gross_rtn': float, 'annual_rtn':float}
    return BT_res









if __name__ == "__main__":
    num_assets = 5
    period_days = 3
    period_rtn_mat = np.random.uniform(-1,1, size=(num_assets, period_days))
    portf_w = np.array([1.0]*5)
    print(period_rtn_mat)
    print('---------------------------------------------------------------')
    r = rtn_period(portf_w, period_rtn_mat)
    print(type(r))
    print(r)