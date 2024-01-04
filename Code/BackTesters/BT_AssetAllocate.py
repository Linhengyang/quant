import numpy as np
from typing import Callable

def rtn_period(portf_w:np.array, period_rtn_mat:np.array):
    assert len(portf_w) == len(period_rtn_mat), 'asset weight array & asset return matrix in period must have same length'
    rtn_array = np.prod((1 + period_rtn_mat), axis=1) - 1
    return (portf_w @ rtn_array)







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