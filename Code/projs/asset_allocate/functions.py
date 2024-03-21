import numpy as np
from Code.Utils.DateTime import (
    first_day_of_week,
    first_day_of_month,
    first_day_of_quarter,
    first_day_of_year
)
from typing import Callable

def cal_rtn_intervals(
        hold_dates_arr: np.ndarray,
        portf_rtn_arr: np.ndarray,
        bchmk_rtn_arr: np.ndarray,
        termidate: str,
        ):
    assert hold_dates_arr.dtype == int, f'wrong dates datatype'

    def get_rtn(
            rtn_arr:np.ndarray,
            first_day_getter:Callable):
        
        # 取出 某日到termidate（包括）的序号
        itv_ind_ = np.argwhere(hold_dates_arr >= int(first_day_getter(termidate)))
        itv_rtn_arr = rtn_arr[itv_ind_]
        return np.prod(1+itv_rtn_arr) - 1

    return {
            'portf_rtn': {
                "today": portf_rtn_arr[-1],
                "thisweek": get_rtn(portf_rtn_arr, first_day_of_week),
                "thismonth": get_rtn(portf_rtn_arr, first_day_of_month),
                "thisquarter": get_rtn(portf_rtn_arr, first_day_of_quarter),
                "thisyear": get_rtn(portf_rtn_arr, first_day_of_year),
                "recent1y": np.prod(1+portf_rtn_arr[-240:]) - 1,
                "recent3y": np.prod(1+portf_rtn_arr[-720:]) - 1
            },
            'bchmk_rtn': {
                "today": bchmk_rtn_arr[-1],
                "thisweek": get_rtn(bchmk_rtn_arr, first_day_of_week),
                "thismonth": get_rtn(bchmk_rtn_arr, first_day_of_month),
                "thisquarter": get_rtn(bchmk_rtn_arr, first_day_of_quarter),
                "thisyear": get_rtn(bchmk_rtn_arr, first_day_of_year),
                "recent1y": np.prod(1+bchmk_rtn_arr[-240:]) - 1,
                "recent3y": np.prod(1+bchmk_rtn_arr[-720:]) - 1
            }
        }





def cal_rtn_years(
        hold_dates_arr: np.ndarray,
        portf_rtn_arr: np.ndarray,
        bchmk_rtn_arr: np.ndarray):
    
    assert hold_dates_arr.dtype == int, f'wrong dates datatype'

    years = hold_dates_arr//10000
    uniq_years = np.unique(years)

    result = []

    for year in uniq_years:
        
        cur_year_portf_rtn = portf_rtn_arr[years==year]
        cur_year_bchmk_rtn = bchmk_rtn_arr[years==year]

        result.append(
            {
                'year':year,
                'portf_rtn': np.prod(1+cur_year_portf_rtn) - 1,
                'bchmk_rtn': np.prod(1+cur_year_bchmk_rtn) - 1,
            }
        )
    
    return result