from typing import TypedDict
import numpy as np






class basicPortfSolveRes(TypedDict):
    portf_w: np.ndarray
    portf_rtn: np.floating
    portf_var: np.floating
    solve_status: str
    assets_idlst: list





class basicBackTestRes(TypedDict):
    rtn: np.floating
    var: np.floating
    trade_days: int
    total_cost: np.floating
    gross_rtn: np.floating

