from typing import TypedDict
import numpy as np





class basicPortfSolveRes(TypedDict):
    portf_w: np.array
    portf_rtn: np.float32
    portf_var: np.float32
    solve_status: str
    assets_idlst: list





class basicBackTestRes(TypedDict):
    rtn: np.float32
    var: np.float32
    trade_days: int
    total_cost: np.float32
    gross_rtn: np.float32

