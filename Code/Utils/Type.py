from typing import TypedDict
import numpy as np
import typing as t






class basicPortfSolveRes(TypedDict):
    portf_w: np.ndarray
    solve_status: str
    assets_idlst: list





class basicBackTestRes(TypedDict):
    rtn: np.floating
    var: np.floating
    trade_days: int
    total_cost: np.floating
    gross_rtn: np.floating
    drawdown: np.floating
