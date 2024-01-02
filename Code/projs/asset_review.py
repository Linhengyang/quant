from Code.Forecasters.BlackLitterman import BlackLitterman
from Code.Allocators.MeanVarOptimal import MeanVarOpt
from Code.Allocators.RiskParity import RiskParity
from Code.DataLoaders.random4test import *

from typing import Callable

def load_data_mvopt(low_constraints, high_constraints, rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    cov_mat = np.cov(rtn_data)
    rtn_rates = rtn_data.mean(axis=1)
    low_constraints = np.array(low_constraints)
    high_constraints = np.array(high_constraints)
    return MeanVarOpt(rtn_rates, cov_mat, low_constraints, high_constraints)


def mvopt_portf_var_from_r(mvopt:MeanVarOpt, r:np.float32):
    return mvopt.get_portf_var_from_r(r)


def mvopt_portf_r_from_var(mvopt:MeanVarOpt, var:np.float32):
    return mvopt.get_portf_var_from_r(var)


def mvopt_constrained_qp_from_r(mvopt:MeanVarOpt, goal_r:np.float32):
    return mvopt.solve_constrained_qp_from_r(goal_r)


def mvopt_constrained_qp_from_var(mvopt:MeanVarOpt, goal_var:np.float32):
    return mvopt.solve_constrained_qp_from_var(goal_var)


