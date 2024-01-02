from Code.Forecasters.BlackLitterman import BlackLitterman
from Code.Allocators.MeanVarOptimal import MeanVarOpt
from Code.Allocators.RiskParity import RiskParity
from Code.DataLoaders.random4test import *

from typing import Callable


# mean-variance optimal
def load_data_mvopt(low_constraints, high_constraints, rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    cov_mat = np.cov(rtn_data)
    rtn_rates = rtn_data.mean(axis=1)
    low_constraints = np.array(low_constraints)
    high_constraints = np.array(high_constraints)
    return MeanVarOpt(rtn_rates, cov_mat, low_constraints, high_constraints)


# black-litterman
# bl_args_dict = {'risk_avers_factor':risk_avers_factor,
#                 'equi_wght_vec':equi_wght_vec,
#                 'tau':0.05}
def load_data_blkltm(view_pick_mat, view_rtn_vec, bl_args_dict, low_constraints, high_constraints, 
                     rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    bl_args_dict['hist_cov_mat'] = np.cov(rtn_data)
    bl_model = BlackLitterman(view_pick_mat, view_rtn_vec)
    expct_rtn_rates = bl_model(bl_args_dict)
    low_constraints = np.array(low_constraints)
    high_constraints = np.array(high_constraints)
    return MeanVarOpt(expct_rtn_rates, bl_args_dict['hist_cov_mat'], low_constraints, high_constraints)


def mvopt_portf_var_from_r(mvopt:MeanVarOpt, r:np.float32):
    return mvopt.get_portf_var_from_r(r)


def mvopt_portf_r_from_var(mvopt:MeanVarOpt, var:np.float32):
    return mvopt.get_portf_var_from_r(var)


def mvopt_constrained_qp_from_r(mvopt:MeanVarOpt, goal_r:np.float32):
    return mvopt.solve_constrained_qp_from_r(goal_r)


def mvopt_constrained_qp_from_var(mvopt:MeanVarOpt, goal_var:np.float32):
    return mvopt.solve_constrained_qp_from_var(goal_var)

# risk parity
def load_data_riskparity(category_mat, rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    return RiskParity(rtn_data, category_mat)

# risk budget
def load_data_riskbudget(category_mat, tgt_contrib_ratio:np.array, rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    return RiskParity(rtn_data, category_mat, tgt_contrib_ratio)

def risk_asign_portf_w(rp:RiskParity):
    return rp.optimal_solver()

def risk_asign_actual_risk_contribs(rp:RiskParity):
    if rp.allocated_weights is None:
        rp.optimal_solver()
    return rp.risk_contribs


def risk_asign_actual_portf_r(rp:RiskParity):
    if rp.allocated_weights is None:
        rp.optimal_solver()
    return rp.portf_return
