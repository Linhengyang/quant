from Code.Forecasters.BlackLitterman import BlackLitterman
from Code.Allocators.MeanVarOptimal import MeanVarOpt
from Code.Allocators.RiskParity import RiskParity
from Code.DataLoaders.random4test import rdm_rtn_data

import numpy as np
from typing import Callable


# mean-variance optimal
def load_data_mvopt(low_constraints, high_constraints, rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    cov_mat = np.cov(rtn_data)
    rtn_rates = rtn_data.mean(axis=1)
    if low_constraints is not None:
        low_constraints = np.array(low_constraints)
    if high_constraints is not None:
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
    if low_constraints is not None:
        low_constraints = np.array(low_constraints)
    if high_constraints is not None:
        high_constraints = np.array(high_constraints)
    return MeanVarOpt(expct_rtn_rates, bl_args_dict['hist_cov_mat'], low_constraints, high_constraints)


def mvopt_portf_var_from_r(mvopt:MeanVarOpt, r:np.float32):
    var_star, weights_star = mvopt.get_portf_var_from_r(r)
    res = {'portf_var':var_star, 'portf_w':weights_star}
    return res


def mvopt_portf_r_from_var(mvopt:MeanVarOpt, var:np.float32):
    r_star, weights_star = mvopt.get_portf_r_from_var(var)
    res = {'portf_r':r_star, 'portf_w':weights_star}
    return res


def mvopt_constrained_qp_from_r(mvopt:MeanVarOpt, goal_r:np.float32):
    # {"portf_w":np_array, "portf_var":float,
    #  "portf_r":float,    "qp_status":string}
    return mvopt.solve_constrained_qp_from_r(goal_r)


def mvopt_constrained_qp_from_var(mvopt:MeanVarOpt, goal_var:np.float32):
    # {"portf_w":np_array, "portf_var":float,
    #  "portf_r":float,    "qp_status":string}
    return mvopt.solve_constrained_qp_from_var(goal_var)

# risk parity
def load_data_riskparity(category_mat, rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    if category_mat is not None:
        category_mat = np.array(category_mat)
    return RiskParity(rtn_data, category_mat)

# risk budget
def load_data_riskbudget(category_mat, tgt_contrib_ratio, rtn_data_loader:Callable, **data_kwargs):
    rtn_data = rtn_data_loader(**data_kwargs)
    if category_mat is not None:
        category_mat = np.array(category_mat)
    tgt_contrib_ratio = np.array(tgt_contrib_ratio)
    return RiskParity(rtn_data, category_mat, tgt_contrib_ratio)


def risk_ctrl(rp:RiskParity):
    portf_w = rp.optimal_solver()
    res = {'portf_r':rp.portf_return, 'risk_contribs':rp.risk_contribs, 'portf_w':portf_w}
    return res