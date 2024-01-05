from Code.Forecaster.BlackLitterman import BlackLitterman
from Code.Allocator.MeanVarOptimal import MeanVarOpt
from Code.Allocator.RiskParity import RiskParity

import numpy as np
from typing import Callable
import traceback


# mean-variance optimal
def load_data_mvopt(low_constraints, high_constraints, rtn_data_loader:Callable, **data_kwargs):
    rtn_data, assets_inds = rtn_data_loader(**data_kwargs)
    cov_mat = np.cov(rtn_data)
    rtn_rates = rtn_data.mean(axis=1)
    if low_constraints is not None:
        low_constraints = np.array(low_constraints)
    if high_constraints is not None:
        high_constraints = np.array(high_constraints)
    finmodel = MeanVarOpt(rtn_rates, cov_mat, low_constraints, high_constraints, assets_inds)
    return finmodel


def mvopt_portf_var_from_r(r:np.float32, low_constraints, high_constraints, rtn_data_loader:Callable, **data_kwargs):
    try:
        mvopt = load_data_mvopt(low_constraints, high_constraints, rtn_data_loader, **data_kwargs)
        if low_constraints is None and high_constraints is None:
            var_star, weights_star = mvopt.get_portf_var_from_r(r)
            res = {'portf_w':list(weights_star), 'portf_var':var_star, 'portf_r':r,
                   'assets_inds':mvopt.assets_inds}
        else:
            # {"portf_w":np_array, "portf_var":float,
            #  "portf_r":float,    "qp_status":string}
            solution = mvopt.solve_constrained_qp_from_r(r)
            res = {'portf_w':list(solution['portf_w']), 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
                   "qp_status":solution['qp_status'], 'assets_inds':mvopt.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    return res


def mvopt_portf_r_from_var(var:np.float32, low_constraints, high_constraints, rtn_data_loader:Callable, **data_kwargs):
    try:
        mvopt = load_data_mvopt(low_constraints, high_constraints, rtn_data_loader, **data_kwargs)
        if low_constraints is None and high_constraints is None:
            r_star, weights_star = mvopt.get_portf_r_from_var(var)
            res = {'portf_w':list(weights_star), 'portf_var':var, 'portf_r':r_star, 
                   'assets_inds':mvopt.assets_inds}
        else:
            # {"portf_w":np_array, "portf_var":float,
            #  "portf_r":float,    "qp_status":string}
            solution = mvopt.solve_constrained_qp_from_var(var)
            res = {'portf_w':list(solution['portf_w']), 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
                   "qp_status":solution['qp_status'], 'assets_inds':mvopt.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    return res


# black-litterman
# bl_args_dict = {'risk_avers_factor':risk_avers_factor,
#                 'equi_wght_vec':equi_wght_vec,
#                 'tau':0.05}
def load_data_blkltm(view_pick_mat, view_rtn_vec,
                     bl_args_dict,
                     low_constraints, high_constraints, 
                     rtn_data_loader:Callable, **data_kwargs):
    if low_constraints is not None:
        low_constraints = np.array(low_constraints)
    if high_constraints is not None:
        high_constraints = np.array(high_constraints)
    view_pick_mat = np.array(view_pick_mat)
    view_rtn_vec = np.array(view_rtn_vec)
    rtn_data, assets_inds = rtn_data_loader(**data_kwargs)
    bl_args_dict['hist_cov_mat'] = np.cov(rtn_data)
    bl_args_dict['equi_wght_vec'] = np.array(bl_args_dict['equi_wght_vec'])
    bl_model = BlackLitterman(view_pick_mat, view_rtn_vec, normalize=False, assets_inds=assets_inds)
    expct_rtn_rates = bl_model(bl_args_dict)
    finmodel = MeanVarOpt(expct_rtn_rates, bl_args_dict['hist_cov_mat'],
                          low_constraints, high_constraints,
                          assets_inds=bl_model.assets_inds)
    return finmodel


def blkltm_portf_var_from_r(r:np.float32,
                            view_pick_mat, view_rtn_vec, bl_args_dict, 
                            low_constraints, high_constraints,
                            rtn_data_loader:Callable, **data_kwargs):
    try:
        blkltm = load_data_blkltm(view_pick_mat, view_rtn_vec, bl_args_dict,
                                  low_constraints, high_constraints,
                                  rtn_data_loader, **data_kwargs)
        if low_constraints is None and high_constraints is None:
            var_star, weights_star = blkltm.get_portf_var_from_r(r)
            res = {'portf_w':list(weights_star), 'portf_var':var_star, 'portf_r':r,
                   'assets_inds':blkltm.assets_inds}
        else:
            # {"portf_w":np_array, "portf_var":float,
            #  "portf_r":float,    "qp_status":string}
            solution = blkltm.solve_constrained_qp_from_r(r)
            res = {'portf_w':list(solution['portf_w']), 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
                   "qp_status":solution['qp_status'], 'assets_inds':blkltm.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    return res



def blkltm_portf_r_from_var(var:np.float32,
                            view_pick_mat, view_rtn_vec, bl_args_dict, 
                            low_constraints, high_constraints,
                            rtn_data_loader:Callable, **data_kwargs):
    try:
        blkltm = load_data_blkltm(view_pick_mat, view_rtn_vec, bl_args_dict,
                                  low_constraints, high_constraints,
                                  rtn_data_loader, **data_kwargs)
        if low_constraints is None and high_constraints is None:
            r_star, weights_star = blkltm.get_portf_r_from_var(var)
            res = {'portf_w':list(weights_star), 'portf_var':var, 'portf_r':r_star,
                   'assets_inds':blkltm.assets_inds}
        else:
            # {"portf_w":np_array, "portf_var":float,
            #  "portf_r":float,    "qp_status":string}
            solution = blkltm.solve_constrained_qp_from_var(var)
            res = {'portf_w':list(solution['portf_w']), 'portf_var':solution['portf_var'], 'portf_r':solution['portf_r'],
                   "qp_status":solution['qp_status'], 'assets_inds':blkltm.assets_inds}
    except Exception as e:
        traceback.print_exc()
        res = {"err_msg":str(e), 'status':'fail'}
    return res




# risk parity
def get_riskparity(category_mat, rtn_data_loader:Callable, **data_kwargs):
    rtn_data, assets_inds = rtn_data_loader(**data_kwargs)
    if category_mat is not None:
        category_mat = np.array(category_mat)
    fin = RiskParity(rtn_data, category_mat, assets_inds=assets_inds)
    portf_w = fin.optimal_solver()
    res = {'portf_w':list(portf_w), 'portf_var':fin.risk_contribs.sum(), 'portf_r':fin.portf_return,
           'risk_contribs':list(fin.risk_contribs), 'assets_inds':fin.assets_inds}
    return res



# risk budget
def get_riskbudget(category_mat, tgt_contrib_ratio, rtn_data_loader:Callable, **data_kwargs):
    rtn_data, assets_inds = rtn_data_loader(**data_kwargs)
    if category_mat is not None:
        category_mat = np.array(category_mat)
    tgt_contrib_ratio = np.array(tgt_contrib_ratio)
    fin = RiskParity(rtn_data, category_mat, tgt_contrib_ratio, assets_inds=assets_inds)
    portf_w = fin.optimal_solver()
    res = {'portf_w':list(portf_w), 'portf_var':fin.risk_contribs.sum(), 'portf_r':fin.portf_return,
           'risk_contribs':list(fin.risk_contribs),'assets_inds':fin.assets_inds}
    return res
