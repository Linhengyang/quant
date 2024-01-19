from Code.Forecaster.BlackLitterman import BlackLitterman
from Code.Allocator.MeanVarOptimal import MeanVarOpt
from Code.Allocator.RiskParity import RiskParity
from Code.projs.asset_allocate.dataload import db_rtn_data
from flask import Flask, request
from datetime import datetime, timedelta
from pprint import pprint
import numpy as np
from typing import Callable
import traceback


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





def application_blkltm_var_from_r():
    print('/blkltm_var_from_r')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'low_constraints' in inputs:
        low_constraints = inputs['low_constraints']
    else:
        low_constraints = None
    if 'high_constraints' in inputs:
        high_constraints = inputs['high_constraints']
    else:
        high_constraints = None
    res = blkltm_portf_var_from_r(inputs['expt_rtn_rate'],
                                  inputs['view_pick_mat'], inputs['view_rtn_vec'], inputs,
                                  low_constraints, high_constraints,
                                  rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                  startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate'])
    print('output: ')
    pprint(res)
    return res




def application_blkltm_r_from_var():
    print('/blkltm_r_from_var')
    print('[{}]'.format(datetime.now()))
    # postdata = request.get_data()
    # inputs = json.loads(postdata)
    inputs = request.json
    print('input params: ', inputs)
    if 'low_constraints' in inputs:
        low_constraints = inputs['low_constraints']
    else:
        low_constraints = None
    if 'high_constraints' in inputs:
        high_constraints = inputs['high_constraints']
    else:
        high_constraints = None
    res = blkltm_portf_r_from_var(inputs['expt_var'], 
                                  inputs['view_pick_mat'], inputs['view_rtn_vec'], inputs,
                                  low_constraints, high_constraints,
                                  rtn_data_loader=db_rtn_data, assets=inputs['assets_idx'],
                                  startdate=inputs['startdate'], enddate=inputs['enddate'], rtn_dilate=inputs['rtn_dilate'])
    print('output: ')
    pprint(res)
    return res
