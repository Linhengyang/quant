from Code.projs.asset_review import *

if __name__ == "__main__":
    np.random.seed(100)
    num_assets = 5
    back_window_size = 180
    # mean-var opt
    low_constraints = np.array([0.0]*num_assets)
    high_constraints = np.array([1.0]*num_assets)
    # mvopt = load_data_mvopt(low_constraints=low_constraints, high_constraints=high_constraints,
    #                         rtn_data_loader=rdm_rtn_data, num_assets=5, back_window_size=180)
    # print(mvopt_constrained_qp_from_var(mvopt, 5.342786287220995))
    # black-litterman
    bl_args_dict = {'risk_avers_factor':0.3,
                    'equi_wght_vec':[0.03, 0.1, 0.2, 0.08, 0.12],
                    'tau':0.05}
    view_pick_mat = np.array([[1, -1, 0, 0, 0],
                              [0, 0, 1, -0.5, -0.5]])
    view_rtn_vec = np.array([0.01, 0.03])

    bl = load_data_blkltm(view_pick_mat, view_rtn_vec, bl_args_dict, low_constraints, high_constraints,
                          rdm_rtn_data, num_assets=5, back_window_size=180)
    print(mvopt_constrained_qp_from_var(bl, 5.342786287220995))
    # # risk parity
    # rp = load_data_riskparity(None, rdm_rtn_data, num_assets=5, back_window_size=180)
    # w_rb = rp.optimal_solver()
    # print('asset weights: ', w_rb)
    # print('risk contributions: ', rp.risk_contribs)
    # print('portf return: ', rp.portf_return)