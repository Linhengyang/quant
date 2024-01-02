from Code.projs.asset_review import *

if __name__ == "__main__":
    np.random.seed(100)
    num_assets = 5
    back_window_size = 180
    low_constraints = np.array([0.0]*num_assets)
    high_constraints = np.array([1.0]*num_assets)
    mvopt = load_data_mvopt(low_constraints=low_constraints, high_constraints=high_constraints,
                            rtn_data_loader=rdm_rtn_data, num_assets=5, back_window_size=180)
    print(mvopt_constrained_qp_from_var(mvopt, 5.342786287220995))


