import numpy as np




# 贝叶斯观点下的对资产收益率向量的预测
class BlackLitterman:
    '''
    Black-Litterman模型，归根结底是利用贝叶斯理论，给出了一种融合「当前/均衡/客观现状」和「新信息/动向/主观观点」，以「预测」的办法。
    其中「当前/均衡/客观现状」对应「先验/prior」，「新信息/动向/主观观点」对应「采样/sampling」，「预测」对应「后验/posterior」。
    多维正态分布的贝叶斯估计：https://stats.stackexchange.com/questions/28744/multivariate-normal-posterior
    BL模型预测资产的收益率向量
    '''
    def __init__(self, view_pick_mat:np.ndarray, view_rtn_vec:np.ndarray, normalize=False):
        # views
        self._view_pick_mat_orthog_flag = self.view_pick_mat_row_orthog_check(view_pick_mat) # 观点矩阵是否正交
        self._view_var_mat = None # Omega
        self._view_precs_mat = None # 是 view_var_mat 的 inverse
        # normalization
        if normalize:
            normalize = np.diag( np.sqrt( np.diag( view_pick_mat @ view_pick_mat.T ) ) )
            self._view_pick_mat = normalize @ view_pick_mat
            self._view_rtn_vec = normalize @ view_rtn_vec
        else:
            self._view_pick_mat =  view_pick_mat
            self._view_rtn_vec = view_rtn_vec
        # priors
        self._prior_cov_mat = None # Phi0
        self._prior_precs_mat = None # 是 prior_cov_mat 的 inverse
        self._prior_rtn_vec = None
    
    def set_prior_rtn_vec(self, args_dict={}, method='equilium', *args, **kwargs):
        if method == 'equilium':
            self._prior_rtn_vec = self.equi_rtn_vec(args_dict['risk_avers_factor'], args_dict['hist_cov_mat'], args_dict['equi_wght_vec'])
        elif method == 'other':
            raise NotImplementedError('other method for assets prior return vector not implemented')
    
    @property
    def view_pick_mat_orthog_flag(self):
        return self._view_pick_mat_orthog_flag

    @property
    def prior_rtn_vec(self):
        return self._prior_rtn_vec
    
    def set_prior_cov_precs_mat(self, args_dict={}, method='shrink', *args, **kwargs):
        '''
        资产的先验 协方差-精度矩阵
        '''
        if method == 'shrink':
            self._prior_cov_mat = self.prior_cov_mat_shrink(args_dict['hist_cov_mat'], args_dict['tau'])
            self._prior_precs_mat = np.linalg.inv(self._prior_cov_mat)
        elif method == 'other':
            raise NotImplementedError('other method for assets prior covariance & precision matrix not implemented')
        else:
            raise ValueError('unknown method {}'.format(method))
    
    @property
    def prior_precs_mat(self):
        return self._prior_precs_mat

    @property
    def prior_cov_mat(self):
        return self._prior_cov_mat

    def set_view_cov_precs_mat(self, args_dict={}, method='default', *args, **kwargs):
        '''
        观点views的不确定性矩阵
        '''
        if method == 'default':
            self._view_var_mat = self.view_var_mat_diag(self._view_pick_mat, self._prior_cov_mat)
            self._view_precs_mat = np.diag( 1.0 / np.diag(self._view_var_mat) )
        elif method == 'default-nondiag':
            self._view_var_mat = self.view_var_mat_diag(self._view_pick_mat, self._prior_cov_mat, diagnal=False)
            self._view_precs_mat = np.linalg.inv(self._view_var_mat)
        elif method == 'idzorek-confidence':
            raise NotImplementedError('confidence method for view covariance & precision matrix not implemented')
        elif method == 'residual-variance':
            raise NotImplementedError('residual-variance method for view covariance & precision matrix not implemented')
        else:
            raise ValueError('Unknown method {}'.format(method))

    @property
    def view_var_mat(self):
        return self._view_var_mat
    
    @property
    def view_precs_mat(self):
        return self._view_precs_mat

    #### build BL model

    def __call__(self, args_dict):
        self.set_prior_rtn_vec(args_dict)
        self.set_prior_cov_precs_mat(args_dict)
        self.set_view_cov_precs_mat(args_dict)
        return self.EXPE_return_BL(self._prior_precs_mat, self._prior_rtn_vec, self._view_pick_mat, self._view_precs_mat, self._view_rtn_vec)
    
    #### compute functions

    @staticmethod
    def EXPE_return_BL(prior_precs_mat:np.ndarray, prior_rtn_vec:np.ndarray, view_pick_mat:np.ndarray, view_precs_mat:np.ndarray, view_rtn_vec:np.ndarray):
        return np.linalg.inv( prior_precs_mat + view_pick_mat.T@view_precs_mat@view_pick_mat ) @ (prior_precs_mat @ prior_rtn_vec + view_pick_mat.T@view_precs_mat@view_rtn_vec)
    
    @staticmethod
    def equi_rtn_vec(risk_avers_factor:float, cov_mat:np.ndarray, equi_wght_vec:np.ndarray):
        return risk_avers_factor * cov_mat @ equi_wght_vec
    
    @staticmethod
    def prior_cov_mat_shrink(hist_cov_mat:np.ndarray, tau:float=0.05):
        return tau * hist_cov_mat
    
    @staticmethod
    def view_var_mat_diag(view_pick_mat:np.ndarray, prior_asset_cov_mat:np.ndarray, diagnal=True):
        if diagnal:
            return np.diag( np.diag( view_pick_mat @ prior_asset_cov_mat @ view_pick_mat.T ) )
        else:
            return view_pick_mat @ prior_asset_cov_mat @ view_pick_mat.T
    
    @staticmethod
    def view_pick_mat_row_orthog_check(view_pick_mat:np.ndarray):
        dot_prod = view_pick_mat @ view_pick_mat.T # 计算 P @ P'
        np.fill_diagonal( dot_prod, 0.0) # 替换主对角线元素为 0.0
        return np.allclose(dot_prod, 0.0) # 对比是否为 0矩阵


















if __name__ == "__main__":
    np.random.seed(100)
    hist_rtn_series = np.random.uniform(-1, 1, size=(5, 180))
    hist_cov_mat = np.cov(hist_rtn_series)
    risk_avers_factor = 0.3
    equi_wght_vec = np.array([0.03, 0.1, 0.2, 0.08, 0.12])

    view_pick_mat = np.array([[1, -1, 0, 0, 0],
                              [0, 0, 1, -0.5, -0.5]])
    view_rtn_vec = np.array([0.01, 0.03])

    bl_model = BlackLitterman(view_pick_mat, view_rtn_vec)
    args_dict = {'risk_avers_factor':risk_avers_factor,
                 'equi_wght_vec':equi_wght_vec,
                 'hist_cov_mat':hist_cov_mat,
                 'tau':0.05}
    
    print('current returns: ', hist_rtn_series.mean(axis=1))
    print( 'BL expected returns: ', bl_model(args_dict) )