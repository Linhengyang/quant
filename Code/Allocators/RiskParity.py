# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name:         RiskParity
   Description :      风险平价：给定各资产波动率，求解波动率平价的portfolio
   Author :           linhengyang
   Create date:       2023/12/15
   Latest version:    v1.0.0
-------------------------------------------------

"""
import scipy
import numpy as np

def total_weight_constraint(x):
    return np.sum(x) - 1.0

def loan_only_constraint(x):
    return x


class RiskParity:
    def __init__(self, asset_r_mat:np.array, category_mat:np.array=None, tgt_contrib_ratio:np.array=None):
        self.asset_r_mat = asset_r_mat
        self.num_assets = asset_r_mat.shape[0] # 资产个数

        self.cov_mat = np.cov(asset_r_mat)

        self.tgt_contrib_ratio = tgt_contrib_ratio
        self.category_mat = category_mat

        if category_mat is not None and tgt_contrib_ratio is not None:
            assert category_mat.shape[0] == len(tgt_contrib_ratio), 'Asset category numbers must match'

        if category_mat is not None:
            self.num_risk = category_mat.shape[0]
        else:
            self.num_risk = asset_r_mat.shape[0]
    
    @staticmethod
    def cal_portf_var(w:np.array, V:np.array):
        # 返回 w @ V @ w.T
        return (w @ V @ w).item()
    
    def cal_optimal_obj(self, risk_contribs:np.array, tgt_contrib_ratio:np.array=None):
        # risk_contribs: 各资产/风险因子/资产类别 的trc向量, 即各资产/风险因子/资产类别贡献的风险。
        # tgt_contrib_ratio: 希望各资产/风险因子/资产类别 风险贡献的目标比例。不输入时，默认各资产/风险因子/资产类别平均分担总风险
        if tgt_contrib_ratio is None:
            tgt_contrib_ratio = np.array([ 1.0/self.num_risk ]*self.num_risk)

        assert len(tgt_contrib_ratio) == self.num_risk, "risk_contribs and tgt_contrib_ratio must have same length"
        assert np.abs( tgt_contrib_ratio.sum() - 1 ) <= 0.01, "tgt_contrib_ratio must be summed to 1"
        # 使用risk_contribs 和 tgt_contrib_ratio 对比, 得到最优化objective函数
        if tgt_contrib_ratio is not None:
            tgt_contribs = tgt_contrib_ratio * np.sum(risk_contribs)
            obj = np.sum( np.square(risk_contribs - tgt_contribs) )
        # 当 tgt_contrib_ratio = None时，使用risk_contribs 自身作对比使得各分量基本相等
        else:
            risk_contribs = risk_contribs.reshape(-1, 1)
            ones = np.ones_like(risk_contribs)
            obj = np.sum( np.square(risk_contribs @ ones.T - ones @ risk_contribs.T) )
        return obj
    
    @staticmethod
    def cal_risk_contribs(w:np.array, V:np.array, category_mat=None):
        if category_mat is None:
            return w * (V @ w)
        assert len( np.unique( category_mat.sum(axis=0) ) ) == 1 and np.unique( category_mat.sum(axis=0) )[0] == 1, \
        "Category Matrix must satisfy column sum shall be 1"
        return category_mat @ (w * (V @ w))
    
    def obj_func_on_assets(self, w, params):
        # params最少包括1个参数，最多包括3个参数，分别是：
        # 1：V 资产的covariance矩阵
        # 2: category_mat 资产的类别归属矩阵
        # 3: tgt_contrib_ratio 风险预算比例
        V, category_mat, tgt_contrib_ratio = params
        risk_contribs = self.cal_risk_contribs(w, V, category_mat)
        return self.cal_optimal_obj(risk_contribs, tgt_contrib_ratio)
    
    def obj_func_on_factor(self, w, params):
        pass

    def optimal_solver(self):
        w0 = np.array([1/self.num_assets ] * self.num_assets ) # 初始值 均分
        cons = ({'type': 'eq', 'fun': total_weight_constraint},
                {'type': 'ineq', 'fun':loan_only_constraint})
        res = scipy.optimize.minimize(self.obj_func_on_assets, w0, args=[self.cov_mat, self.category_mat, self.tgt_contrib_ratio],\
                                      method='SLSQP', constraints=cons, options={'disp':True})
        w_rb = np.asmatrix(res.x)
        return w_rb



if __name__ == "__main__":
    np.random.seed(10)
    x1 = np.random.uniform(low=-1, high=1, size=(3, 180))
    x2 = np.random.uniform(low=-2, high=2, size=(2, 180))
    x3 = np.random.uniform(low=-3, high=3, size=(1, 180))
    x = np.concatenate([x1, x2, x3], axis=0)
    fin = RiskParity(x,
                     category_mat = np.array([[1,1,1,0,0,0],
                                              [0,0,0,1,1,0],
                                              [0,0,0,0,0,1]]),
                     tgt_contrib_ratio= np.array([1/6]*6)
                     )
    print(fin.optimal_solver())