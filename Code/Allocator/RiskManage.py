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
import scipy.optimize as scipyopt
import typing as t
from Code.Utils.Type import basicPortfSolveRes
import numpy as np


def total_weight_constraint(x):
    return np.sum(x) - 1.0

def loan_only_constraint(x):
    return x

def lower_bounds_constraint(x, l_b):
    return x - l_b

def upper_bounds_constraint(x, u_b):
    return u_b - x


class RiskManage:
    '''
    return:
    basicPortfSolveRes
    {
        'portf_w': np.ndarray
        'portf_rtn': np.floating
        'portf_var': np.floating
        'solve_status': str
        'assets_idlst': list
    }
    '''


    __slots__ = ("assets_idlst",
                "__solve_status", "__portf_w", "__portf_var", "__portf_rtn", 
                "__asset_r_mat", "__cov_mat", "__category_mat", "__tgt_contrib_ratio",
                "__low_constraints", "__high_constraints",
                "__num_risk", "__num_assets",)
    

    def __init__(
            self,
            asset_r_mat: np.ndarray,
            category_mat: t.Union[np.ndarray, None],
            tgt_contrib_ratio: t.Union[np.ndarray, None],
            constraints: t.List[t.Union[np.ndarray, None]],
            assets_idlst: list
            ) -> None:
        
        self.assets_idlst = assets_idlst # 记录资产的排列

        self.__asset_r_mat = asset_r_mat
        self.__num_assets = asset_r_mat.shape[0] # 资产个数

        self.__cov_mat = np.cov(asset_r_mat)

        if category_mat is not None:

            categ_mat_col_sum = np.unique( category_mat.sum(axis=0) )

            assert len( categ_mat_col_sum ) == 1 and categ_mat_col_sum[0] == 1, \
                    "Category Matrix must satisfy column sum shall be 1"
            
        self.__category_mat = category_mat

        # 风险的个数
        self.__num_risk = asset_r_mat.shape[0] if category_mat is None else category_mat.shape[0]

        if tgt_contrib_ratio is not None:
            # 风险预算

            # 检查1: 风险的个数 等于 tgt_contrib_ratio的长度
            assert self.__num_risk == len(tgt_contrib_ratio),\
                f'risk number {self.__num_risk} mismatch target ratio {len(tgt_contrib_ratio)}'
            
            # 检查2: tgt_contrib_ratio 之和要接近1
            assert np.abs(tgt_contrib_ratio.sum() - 1) <= 0.01,\
                f"tgt_contrib_ratio must be summed to 1, now is {tgt_contrib_ratio.sum()}"
            
        self.__tgt_contrib_ratio = tgt_contrib_ratio

        self.__low_constraints, self.__high_constraints = constraints

        self.__solve_status: str = "" # 初始化求解状态为空字符

        self.__portf_w: np.ndarray = np.array([]) # portfolio 实际权重 待求解
        self.__portf_var: np.floating = np.float32(-1) # porfolio 实际var 待求解
        self.__portf_rtn: np.floating = np.float32(0) # porfolio 实际rtn 待求解

    
    @property
    def portf_var(self) -> np.floating:
        if self.__portf_var != np.float32(-1):
            return self.__portf_var
        else:
            return self.__portf_w @ self.__cov_mat @ self.__portf_w
    
    @property
    def portf_w(self) -> np.ndarray:
        if len(self.__portf_w) > 0:
            return self.__portf_w
        else:
            raise NotImplementedError('portf_w not calculated')
    
    @property
    def portf_rtn(self) -> np.floating:
        if self.__portf_rtn != np.float32(0):
            return self.__portf_rtn
        else:
            return self.__asset_r_mat.mean(axis=1) @ self.__portf_w

    @property
    def solve_status(self) -> str:
        if self.__solve_status != '':
            return self.__solve_status
        else:
            raise NotImplementedError('solve status not obtained')


    @staticmethod
    def __cal_optimal_obj(
            risk_contribs: np.ndarray,
            tgt_contrib_ratio: t.Union[np.ndarray, None]
            ) -> np.floating:

        # risk_contribs: 各资产/风险因子/资产类别 的trc向量, 即各资产/风险因子/资产类别贡献的风险。
        # tgt_contrib_ratio: 希望各资产/风险因子/资产类别 风险贡献的目标比例。不输入时，默认各资产/风险因子/资产类别平均分担总风险
        if tgt_contrib_ratio is None:

            risk_contribs = risk_contribs.reshape(-1, 1)

            ones = np.ones_like(risk_contribs)

            obj = np.sum(
                np.square(risk_contribs @ ones.T - ones @ risk_contribs.T)
                )
        else:

            tgt_contribs = tgt_contrib_ratio * np.sum(risk_contribs)

            obj = np.sum(
                np.square(risk_contribs - tgt_contribs)
                )

        return obj
    

    @staticmethod
    def __cal_risk_contribs(
        w:np.ndarray,
        V:np.ndarray,
        category_mat: t.Union[np.ndarray, None]
        ) -> np.ndarray:

        if category_mat is None:
            return w * (V @ w)
        
        return category_mat @ (w * (V @ w))
    

    @property
    def risk_contribs(self):
        return self.__cal_risk_contribs(
            self.__portf_w,
            self.__cov_mat,
            self.__category_mat
            )
    

    def obj_func_on_assets(self, w, params):
        # params最少包括1个参数，最多包括3个参数，分别是：
        # 1：V 资产的covariance矩阵
        # 2: category_mat 资产的类别归属矩阵
        # 3: tgt_contrib_ratio 风险预算比例
        V, category_mat, tgt_contrib_ratio = params

        risk_contribs = self.__cal_risk_contribs(w, V, category_mat)

        return self.__cal_optimal_obj(risk_contribs, tgt_contrib_ratio)
    

    def __call__(
            self
            ) -> basicPortfSolveRes:
        
        # 初始猜测 均分点
        w0 = np.array([1/self.__num_assets ] * self.__num_assets)

        # 如果下限是 None, 设定为 all 0.0
        if self.__low_constraints is None:
            lower_bounds = np.array([0.0] * self.__num_assets)
        else:
            lower_bounds = self.__low_constraints
        
        # 如果上限是 None, 设定为 all 1.0
        if self.__high_constraints is None:
            upper_bounds = np.array([1.0] * self.__num_assets)
        else:
            upper_bounds = self.__high_constraints
        
        cons = (
            {'type': 'eq', 'fun': total_weight_constraint},
            {'type': 'ineq', 'fun':loan_only_constraint},
            {'type': 'ineq', 'fun':lower_bounds_constraint, 'args':(lower_bounds,)},
            {'type': 'ineq', 'fun':upper_bounds_constraint, 'args':(upper_bounds,)},
            )
        
        res = scipyopt.minimize(self.obj_func_on_assets, w0,
                                args=[self.__cov_mat,
                                      self.__category_mat,
                                      self.__tgt_contrib_ratio],
                                method='SLSQP',
                                constraints=cons,
                                options={'disp':True})
        
        self.__portf_w = res.x

        self.__solve_status = 'optimal' if res.success else 'unknown'

        return {
            "portf_w": self.portf_w,
            "portf_rtn": self.portf_rtn,
            "portf_var": self.risk_contribs.sum(),
            "solve_status": self.solve_status,
            'assets_idlst': self.assets_idlst
            }
    











if __name__ == "__main__":
    np.random.seed(10)
    x1 = np.random.uniform(low=-1, high=1, size=(3, 180))
    x2 = np.random.uniform(low=-2, high=2, size=(2, 180))
    x3 = np.random.uniform(low=-3, high=3, size=(1, 180))
    x = np.concatenate([x1, x2, x3], axis=0)
    fin = RiskManage(x,
                     category_mat = np.array([[1,1,1,0,0,0],
                                              [0,0,0,1,1,0],
                                              [0,0,0,0,0,1]]),
                     tgt_contrib_ratio= np.array([1/3]*3),
                     constraints = [np.array([0]*6), np.array([1]*6)],

                     assets_idlst = ['0001.SH', '0002.SH', '0003.SH', '0004.SH', '0005.SH', '0006.SH']
                     )
    w_rb = fin()
    print('asset weights: ', w_rb)
    print('risk contributions: ', fin.risk_contribs)
    print('portf return: ', fin.portf_rtn)