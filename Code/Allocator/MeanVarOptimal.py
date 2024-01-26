# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name:         MeanVarOptimal
   Description :      均值-方差最优化：给定预期收益率/波动率，求解波动率/预期收益率最优化的portfolio
   Author :           linhengyang
   Create date:       2023/12/11
   Latest version:    v1.0.0
-------------------------------------------------

"""

import numpy as np
import cvxopt

## 均值-方差最优化求解器
class MeanVarOpt:
    def __init__(
            self,
            expct_rtn_rates:np.array,
            expct_cov_mat:np.array,
            low_constraints:np.array=None,
            high_constraints:np.array=None,
            assets_idlst:list=[]
            ):
        
        self.assets_idlst = assets_idlst # 记录资产的排列
        # 检查条件0: 预期收益率向量长度等于协方差矩阵的维度
        assert len(expct_rtn_rates) == expct_cov_mat.shape[0],\
            "Assets number conflicts between returns & covariance"
        
        self.expct_rtn_rates = expct_rtn_rates
        self.expct_cov_mat = expct_cov_mat
        self.goal_rtn_rate_portf = None # 目标portfolio预期收益率待设定

        self._build_quad_curve() # 已经足够画出mean-var曲线

        # 不等式约束（上下限）输入检查
        if low_constraints is not None: # 对权重下限作检查
            # 检查条件1：权重下限之和要小于等于1
            assert low_constraints.sum() <= 1.0,\
                "Sum of low bounds of weights of assets must be smaller or equal to 1"
        #     # 检查条件2：每个权重下限都要大于等于0且小于等于1（即不允许卖空）
        #     assert all(low_constraints >= 0.0), "Low bounds must be larger or equal to 0"
        #     assert all(low_constraints <= 1.0), "Low bounds must be smaller or equal to 1"
        # if high_constraints is not None:
        #     # 检查条件3：每个权重上限都要大于等于0且小于等于1（即不允许卖空）
        #     assert all(high_constraints >= 0.0), "High bounds must be larger or equal to 0"
        #     assert all(high_constraints <= 1.0), "High bounds must be smaller or equal to 1"
        if low_constraints  is not None and high_constraints is not None:
            # 检查条件4：每个权重下限都要小于等于各自权重上限
            assert all(low_constraints <= high_constraints),\
                "low bound must be be smaller or equal to high bound"
            
        self.low_constraints = low_constraints
        self.high_constraints = high_constraints

        self._build_quad_program()

    def _build_quad_curve(self):
        # 组建不带不等式约束的经典mean-var二次曲线所需要参数
        self.cov_mat_inv = np.linalg.inv(self.expct_cov_mat)
        ones = np.ones_like(self.expct_rtn_rates)
        self.c = ones @ self.cov_mat_inv @ ones
        self.b = self.expct_rtn_rates @ self.cov_mat_inv @ self.expct_rtn_rates
        self.a = ones @ self.cov_mat_inv @ self.expct_rtn_rates
        self.d = self.b*self.c - self.a*self.a
        # var最小的return-var点是(var=1/c, r=a/c)

    def _build_quad_program(self):
        # 组建带不等式约束的二次规划所需要的参数（除了预期收益率参数b）
        ## P: 二次规划-目标函数中的正定矩阵
        self._P = self.expct_cov_mat.astype(np.float64)
        ## q: 二次规划-目标函数中的一次项系数
        self._q = np.zeros_like(self.expct_rtn_rates).astype(np.float64)
        ## A: 二次规划-等式约束中的系数矩阵.有两个等式约束：1、以未定元为权重的加权预期收益率为goal_rtn_rate_portf，2、未定元相加之和为1
        ## 第1个约束要等到goal_rtn_rate_portf加进来之后
        self._A = np.stack( [self.expct_rtn_rates, np.ones_like(self.expct_rtn_rates)], axis=0).astype(np.float64)
        ## G: 二次规划-不等式约束中的系数矩阵. 有两个不等式约束：1、下限，2、上限
        self._num_assets = len(self.expct_rtn_rates)

        if self.low_constraints is not None and self.high_constraints is None: # 下限给定，上限未给时
            self.high_constraints = np.ones_like(self.low_constraints) # 默认上限是小于等于1.0
        elif self.low_constraints is None and self.high_constraints is not None: # 下限未给，上限给定时
            self.low_constraints = np.zeros_like(self.high_constraints) # 默认下限是大于等于0.0
        else: # 上下限都未给 或 上下限都已给出，不作处理
            pass
        if self.low_constraints is not None and self.high_constraints is not None:
            self._G = np.concatenate([-np.eye(self._num_assets), np.eye(self._num_assets)], axis=0).astype(np.float64)
            self._h = np.concatenate( [-self.low_constraints, self.high_constraints], axis=0).astype(np.float64)
        else: # 上下限都未给时，默认为无不等式约束
            self._G = None
            self._h = None

    # 不考虑不等式约束，根据给定的预期收益率r，直接得到 最优var和最优protf权重
    def get_portf_var_from_r(self, r:np.float32):
        if r < self.a/self.c:
            raise ValueError(
                f"Minimum expected return rate for current combination of assets is {self.a / self.c}"
                )
        ones = np.ones_like(self.expct_rtn_rates)
        weights_star = r * 1.0/self.d * self.cov_mat_inv @ ( self.c*self.expct_rtn_rates - self.a*ones ) + \
                       1.0/self.d * self.cov_mat_inv @ ( self.b*ones - self.a*self.expct_rtn_rates )
        var_star = 1.0 / self.d * (self.c*r*r - 2*self.a*r + self.b)

        return (var_star, weights_star)

    # 不考虑不等式约束，根据给定的预期波动率var，直接得到 最优收益率r和最优protf权重
    def get_portf_r_from_var(self, var:np.float32):
        if var < 1.0/self.c:
            raise ValueError(
                f"Minimum expected variance for current combination of assets is {1.0 / self.c}"
                )
        r_star = self.a/self.c +\
                 np.sqrt( self.d/self.c * ( var + self.a*self.a/(self.d*self.c) - self.b/self.d ) )
        _, weights_star = self.get_portf_var_from_r(r_star)
        return (r_star, weights_star)

    # 考虑不等式约束，根据给定的预期收益率r(即满足至少要r的预期收益率)，求解portfolio波动最小的protf权重，以及此时的var
    def solve_constrained_qp_from_r(self, goal_r:np.float32):
        if goal_r < self.a/self.c:
            raise ValueError(
                f"Minimum expected return rate for current combination of assets is {self.a / self.c}"
                )
        self.goal_rtn_rate_portf = goal_r
        self._b = np.array([self.goal_rtn_rate_portf, 1.0]).astype(np.float64)
        qp_args = [self._P, self._q, self._G, self._h, self._A, self._b]
        qp_args = [cvxopt.matrix(i) if i is not None else None for i in qp_args]
        qp_result = cvxopt.solvers.qp(*qp_args)
        # 返回解出的portf权重，此时达到的最优（小）var, 此时达到的最优（大）r（可能比goal_r要大），求解status
        return {"portf_w":np.array(qp_result['x']).squeeze(1),
                "portf_var":2.0*qp_result['primal objective'],
                "portf_rtn":(self.expct_rtn_rates @ np.array(qp_result['x'])).item(),
                "qp_status":qp_result['status']
                }

    # 考虑不等式约束，根据给定的预期波动var(即能承受的最低波动var)，求解portfolio预期收益最大的protf权重，以及此时的r
    def solve_constrained_qp_from_var(self, goal_var:np.float32):
        if goal_var < 1.0/self.c:
            raise ValueError(
                f"Minimum expected variance for current combination of assets is {1.0 / self.c}"
                )
        self.goal_rtn_rate_portf = self.get_portf_r_from_var(goal_var)[0]
        self._b = np.array([self.goal_rtn_rate_portf, 1.0]).astype(np.float64)
        qp_args = [self._P, self._q, self._G, self._h, self._A, self._b]
        qp_args = [cvxopt.matrix(i) if i is not None else None for i in qp_args]
        qp_result = cvxopt.solvers.qp(*qp_args)
        # 返回解出的portf权重，此时达到的最优（小）var（可能比goal_var要大）, 此时达到的最优（大）r，求解status
        return {"portf_w": np.array(qp_result['x']).squeeze(1),
                "portf_var": 2.0 * qp_result['primal objective'],
                "portf_rtn": (self.expct_rtn_rates @ np.array(qp_result['x'])).item(),
                "qp_status": qp_result['status']
                }

































if __name__ == "__main__":
    np.random.seed(100)
    back_window_size = 180  # 回看的交易日窗口天数
    num_assets = 5  # 考虑的资产总个数
    # 历史资产收益率矩阵, size = (num_assets, bach_window_size), 每行是某资产在某历史交易日的收益率
    h_rtn_rate_mat = np.random.uniform(low=-10, high=10, size=(num_assets, back_window_size))
    # 历史资产收益率的协方差矩阵（方差矩阵）
    h_cov_mat = np.cov(m=h_rtn_rate_mat)
    # 预期资产收益率向量, size = (num_assets,)：在这里直接使用历史平均收益率（算术平均）作为预测
    expct_rtn_rate_vec = h_rtn_rate_mat.mean(axis=1)
    # 预期资产收益率的协方差矩阵（方差矩阵）：在这里直接使用历史收益率的协方差矩阵作为预测
    expct_cov_mat = h_cov_mat

    low_constraints = np.array([0.0]*num_assets)
    high_constraints = np.array([1.0] * num_assets)
    constraints = [None, None]
    # constraints = []
    model = MeanVarOpt(expct_rtn_rate_vec, expct_cov_mat, *constraints)
    # print(model.get_portf_var_from_r(goal_r=0.020017337450874609))
    # result = model.solve_constrained_qp_from_r(goal_r=0.020017337450874608 )
    result = model.get_portf_r_from_var(var=5.342786287220995)
    print( result )


# 求portfolio 夏普比率（return_rate of portfolio - risk-free rate）/ std of portfolio