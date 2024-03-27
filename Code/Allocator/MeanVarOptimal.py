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
import typing as t
from scipy.optimize import linprog
from Code.Utils.Type import basicPortfSolveRes
from Code.Utils.Statistic import (
    multiCoLinear
    )

## 均值-方差最优化求解器
class MeanVarOpt:
    '''
    return:
    basicPortfSolveRes
    {
        'portf_w': np.ndarray
        'solve_status': str
        'assets_idlst': list
    }
    '''


    __slots__ = ("assets_idlst", "__low_constraints", "__high_constraints",
                 "__no_bounds", "__expct_rtn_rates", "__expct_cov_mat", 
                 "__solve_status", "__portf_w", "__portf_var", "__portf_rtn", 
                 "__cov_mat_inv", "__quad_term", "__const_term", "__lin_term", 
                 "__norm_term", "__vertex", "__num_assets", "__tangency",
                 "__P", "__q", "__A", "__G", "__h", "__b")


    def __init__(
            self,
            expct_rtn_rates: np.ndarray,
            expct_cov_mat: np.ndarray,
            constraints: t.List[t.Union[np.ndarray, None]],
            assets_idlst: list
            ) -> None:
        
        self.assets_idlst = assets_idlst # 记录资产的排列
        # 下限，上限
        self.__low_constraints, self.__high_constraints = constraints
        
        self.__no_bounds = self.__low_constraints is None and self.__high_constraints is None

        # 检查条件0: 预期收益率向量长度等于协方差矩阵的维度
        assert len(expct_rtn_rates) == expct_cov_mat.shape[0],\
            "Assets number conflicts between returns & covariance"
        
        # 检查条件1: 共线性检查
        colinear = multiCoLinear(expct_cov_mat, assets_idlst)
        assert len(colinear) == 0, \
            f'Co-Linearity found with assets {str(colinear)}'


        self.__expct_rtn_rates: np.ndarray = expct_rtn_rates
        self.__expct_cov_mat: np.ndarray = expct_cov_mat

        self.__build_quad_curve() # 已经足够画出mean-var曲线

        self.__build_quad_program()
        self.__solve_status: str = "" # 求解状态

        self.__portf_w: np.ndarray = np.array([]) # portfolio 实际权重 待求解
        self.__portf_var: np.floating = np.float32(-1) # porfolio 实际var 待求解
        self.__portf_rtn: np.floating = np.float32(0) # porfolio 实际rtn 待求解


    def __build_quad_curve(self) -> None:
        # 组建不带不等式约束的经典mean-var二次曲线所需要参数
        '''
        var = 1/norm_term * (qua_term * r^2 - 2 * lin_term * r + cons_term )
        var = 1/d * (c * r^2 - 2 * a * r + b )
        d = norm_term
        c = quad_term
        a = lin_term
        b = cons_term
        '''

        self.__cov_mat_inv = np.linalg.inv(self.__expct_cov_mat)
        ones = np.ones_like(self.__expct_rtn_rates)
        self.__quad_term = ones @ self.__cov_mat_inv @ ones
        self.__const_term = self.__expct_rtn_rates @ \
                            self.__cov_mat_inv @ \
                            self.__expct_rtn_rates
        self.__lin_term = ones @ self.__cov_mat_inv @ self.__expct_rtn_rates
        self.__norm_term = self.__const_term*self.__quad_term -\
                            np.power(self.__lin_term, 2)
        
        # var最小的return-var点是(var=1/c, r=a/c)
        self.__vertex = (1.0/self.__quad_term,
                         self.__lin_term/self.__quad_term)
        

    def __build_quad_program(self) -> None:
        # 组建带不等式约束的二次规划所需要的参数(除了预期收益率参数b)
        '''
        Minimize obj = 1/2 * x @ P @ x + q @ x
        subject to G @ x <= h, A @ x = b
        '''
        ## P: 二次规划-目标函数中的正定矩阵
        self.__P = self.__expct_cov_mat.astype(np.float64)
        ## q: 二次规划-目标函数中的一次项系数
        self.__q = np.zeros_like(self.__expct_rtn_rates).astype(np.float64)
        ## A: 二次规划-等式约束中的系数矩阵.有两个等式约束：1、以未定元为权重的加权预期收益率为goal_r，2、未定元相加之和为1
        ## 第1个约束要等到goal_r加进来之后
        self.__A = np.stack(
                            [self.__expct_rtn_rates, np.ones_like(self.__expct_rtn_rates)],
                             axis=0
                             ).astype(np.float64)
        ## G: 二次规划-不等式约束中的系数矩阵. 有两个不等式约束：1、下限，2、上限
        self.__num_assets = len(self.__expct_rtn_rates)

        if not self.__no_bounds:
            self.__G = np.concatenate(
                                      [-np.eye(self.__num_assets), np.eye(self.__num_assets)],
                                      axis=0
                                      ).astype(np.float64)
            self.__h = np.concatenate(
                                      [-self.__low_constraints, self.__high_constraints],
                                      axis=0
                                      ).astype(np.float64)
        else: # 上下限都未给时，默认为无不等式约束
            self.__G = None
            self.__h = None


    def __build_riskf_tangt_line(self, riskf) -> None:
        # 组建不等式约束的经典mean-var二次曲线，与无风险利率点的切线

        ## 首先要建立经典rtn-std曲线(x轴是std，y轴是rtn)
        ## 该曲线所有参数都可从 mean-var二次抛物线中得到

        def std2rtn_curve(x: np.floating):

            return self.__lin_term / self.__quad_term + \
                np.sqrt(
                    self.__norm_term / self.__quad_term * (np.power(x, 2) - 1.0/self.__quad_term)
                    )
        
        ## 切点x坐标是 sqrt( 1/c * ( 1 + d/(a-rf)^2 ) ), 即sharpe最高时的波动率std
        tangent_x = np.sqrt(
            (1 + self.__norm_term / np.power(self.__lin_term - riskf, 2) )  /  \
                                self.__quad_term
        )

        ## sharpe最大时的 std = tangent_x， rtn = std2rtn_curve(tangent_x)
        self.__tangency = tangent_x, std2rtn_curve(tangent_x)



    @property
    def portf_rtn(self) -> np.floating:
        if self.__portf_rtn != np.float32(0):
            return self.__portf_rtn
        else:
            return self.__expct_rtn_rates @ self.__portf_w
    
    @property
    def portf_var(self) -> np.floating:
        if self.__portf_var != np.float32(-1):
            return self.__portf_var
        else:
            return self.__portf_w @ self.__expct_cov_mat @ self.__portf_w

    @property
    def portf_w(self) -> np.ndarray:
        if len(self.__portf_w) > 0:
            return self.__portf_w
        else:
            raise NotImplementedError('portf_w not calculated')

    @property
    def solve_status(self) -> str:
        if self.__solve_status != '':
            return self.__solve_status
        else:
            raise NotImplementedError('solve status not obtained')


    @staticmethod
    def __cal_portf_w_unbounds_from_rtn(
        goal_r: np.floating,
        expct_rtn_rates: np.ndarray,
        cov_mat_inv: np.ndarray,
        norm_term: np.floating,
        quad_term: np.floating,
        lin_term: np.floating,
        const_term: np.floating) -> np.ndarray:

        ones = np.ones_like(expct_rtn_rates)
        portf_w =  \
            goal_r * 1.0 / norm_term * cov_mat_inv @ ( quad_term * expct_rtn_rates - lin_term * ones ) \
            + \
            1.0 / norm_term * cov_mat_inv @ ( const_term * ones - lin_term * expct_rtn_rates )
        
        assert not np.isnan(portf_w).any(),\
            f'NaN calculation on __cal_portf_w_unbounds_from_rtn'
        
        return portf_w

    @staticmethod
    def __cal_portf_var_unbounds_from_rtn(
        goal_r: np.floating,
        norm_term: np.floating,
        quad_term: np.floating,
        lin_term: np.floating,
        const_term: np.floating) -> np.floating:

        portf_var = 1.0 / norm_term * \
            (quad_term * np.power(goal_r, 2) - 2 * lin_term * goal_r + const_term)
        
        assert not np.isnan(portf_var),\
            f'NaN calculation on __cal_portf_var_unbounds_from_rtn'
        
        return portf_var

    @staticmethod
    def __cal_portf_rtn_unbounds_from_var(
        goal_var: np.floating,
        norm_term: np.floating,
        quad_term: np.floating,
        lin_term: np.floating,
        const_term: np.floating) -> np.floating:

        porft_rtn = lin_term/quad_term + \
            np.sqrt(
                norm_term/quad_term *\
                (goal_var + np.power(lin_term, 2)/(norm_term*quad_term) - const_term/norm_term)
                )
        
        assert not np.isnan(porft_rtn),\
            f'NaN calculation on __cal_portf_rtn_unbounds_from_var'
        
        return porft_rtn

    # 不考虑不等式约束，根据给定的预期收益率r，直接得到 最优var和最优protf权重
    def __get_portf_unbounds_from_rtn(
            self,
            goal_r:np.floating) -> None:
        
        if goal_r < self.__vertex[1]:
            raise ValueError(
                f"minimum expected target return value(after dilate) for "
                f"this process is {round(self.__vertex[1],3)}. Raise goal return"
                )
        
        self.__portf_w = self.__cal_portf_w_unbounds_from_rtn(
            goal_r,
            self.__expct_rtn_rates,
            self.__cov_mat_inv,
            self.__norm_term,
            self.__quad_term,
            self.__lin_term,
            self.__const_term
        )
        
        self.__portf_var = self.__cal_portf_var_unbounds_from_rtn(
            goal_r,
            self.__norm_term,
            self.__quad_term,
            self.__lin_term,
            self.__const_term
        )
        
        self.__portf_rtn = goal_r

        self.__solve_status = "direct"

    # 不考虑不等式约束，根据给定的预期波动率var，直接得到 最优收益率r和最优protf权重
    def __get_portf_unbounds_from_var(
            self,
            goal_var:np.floating) -> None:
        
        if goal_var < self.__vertex[0]:
            raise ValueError(
                f'minimum expected target variance value(after dilate) for '
                f'this process is {round(self.__vertex[0],3)}. Raise goal variance'
                )
        
        self.__portf_rtn = self.__cal_portf_rtn_unbounds_from_var(
            goal_var,
            self.__norm_term,
            self.__quad_term,
            self.__lin_term,
            self.__const_term
        )

        self.__portf_w = self.__cal_portf_w_unbounds_from_rtn(
            self.__portf_rtn,
            self.__expct_rtn_rates,
            self.__cov_mat_inv,
            self.__norm_term,
            self.__quad_term,
            self.__lin_term,
            self.__const_term
        )

        self.__portf_var = goal_var

        self.__solve_status = "direct"


    # 考虑不等式约束，根据给定的预期收益率r(即满足至少要r的预期收益率)，求解portfolio波动最小的protf权重，以及此时的var
    def __get_portf_bounds_from_rtn(
            self,
            goal_r:np.floating) -> None:
        
        if goal_r < self.__vertex[1]:
            raise ValueError(
                f"minimum expected target return value(after dilate) for "
                f"this process is {round(self.__vertex[1],3)}. Raise goal return"
                )
        
        self.__b = np.array([goal_r, 1.0]).astype(np.float64)
        qp_args = [self.__P, self.__q, self.__G, self.__h, self.__A, self.__b]
        qp_args = [cvxopt.matrix(i) if i is not None else None for i in qp_args]
        qp_result = cvxopt.solvers.qp(*qp_args)

        self.__portf_w = np.array(qp_result['x']).squeeze(1)

        self.__portf_var = np.float32( 2.0 * qp_result['primal objective'] )

        self.__portf_rtn = self.__expct_rtn_rates @ self.__portf_w

        self.__solve_status = "qp_" + qp_result['status']


    # 考虑不等式约束，根据给定的预期波动var(即能承受的最低波动var)，求解portfolio预期收益最大的protf权重，以及此时的r
    def __get_portf_bounds_from_var(
            self,
            goal_var:np.floating) -> None:
        
        if goal_var < self.__vertex[0]:
            raise ValueError(
                f"minimum expected target variance value(after dilate) for "
                f"this process is {round(self.__vertex[0],3)}. Raise goal variance"
                )
        
        goal_r = self.__cal_portf_rtn_unbounds_from_var(
            goal_var,
            self.__norm_term,
            self.__quad_term,
            self.__lin_term,
            self.__const_term
            )

        self.__b = np.array([goal_r, 1.0]).astype(np.float64)
        qp_args = [self.__P, self.__q, self.__G, self.__h, self.__A, self.__b]
        qp_args = [cvxopt.matrix(i) if i is not None else None for i in qp_args]
        qp_result = cvxopt.solvers.qp(*qp_args)

        self.__portf_w = np.array(qp_result['x']).squeeze(1)

        self.__portf_var = np.float32( 2.0 * qp_result['primal objective'] )

        self.__portf_rtn = self.__expct_rtn_rates @ self.__portf_w

        self.__solve_status = "qp_" + qp_result['status']


    # 不考虑不等式约束，直接得到sharpe最高时的 收益率r和protf权重
    def __get_portf_unbounds_sharpe(
            self,
            riskf:np.floating) -> None:
        
        self.__build_riskf_tangt_line(riskf)

        self.__portf_var = np.power( self.__tangency[0], 2)

        self.__portf_rtn = self.__tangency[1]

        self.__portf_w = self.__cal_portf_w_unbounds_from_rtn(
            self.__portf_rtn,
            self.__expct_rtn_rates,
            self.__cov_mat_inv,
            self.__norm_term,
            self.__quad_term,
            self.__lin_term,
            self.__const_term
        )

        self.__solve_status = "direct"


    # 考虑不等式约束，运用线性规划，找到离切点最近的上下两个可行域内的点，有更大斜率的端点就是解
    def __get_portf_bounds_sharpe(
            self,
            riskf:np.floating) -> None:
        
        self.__build_riskf_tangt_line(riskf)

        r_vec = self.__expct_rtn_rates.astype(np.float64)
        # 切点上方
        c_vec = r_vec
        A_eq = np.expand_dims(np.ones_like(c_vec), axis=0)
        b_eq = np.array([1.], dtype=np.float64)
        A_ub = np.expand_dims(-c_vec, axis=0)
        b_ub = np.array([-self.__tangency[1]], dtype=np.float64)
        bounds = list( np.array([self.__low_constraints, self.__high_constraints]).T )

        upper_result = linprog(c_vec, A_ub, b_ub, A_eq, b_eq, bounds)

        # 切点下方
        c_vec = -r_vec
        A_ub = np.stack([r_vec, -r_vec], axis=0)
        b_ub = np.array([self.__tangency[1], self.__vertex[1]], dtype=np.float64)

        lower_result = linprog(c_vec, A_ub, b_ub, A_eq, b_eq, bounds)

        ## 当且仅当两个都成功求解时，对比一下两个候选点，哪个的sharpe更大
        ## 切点坐标是( std, rtn ) = (sqrt(x@cov@x), x@r)
        
        if upper_result.success and lower_result.success:
            upper_sharpe = (upper_result.fun - riskf)/np.sqrt(upper_result.x @ self.__expct_cov_mat @ upper_result.x)
            lower_sharpe = (lower_result.fun - riskf)/np.sqrt(lower_result.x @ self.__expct_cov_mat @ lower_result.x)

            if upper_sharpe >= lower_sharpe:
                result = upper_result
            else:
                result = lower_result
        
        elif upper_result.success:
            result = upper_result
        
        elif lower_result.success:
            result = lower_result
        
        else:
            raise ValueError(f"wrong linear program with risk_free rate {riskf}")
        

        self.__portf_var = np.sqrt(result.x @ self.__expct_cov_mat @ result.x)

        self.__portf_rtn = result.fun

        self.__portf_w = result.x

        self.__solve_status = "lp_success"



    def __call__(
            self,
            tgt_value: np.floating,
            mode: str,
            riskf: np.floating) -> basicPortfSolveRes:
        
        # 计算模式
        if mode not in ['minWave', 'maxReturn', 'sharpe']:
            raise ValueError(
                f'wrong mode for mean-variance optimal with {mode}'
                )
        
        # 按模式求解
        if mode == 'minWave' and self.__no_bounds:
            self.__get_portf_unbounds_from_rtn(tgt_value)
        elif mode == 'minWave':
            self.__get_portf_bounds_from_rtn(tgt_value)
        elif mode == 'maxReturn' and self.__no_bounds:
            self.__get_portf_unbounds_from_var(tgt_value)
        elif mode == 'maxReturn':
            self.__get_portf_bounds_from_var(tgt_value)
        elif mode == 'sharpe' and self.__no_bounds:
            self.__get_portf_unbounds_sharpe(riskf)
        elif mode == 'sharpe':
            self.__get_portf_bounds_sharpe(riskf)
        else:
            raise NotImplementedError('mode {mode} implemented')
        
        return {
            'portf_w': self.portf_w,
            'solve_status': self.solve_status,
            'assets_idlst': self.assets_idlst
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
    constraints = [low_constraints, high_constraints]
    # constraints = [None, None]
    # constraints = []
    model = MeanVarOpt(expct_rtn_rate_vec, expct_cov_mat, constraints, ['test']*num_assets)
    # print(model.get_portf_var_from_r(goal_r=0.020017337450874609))
    # result = model.solve_constrained_qp_from_r(goal_r=0.020017337450874608 )
    result = model(5.342786287220995, 'maxReturn')
    print( result )


# 求portfolio 夏普比率（return_rate of portfolio - risk-free rate）/ std of portfolio