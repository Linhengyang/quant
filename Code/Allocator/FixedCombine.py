# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name:         RiskParity
   Description :      风险管理: 给定各资产波动率, 求解波动率平价/预算的portfolio
   Author :           linhengyang
   Create date:       2023/12/15
   Latest version:    v1.0.0
-------------------------------------------------

"""
import typing as t
from Code.Utils.Type import basicPortfSolveRes
import numpy as np


class FixedCombo:
    '''
    return:
    basicPortfSolveRes
    {
        'portf_w': np.ndarray
        'solve_status': str
        'assets_idlst': list
    }
    '''


    __slots__ = ("assets_idlst",
                "__solve_status", "__portf_w", "__portf_var", "__portf_rtn", 
                "__asset_r_mat", "__num_assets",)
    

    def __init__(
            self,
            asset_r_mat: np.ndarray,
            fixed_weights: t.Union[np.ndarray, None],
            assets_idlst: list
            ) -> None:
        
        self.assets_idlst = assets_idlst # 记录资产的排列

        self.__asset_r_mat = asset_r_mat
        self.__num_assets = asset_r_mat.shape[0] # 资产个数
        
        self.__solve_status: str = "direct" # 初始化求解状态为空字符
        
        if fixed_weights is not None:
            self.__portf_w: np.ndarray = fixed_weights
        else:
            # 当输入固定权重为None时，默认为平均分配
            self.__portf_w = np.repeat(1/self.__num_assets, self.__num_assets)

        self.__portf_var: np.floating = np.float32(-1) # porfolio 实际var 待求解
        self.__portf_rtn: np.floating = np.float32(0) # porfolio 实际rtn 待求解

    
    @property
    def portf_var(self) -> np.floating:
        return self.__portf_w @ np.cov(self.__asset_r_mat) @ self.__portf_w
    

    @property
    def portf_w(self) -> np.ndarray:
        return self.__portf_w
    

    @property
    def portf_rtn(self) -> np.floating:
        return self.__asset_r_mat.mean(axis=1) @ self.__portf_w


    @property
    def solve_status(self) -> str:
        return self.__solve_status

    

    def __call__(
            self
            ) -> basicPortfSolveRes:

        return {
            "portf_w": self.portf_w,
            "solve_status": self.solve_status,
            'assets_idlst': self.assets_idlst
            }