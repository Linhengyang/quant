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

class RiskParity:
    def __init__(self, ):
        pass

    def cal_portf_var(w:np.array, V:np.array):
        pass

    def cal_risk_contrib_obj(risk_contribs:np.array, tgt_contrib_ratio:np.array=None):
        # risk_contribs: 各资产/风险因子/资产类别 的trc向量, 即各资产/风险因子/资产类别贡献的风险。
        # tgt_contrib_ratio: 希望各资产/风险因子/资产类别 风险贡献的目标比例。不输入时，默认各资产/风险因子/资产类别平均分担总风险
        num_risk = len(risk_contribs)
        if tgt_contrib_ratio is None:
            tgt_contrib_ratio = np.array([ 1.0/num_risk ]*num_risk)
        assert len(tgt_contrib_ratio) == num_risk, "risk_contribs and tgt_contrib_ratio must have same length"
        assert np.abs( tgt_contrib_ratio.sum() - 1 ) <= 0.01, "tgt_contrib_ratio must be summed to 1"

        


if __name__ == "__main__":
    x = np.array([1/7] * 7)
    y = x.sum()
    print( y )