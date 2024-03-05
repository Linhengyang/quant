import numpy as np
import traceback
import typing as t
from typing import Any
from operator import itemgetter

from Code.Allocator.RiskManage import RiskManage
from Code.projs.asset_allocate.dataLoad import (
    get_train_hold_rtn_data, 
    _DB,
    _MKT_DATE_TABLE
    )
from Code.BackTester.BT_AssetAllocate import (
    basicBT_multiPeriods
    )
from Code.projs.asset_allocate.runner import *
from Code.projs.asset_allocate.inputParser import (
    parseAssets2dicts,
    get_constraints,
    get_tgt_risk_ratios
    )
from Code.Utils.Decorator import (
    tagFunc,
    deDilate,
    addAnnual,
    addSTD
    )



class riskMngStrat:
    '''
    attributes:
        1. assets_idlst
        2. flag
        3. portf_w_list
        4. detail_solve_results
    
    methods:
        1. backtest()  get backtest result
    '''


    __slots__ = ("__inputs", "__assets_idlst", "__flag",  "__portf_w_list", "__detail_solve_results")



    def __init__(self,
                 inputs: Any
                 ) -> None:
        
        self.__inputs = inputs
        self.__assets_idlst = []
        self.__portf_w_list = []
        self.__detail_solve_results = []
        self.__flag = ''



    @addAnnual('rtn', begindate, termidate)
    @tagFunc('dedilated')
    @addSTD('var')
    def backtest(self) -> dict:
        '''
        de-dilated
            'rtn': np.floating
            'var': np.floating,
            'std': np.floating
            'trade_days': int,
            'total_cost': float,
            'gross_rtn': np.floating
            'annual_rtn': np.floating
        '''

        train_rtn_mat_list, hold_rtn_mat_list, self.__assets_idlst, constraints, self.__flag,\
            tgt_contrib_ratio = self._get_riskmng_data_params()
        
        num_assets = len(self.__assets_idlst)

        # 初始化为平均分配
        self.__portf_w_list, self.__detail_solve_results = \
            [np.repeat(1/num_assets, num_assets), ], []

        for i, train_rtn_mat in enumerate(train_rtn_mat_list):

            cur_res = self.__solve_single_riskmng(
                train_rtn_mat,
                self.__assets_idlst,
                constraints,
                tgt_contrib_ratio
                )
            cur_res['position_no'] = i + 1

            if cur_res['solve_status'] in ('optimal', ):
                self.__portf_w_list.append( cur_res['portf_w'] )
            else:
                self.__portf_w_list.append( self.__portf_w_list[-1] )
            
            self.__detail_solve_results.append(cur_res)

        # 在 basicBT_multiPeriods 中，由于涉及到复利累乘，所以需要考虑 1+de-dilated rtn
        # 所以必须在这里传入 de-dilated hold_rtn_mat. 在这之后，BT的结果不需要de-dilate
        hold_rtn_mat_list = [ hold_rtn_mat/dilate for hold_rtn_mat in hold_rtn_mat_list ]

        return basicBT_multiPeriods(self.__portf_w_list[1:], hold_rtn_mat_list)
    


    def _get_riskmng_data_params(self) -> Any:
        '''
        return:
            train_rtn_mat_list: list of ndarray
            hold_rtn_mat_list: list of ndarray
            assets_idlst: list of str
            constraints: list of ndarray or none
            mvo_target: str
            expt_tgt_value: npfloat
        '''
        assets_info_lst = self.__inputs["assets_info"] # assets_info

        assets_dict, src_tbl_dict = parseAssets2dicts(assets_info_lst)

        tgt_contrib_ratio = get_tgt_risk_ratios( assets_dict, assets_idlst )
        
        riskmng_target = "parity" if tgt_contrib_ratio is None else "budget"

        print(
            f'Back Test for risk {riskmng_target} strategy \
            from {begindate} to {termidate} trading on every {gapday} \
            upon {len(assets_dict)} assets')


        tbl_names = list( src_tbl_dict.keys() ) # list of str
        assets_ids = [ src_tbl_dict[tbl] for tbl in tbl_names] # list of lists

        train_rtn_mat_list, hold_rtn_mat_list, assets_idlst = \
            get_train_hold_rtn_data(
                begindate,
                termidate,
                gapday,
                back_window_size,
                dilate,
                assets_ids,
                tbl_names,
                _DB,
                _MKT_DATE_TABLE
            )
        
        constraints = get_constraints( assets_dict, assets_idlst )

        return train_rtn_mat_list, hold_rtn_mat_list, assets_idlst, \
               constraints, riskmng_target, tgt_contrib_ratio
    

    @staticmethod
    @addSTD('portf_var')
    @deDilate(dilate)
    def __solve_single_riskmng(
        train_rtn_mat: np.ndarray,
        assets_idlst: t.List[str],
        constraints: t.List[t.Union[np.ndarray, None]],
        tgt_contrib_ratio: t.Union[np.ndarray, None],
        ) -> Any:
        '''
        input:
            train_rtn_mat: np.ndarray,
            assets_idlst: t.List[str],
            constraints: t.List[t.Union[np.ndarray, None]],
            tgt_contrib_ratio: t.Union[np.ndarray, None]
        \n.
        return:
        de-dilate
            portf_w: np.ndarray
            portf_rtn: np.floating
            portf_var: np.floating
            portf_std: np.floating
            solve_status: str
            assets_idlst: list
        '''
        
        try:
            fin = RiskManage(
                asset_r_mat = train_rtn_mat,
                category_mat = None,
                tgt_contrib_ratio = tgt_contrib_ratio,
                constraints = constraints,
                assets_idlst = assets_idlst
                )
            
            res = fin()
            
        except Exception as e:
            traceback.print_exc()

            res = {
                'portf_w': np.array([]),
                'portf_rtn': 0,
                'portf_var': -dilate,
                'solve_status': 'FAIL_' + str(e),
                'assets_idlst': assets_idlst
                }
        
        return res
    

    def detail_window(self, position_no: int):
        '''
        return every details about one of single position window

        "position_no": int, starts from 1
        "train_rtn_mat": np.ndarray
        "constraints": list of None or np.ndarray
        "riskmng_target": str
        "expt_tgt_value": np.ndarray
        "solve_res": basicPortfSolveRes
        "hold_rtn_mat": np.ndarray
        '''

        train_rtn_mat_list, hold_rtn_mat_list, assets_idlst, constraints, flag,\
            tgt_contrib_ratio = self._get_riskmng_data_params()
        
        assert position_no <= len(train_rtn_mat_list), \
            f"position_no must no larger than {len(train_rtn_mat_list)}"

        train_rtn_mat = train_rtn_mat_list[position_no-1]

        cur_res = self.__solve_single_riskmng(
            train_rtn_mat,
            assets_idlst,
            constraints,
            tgt_contrib_ratio
            )
        
        hold_rtn_mat = hold_rtn_mat_list[position_no-1] / dilate

        return {
            "position_no": position_no,
            "train_rtn_mat": train_rtn_mat,
            "constraints": constraints,
            "riskmng_target": flag,
            "tgt_contrib_ratio": tgt_contrib_ratio,
            "solve_res": cur_res,
            "hold_rtn_mat": hold_rtn_mat
        }