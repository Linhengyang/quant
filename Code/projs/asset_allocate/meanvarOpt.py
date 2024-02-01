import numpy as np
import traceback
import typing as t
from typing import Any
from operator import itemgetter

from Code.Allocator.MeanVarOptimal import MeanVarOpt
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
    get_constraints
    )
from Code.Utils.Decorator import (
    deDilate,
    addAnnual,
    addSTD
    )




class meanvarOptStrat:
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
    @addSTD('var')
    def backtest(self) -> dict:
        '''
        de-dilated
            'rtn': np.float32
            'var': np.float32,
            'std': np.float32
            'trade_days': int,
            'total_cost': float,
            'gross_rtn': np.float32
            'annual_rtn': np.float32
        '''

        train_rtn_mat_list, hold_rtn_mat_list, self.__assets_idlst, constraints, self.__flag,\
            expt_tgt_value = self._get_meanvar_data_params()
        
        num_assets = len(self.__assets_idlst)

        # 初始化为平均分配
        self.__portf_w_list, self.__detail_solve_results = \
            [np.repeat(1/num_assets, num_assets), ], []

        for train_rtn_mat in train_rtn_mat_list:

            cur_res = self.__solve_single_mvopt(
                train_rtn_mat,
                self.__assets_idlst,
                constraints,
                self.__flag,
                expt_tgt_value
                )

            if cur_res['solve_status'] in ('direct', 'qp_optimal'):
                self.__portf_w_list.append( cur_res['portf_w'] )
            else:
                self.__portf_w_list.append( self.__portf_w_list[-1] )
            
            self.__detail_solve_results.append(cur_res)



        # 在 basicBT_multiPeriods 中，由于涉及到复利累乘，所以需要考虑 1 + de-dilated rtn
        # 所以必须在这里传入 de-dilated hold_rtn_mat. 在这之后，BT的结果不需要de-dilate
        hold_rtn_mat_list = [ hold_rtn_mat/dilate for hold_rtn_mat in hold_rtn_mat_list ]

        BT_mvopt = basicBT_multiPeriods(self.__portf_w_list[1:], hold_rtn_mat_list)
        
        return BT_mvopt




    @property
    def assets_idlst(self) -> list:
        return self.__assets_idlst




    @property
    def flag(self) -> str:
        return self.__flag




    @property
    def portf_w_list(self) -> list:
        return self.__portf_w_list[1:]




    @property
    def detail_solve_results(self) -> list:
        return self.__detail_solve_results
    



    def _get_meanvar_data_params(self) -> Any:
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

        mvo_target, expt_tgt_value = \
            itemgetter("mvo_target", "expt_tgt_value")(self.__inputs)
        
        print(
            f'Back Test for mean-variance-optimal {mvo_target} strategy \
            from {begindate} to {termidate} trading on every {gapday} \
            upon {len(assets_dict)} assets')

        expt_tgt_value = np.float32(expt_tgt_value)

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
               constraints, mvo_target, expt_tgt_value



    @staticmethod
    @addSTD('portf_var')
    @deDilate(dilate)
    def __solve_single_mvopt(
        train_rtn_mat: np.ndarray,
        assets_idlst: t.List[str],
        constraints: t.List[t.Union[np.ndarray, None]],
        mvo_target: str,
        expt_tgt_value: np.float32,
        ) -> Any:
        '''
        input:
            train_rtn_mat: np.ndarray,
            assets_idlst: t.List[str],
            constraints: t.List[t.Union[np.ndarray, None]],
            mvo_target: str,
            expt_tgt_value: np.float32,
        return:
        de-dilate
            portf_w: np.ndarray
            portf_rtn: np.float32
            portf_var: np.float32
            portf_std: np.float32
            solve_status: str
            assets_idlst: list
        '''
        cov_mat = np.cov(train_rtn_mat)
        rtn_rates = train_rtn_mat.mean(axis=1)


        fin = MeanVarOpt(rtn_rates, cov_mat, constraints, assets_idlst)
        
        res = fin(expt_tgt_value, mvo_target)
        try:
            fin = MeanVarOpt(rtn_rates, cov_mat, constraints, assets_idlst)
            
            res = fin(expt_tgt_value, mvo_target)
            
        except Exception as e:
            traceback.print_exc()

            res = {
                'portf_w': np.array([]),
                'portf_rtn': 0,
                'portf_var': -1,
                'solve_status': 'FAIL_' + str(e),
                'assets_idlst': assets_idlst
                }
        
        return res