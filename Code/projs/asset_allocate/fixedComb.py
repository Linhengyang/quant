import numpy as np
import traceback
import typing as t
from typing import Any

from Code.Allocator.FixedCombine import FixedCombo
from Code.projs.asset_allocate.dataLoad import (
    get_train_hold_rtn_data, 
    _DB,
    _MKT_DATE_TABLE
    )
from Code.BackTester.BT_AssetAllocate import (
    basicBT_rtnarr_1prd,
    BTeval_on_portfrtn
    )
from Code.projs.asset_allocate.runner import *
from Code.projs.asset_allocate.inputParser import (
    parseAssets2dicts,
    get_linear_ratios
    )
from Code.Utils.Decorator import (
    tagAttr2T,
    addAnnual,
    addSTD
    )
from Code.Utils.Type import basicPortfSolveRes



class FxdCombStrat:
    '''
    attributes: 
        __inputs
        __assets_idlst
        __portf_w_list
        __details
        __flag

    methods:
        1. backtest()  get backtest evaluation result
        2. details     get backtest details of every period
        3. assets_idlst
        4. weights     get solver returned weights of every period
        5. flag        get strategt flag
    '''


    __slots__ = ("__inputs", "__assets_idlst", "__flag",  "__portf_w_list", "__details")



    def __init__(self,
                 inputs: Any
                 ) -> None:
        
        self.__inputs = inputs
        self.__assets_idlst = []
        self.__portf_w_list = []
        self.__details = []
        self.__flag = 'fixed_comb'

    

    @addAnnual('rtn', begindate, termidate)
    @tagAttr2T('dedilated')
    @addSTD('var')
    def backtest(
            self,
            solve_fail: str = 'use-last',
            cost: Any = None
            ):
        '''
        de-dilated
            'rtn': np.floating
            'var': np.floating,
            'std': np.floating
            'trade_days': int,
            'total_cost': float,
            'gross_rtn': np.floating
            'annual_rtn': np.floating
            'drawdown': np.floating
            'rtn_series': np.ndarray
        '''

        # 这里 fixed_weights 可以是None
        train_rtn_mat_list, hold_rtn_mat_list, hold_dates_lst,\
            assets_idlst, fixed_weights = self._get_data_params()

        # 在 回测过程 中，由于涉及到复利累乘，所以需要考虑 1+de-dilated rtn
        # 所以必须在这里传入 de-dilated hold_rtn_mat. 在这之后，BT的结果不需要de-dilate
        hold_rtn_mat_list = [ hold_rtn_mat/dilate for hold_rtn_mat in hold_rtn_mat_list ]

        # 初始化
        num_assets = len(assets_idlst)
        portf_w_list = [np.repeat(1/num_assets, num_assets), ]

        portf_rtn_arr_lst, details = [], []


        for i, (train_rtn_mat, hold_rtn_mat, hold_dates) in enumerate(
            zip(train_rtn_mat_list, hold_rtn_mat_list, hold_dates_lst)
        ):
            
            solve_res = self.__solve_portf_1prd(
                train_rtn_mat,
                assets_idlst,
                fixed_weights
                )
            
            if solve_res['solve_status'] in ('direct', ):
                portf_w = solve_res['portf_w']
            elif solve_fail == 'use-last':
                portf_w = portf_w_list[-1]
            else:
                raise NotImplementedError(
                    f'default allocation method for fail-solve is not implemented'
                    )
            
            portf_rtn_arr, early_stop, _ = basicBT_rtnarr_1prd(portf_w, hold_rtn_mat, cost)
            
            detail = {
                'position_no': i+1,
                'assets_idlst': assets_idlst,
                'solve_status': solve_res['solve_status'],
                'hold_dates': hold_dates,
                'portf_w': portf_w,
                'portf_rtn': np.prod(1+portf_rtn_arr) - 1,
                'portf_rtn_series': portf_rtn_arr,
                'portf_var': np.var(portf_rtn_arr),
                'portf_std': np.std(portf_rtn_arr),
            }

            details.append( detail )
            portf_w_list.append( portf_w )
            portf_rtn_arr_lst.append( portf_rtn_arr )

            # if -1 rtn happens in this hold
            if early_stop:
                break
        
        self.__details = details
        self.__assets_idlst = assets_idlst
        self.__portf_w_list = portf_w_list[1:]
        
        # 全周期收益率，并evaluate全周期结果
        portf_rtn_arr = np.concatenate(portf_rtn_arr_lst)

        return BTeval_on_portfrtn(portf_rtn_arr)
            



    @property
    def details(self) -> list:
        return self.__details


    @property
    def assets_idlst(self) -> list:
        return self.__assets_idlst




    @property
    def flag(self) -> str:
        return self.__flag




    @property
    def weights(self) -> list:
        return self.__portf_w_list




    def _get_data_params(self) -> Any:
        '''
        return:
            train_rtn_mat_list: list of ndarray
            hold_rtn_mat_list: list of ndarray
            rebal_dates_lst: list of ndarray
            assets_idlst: list of str
            fixed_weights: ndarray or None
        '''
        assets_info_lst = self.__inputs["assets_info"] # assets_info

        assets_dict, src_tbl_dict = parseAssets2dicts(assets_info_lst)
        
        tbl_names = list( src_tbl_dict.keys() ) # list of str
        assets_ids = [ src_tbl_dict[tbl] for tbl in tbl_names] # list of lists

        train_rtn_mat_list, hold_rtn_mat_list, rebal_dates_lst , assets_idlst = \
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
        
        fixed_weights = get_linear_ratios( assets_dict, assets_idlst, 'fixed_w')

        print(
            f'Back Test for fixed-weight strategy \
            from {begindate} to {termidate} trading on every {gapday} \
            upon {len(assets_dict)} assets')
        
        return train_rtn_mat_list, hold_rtn_mat_list, rebal_dates_lst,\
               assets_idlst, fixed_weights
    


    @staticmethod
    def __solve_portf_1prd(
        train_rtn_mat: np.ndarray,
        assets_idlst: t.List[str],
        fixed_weights: t.List[t.Union[np.ndarray, None]],
        ) -> basicPortfSolveRes:
        '''
        input:
            train_rtn_mat: np.ndarray,
            assets_idlst: t.List[str],
            fixed_weights: t.List[t.Union[np.ndarray, None]],
        \n.
        return:
        de-dilate
            portf_w: np.ndarray
            solve_status: str
            assets_idlst: list
        '''
        if fixed_weights is None:
            num_assets = len(assets_idlst)
            fixed_weights = np.repeat(1/num_assets, num_assets)

        try:
            fin = FixedCombo(train_rtn_mat, fixed_weights, assets_idlst)

            res = fin()

        except Exception as e:
            traceback.print_exc()

            res = {
                'portf_w': np.array([]),
                'solve_status': 'FAIL_' + str(e),
                'assets_idlst': assets_idlst
                }
        
        return res
    


    def debug_window(self, position_no: int):
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

        train_rtn_mat_list, hold_rtn_mat_list, rebal_dates_lst, assets_idlst, fixed_weights = \
            self._get_data_params()
        
        assert position_no <= len(train_rtn_mat_list), \
            f"position_no must no larger than {len(train_rtn_mat_list)}"

        train_rtn_mat = train_rtn_mat_list[position_no-1]

        cur_res = self.__solve_portf_1prd(
            train_rtn_mat,
            assets_idlst,
            fixed_weights
            )
        
        hold_rtn_mat = hold_rtn_mat_list[position_no-1] / dilate

        return {
            "position_no": position_no,
            "train_rtn_mat": train_rtn_mat,
            "fxdcomb_target": 'fixed',
            "fixed_weights": fixed_weights,
            "solve_res": cur_res,
            "hold_rtn_mat": hold_rtn_mat
        }