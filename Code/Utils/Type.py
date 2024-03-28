from typing import TypedDict
import numpy as np
import typing as t






class basicPortfSolveRes(TypedDict):
    portf_w: np.ndarray
    solve_status: str
    assets_idlst: list





class basicBackTestRes(TypedDict):
    rtn: np.floating
    var: np.floating
    trade_days: int
    total_cost: np.floating
    gross_rtn: np.floating
    drawdown: np.floating




class LowFrequencyStrategy:
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
                 inputs: t.Any
                 ) -> None:
        
        self.__inputs = inputs
        self.__assets_idlst = []
        self.__portf_w_list = []
        self.__details = []
        self.__flag = ''

    def backtest(self, *args, **kwargs):
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
        '''
        pass


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
    

    def _get_data_params(self) -> t.Any:
        '''
        return:
            train_rtn_mat_list: list of ndarray
            hold_rtn_mat_list: list of ndarray
            rebal_dates_lst: list of ndarray
            assets_idlst: list of str
            constraints: list of ndarray or none
            mvo_target: str
            expt_tgt_value: npfloat
        '''
        pass


    @staticmethod
    def __solve_portf_1prd(*args, **kwargs):
        '''
        input:
            train_rtn_mat: np.ndarray,
            assets_idlst: t.List[str],
            constraints: t.List[t.Union[np.ndarray, None]],
            mvo_target: str,
            expt_tgt_value: np.floating,
        return:
            portf_w: np.ndarray
            solve_status: str
            assets_idlst: list
        '''
        pass


    def detail_window(self, *args, **kwargs):
        '''
        return every details about one of single position window

        "position_no": int, starts from 1
        "train_rtn_mat": np.ndarray
        "constraints": list of None or np.ndarray
        "mvo_target": str
        "expt_tgt_value": np.floating
        "solve_res": basicPortfSolveRes
        "hold_rtn_mat": np.ndarray
        '''
        pass