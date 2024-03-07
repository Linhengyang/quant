from Code.projs.asset_allocate.dataLoad import(
    get_benchmark_rtn_data,
    _DB
    )
from Code.BackTester.BT_AssetAllocate import basicBT_multiPeriods
from Code.projs.asset_allocate.runner import *
from Code.Utils.Decorator import (
    tagFunc,
    addAnnual,
    addSTD
    )

import numpy as np
import typing as t
import sys
sys.dont_write_bytecode = True



class benchmarkStrat:
    '''
    attributes:
        1. assets_idlst
        2. flag
        3. portf_w_list
    
    methods:
        1. backtest()  get backtest result
    '''

    __slots__ = ("__assets_idlst", "__flag", "__portf_w_list")


    _BENCHMARK_WEIGHTS = {
    "CSI800":{
        "h00906.CSI":1
    },
    "CDCSI":{
        "CBA00101.CS":1
    },
    "S2D8_Mon":{
        "h00906.CSI":0.2,
        "CBA00101.CS":0.8
    },
    "S4D6_Mon":{
        "h00906.CSI":0.4,
        "CBA00101.CS":0.6
    },
    "S5D5_Mon":{
        "h00906.CSI":0.5,
        "CBA00101.CS":0.5
    },
    "S8D2_Mon":{
        "h00906.CSI":0.8,
        "CBA00101.CS":0.2
    }
    }


    def __init__(self,
                 benchmark: str
                 ) -> None:
        
        self.__flag = benchmark
        self.__assets_idlst = []
        self.__portf_w_list = []

    

    @addAnnual('rtn', begindate, termidate)
    @tagFunc('dedilated')
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
            'drawdown': np.float32
        '''

        assets_ids, tbl_names, rebal_gapday = self.parse_benchmark(self.__flag)

        hold_rtn_mat_list, self.__assets_idlst = get_benchmark_rtn_data(
            begindate,
            termidate,
            assets_ids,
            tbl_names,
            1,
            _DB,
            rebal_gapday)
        
        weights = [ self._BENCHMARK_WEIGHTS[benchmark][asset] for asset in self.__assets_idlst ]

        num_assets = hold_rtn_mat_list[0].shape[0]

        assert num_assets == len(weights),\
            "hold_rtn_mat axis 0 length must match with weights"

        num_hold_periods = len(hold_rtn_mat_list) # 持仓期数

        self.__portf_w_list = [np.array(weights), ] * num_hold_periods

        return basicBT_multiPeriods(self.__portf_w_list, hold_rtn_mat_list)
    
    
    @property
    def assets_idlst(self):
        return self.__assets_idlst


    @property
    def portf_w_list(self):
        return self.__portf_w_list


    @property
    def flag(self):
        return self.__flag
    

    @staticmethod
    def parse_benchmark(
            benchmark: str
            ) -> t.Tuple[
                t.Union[t.List[str], t.List[list]],
                t.Union[str, t.List[str]],
                t.Union[int, None]
                ]:
        if benchmark == 'CSI800':
            assets_ids = ["h00906.CSI"]
            rebal_gapday = None
            tbl_names = 'aidx_eod_prices'
        elif benchmark == "CDCSI":
            assets_ids = ["CBA00101.CS"]
            rebal_gapday = None
            tbl_names = 'cbidx_eod_prices'
        elif benchmark == "S2D8_Mon":
            assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
            rebal_gapday = 20
            tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
        elif benchmark == "S4D6_Mon":
            assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
            rebal_gapday = 20
            tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
        elif benchmark == "S5D5_Mon":
            assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
            rebal_gapday = 20
            tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
        elif benchmark == "S8D2_Mon":
            assets_ids = [["h00906.CSI"], ["CBA00101.CS"]]
            rebal_gapday = 20
            tbl_names = ['aidx_eod_prices', 'cbidx_eod_prices']
        else:
            raise ValueError(
                f"wrong code {benchmark} for benchmark"
                )
        return assets_ids, tbl_names, rebal_gapday