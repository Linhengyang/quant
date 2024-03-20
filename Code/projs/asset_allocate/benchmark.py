from Code.projs.asset_allocate.dataLoad import(
    get_benchmark_rtn_data,
    _DB
    )
from Code.Allocator.FixedCombine import FixedCombo
from Code.BackTester.BT_AssetAllocate import (
    BTeval_on_portfrtn,
    day_portfw_on_period
)
from Code.Utils.Sequence import strided_slicing_w_residual
from Code.projs.asset_allocate.runner import *
from Code.Utils.Decorator import (
    tagAttr2T,
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
        __assets_idlst
        __flag
        __portf_w_list
        __details
    
    methods:
        1. backtest()  get backtest evaluation result
        2. details     get backtest details of every period
        3. assets_idlst
        4. weights     get solver returned weights of every period
        5. flag        get strategt flag
    '''

    __slots__ = ("__assets_idlst", "__flag", "__portf_w_list", "__details")


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
        self.__details = []

    

    @addAnnual('rtn', begindate, termidate)
    @tagAttr2T('dedilated')
    @addSTD('var')
    def backtest(
        self,
        cost: t.Any = None
        ) -> dict:
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

        bchmk_rtn_mat_list, self.__assets_idlst = get_benchmark_rtn_data(
            begindate,
            termidate,
            assets_ids,
            tbl_names,
            1,
            _DB,
            rebal_gapday)
        
        weights = [ self._BENCHMARK_WEIGHTS[benchmark][asset] for asset in self.__assets_idlst ]

        num_assets = bchmk_rtn_mat_list[0].shape[0]
        assert num_assets == len(weights),\
            "hold_rtn_mat axis 0 length must match with weights"

        # 先求出 benchmark 在全周期的日收益率序列，然后按照 策略的 slice切开, 才能对齐period
        # 初始化
        bchmk_rtn_arr_lst, bchmk_w  = [], np.array(weights)

        num_hold_periods = len(bchmk_rtn_mat_list) # 持仓期数
        self.__portf_w_list = [bchmk_w, ] * num_hold_periods

        for bchmk_rtn_mat in bchmk_rtn_mat_list:

            # benchmark 不作 early stop, 即不作原始本金假设，发生 -1 的rtn也无所谓
            w_mat = day_portfw_on_period(bchmk_w, bchmk_rtn_mat)

            bchmk_rtn_arr = (w_mat * bchmk_rtn_mat).sum(axis=0)
            
            bchmk_rtn_arr_lst.append( bchmk_rtn_arr )


        # 全周期收益率
        bchmk_rtn_arr = np.concatenate(bchmk_rtn_arr_lst)

        # 切分 portf_rtn_arr
        # 每一期持仓起始，往后持仓gapday天或最后一天
        hold_strided_slices, _, last_range = strided_slicing_w_residual(
            len(bchmk_rtn_arr),
            gapday,
            gapday
            )

        hold_bchmk_rtn_arr_lst = list( bchmk_rtn_arr[hold_strided_slices] )

        if list(last_range):
            hold_bchmk_rtn_arr_lst.append( bchmk_rtn_arr[last_range] )
        
        details = []
        for i, hold_bchmk_rtn_arr in enumerate(hold_bchmk_rtn_arr_lst):
            detail = {
                'position_no': i+1,
                'bchmk_rtn': np.prod(1+hold_bchmk_rtn_arr) - 1,
                'bchmk_var': np.var(hold_bchmk_rtn_arr),
                'bchmk_std': np.std(hold_bchmk_rtn_arr),
                'bchmk_rtn_series': hold_bchmk_rtn_arr
            }

            details.append( detail )

        self.__details = details

        return BTeval_on_portfrtn(bchmk_rtn_arr)
    

    @property
    def details(self) -> list:
        return self.__details


    @property
    def assets_idlst(self) -> list:
        return self.__assets_idlst


    @property
    def weights(self) -> list:
        return self.__portf_w_list


    @property
    def flag(self) -> str:
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