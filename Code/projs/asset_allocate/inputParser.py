from typing import Any, Callable
from operator import itemgetter
import functools
import typing as t
import numpy as np
from Code.projs.asset_allocate.dataLoad import (
    get_train_hold_rtn_data,
    _DB
)




__all__ = [
    "begindate",
    "termidate",
    "dilate",
    "back_window_size",
    "gapday",
    "benchmark",
]




class ConstraintsCheck:

    def __init__(self, func: Callable) -> None:
        self.__func = func

    def __call__(self) -> Callable:
        
        @functools.wraps(self.__func)
        def wrapper(*args, **kwargs):
            low_bounds, upper_bounds = self.__func(*args, **kwargs)

            if isinstance(low_bounds, np.array) and isinstance(upper_bounds, np.array):

                assert np.sum(low_bounds) <= 1.0, \
                    "sum of low bounds of weights must be <= 1"
                
                assert all(low_bounds <= upper_bounds), \
                    "all low bounds must be be smaller or equal to high bounds"

                # assert all(low_bounds >= 0.0), \
                #     "no short trading! low bounds must be >= 0"
                # assert all(low_bounds <= 1.0), \
                #     "no leverage trading! low bounds must be <= 1"
                # assert all(upper_bounds >= 0.0), \
                #     "no short trading! upper bounds must be >= 0"
                # assert all(upper_bounds <= 1.0), \
                #     "no leverage trading! upper bounds must <= 1"
        
            return (low_bounds, upper_bounds)

        return wrapper
    



@ConstraintsCheck
def get_constraints(
        assets_dict: dict,
        assets_idlst: list,
        def_l_b: float = -1000.0,
        def_u_b: float = 1000.0
        ) -> t.Tuple[
            t.Union[np.array, None],
            t.Union[np.array, None]
            ]:
    '''
    input:
        1. assets_dict,  {'id': {'categ':, 'l_b', 'u_b'} }
            key is asset_id, value is {'categ', 'l_b', 'u_b'}
        2. assets_idlst, [ 'id1', 'id2', 'id3',... ]
    
    return:
        low_bounds
        upper_bounds
    np.array of low/upper bounds with the order of assets_idlst
    '''
    low_bounds, upper_bounds = [], []

    for asset_id in assets_idlst:
        l_b = assets_dict[asset_id]['l_b']
        u_b = assets_dict[asset_id]['u_b']

        try:
            low_bounds.append( float(l_b) )
        except ValueError as err:
            low_bounds.append( def_l_b )

        try:
            upper_bounds.append( float(u_b) )
        except ValueError as err:
            upper_bounds.append( def_u_b )
    
    low_bounds = np.array(low_bounds)
    upper_bounds = np.array(upper_bounds)

    if np.mean(low_bounds) <= -999.0 and np.mean(upper_bounds) >= 999.0:
        # 如果下限都是 -1000.0, 且上限都是1000.0，说明没有输入任何上下限
        low_bounds, upper_bounds = None, None

    return (low_bounds, upper_bounds)




def get_tbl_asset(asset_id:str, *args, **kwargs) -> str:
    return "aidx_eod_prices"




def parseAssets2dicts(
        assets_info_lst: t.List[dict]
        ) -> t.Tuple[
            t.Dict[str, dict],
            t.Dict[str, t.List[str]]
            ]:
    '''
    Input:
        assets_info_lst, list of {'id':, 'low_b':, 'upper_b':, 'categ': ...}

    return:
        1. assets_dict, {'id': {'categ':, 'l_b', 'u_b'} }
            key is asset_id, value is {'categ', 'l_b', 'u_b'}
        2. src_tbl_dict, {'tbl_name': ['id1', 'id2'] }
            key is table name, value is [asset ids from table]
    
    duplicated asset ID will only keep the last record
    '''
    assets_dict, src_tbl_dict = {}, {}

    for asset in assets_info_lst:
        # 如果输入为空，将得到空字符串 ''
        asset_id, categ = asset.get('id'), asset.get('category')
        l_b, u_b = asset.get('lower_bound'), asset.get('upper_bound')

        assets_dict[asset_id] = [categ, l_b, u_b]

        tbl = get_tbl_asset(asset_id, categ)

        if tbl not in src_tbl_dict:
            src_tbl_dict[tbl] = []
        src_tbl_dict[tbl].append( asset_id )

        assets_dict['asset_id'] = {'categ':categ, 'l_b':l_b, 'u_b':u_b}
    
    return assets_dict, src_tbl_dict
    


def getAssetInfo_rls2glob(
        inputs: Any
        ) -> t.Tuple[
            t.Dict[str, dict],
            t.Dict[str, t.List[str]]
            ]:
    
    assets_info_lst = inputs["assets_info"] # assets_info

    # 持仓起始日，持仓终结日，膨胀系数, 调仓周期, 基准代号, 回看窗口天数
    global begindate, termidate, dilate, gapday, benchmark, back_window_size

    begindate, termidate, dilate, gapday, benchmark, back_window_size = \
        itemgetter(
            "begindate",
            "termidate",
            "rtn_dilate",
            "gapday",
            "benchmark",
            "back_window_size",
        )(inputs)
    
    print(f'release begindate {begindate}, termidate {termidate}, dilate {dilate},\
            gapday {gapday}, benchmark {benchmark}, back_window_size {back_window_size}\
            to global')
    
    # 类型转换
    gapday, back_window_size, dilate = int(gapday), int(back_window_size), int(dilate)

    assets_dict, src_tbl_dict = parseAssets2dicts(assets_info_lst)

    return assets_dict, src_tbl_dict
