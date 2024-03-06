from typing import Any, Callable
import typing as t
import numpy as np
import warnings
import sys
sys.dont_write_bytecode = True



warnings.filterwarnings('ignore')
app_name = __name__
static_folder = "Static"
template_folder = 'Template'
    


class ConstraintsCheck:

    def __init__(self, func: Callable) -> None:
        self.func = func


    def __call__(self, *args, **kwargs) -> Any:
        
        low_bounds, upper_bounds = self.func(*args, **kwargs)

        if isinstance(low_bounds, np.ndarray) and isinstance(upper_bounds, np.ndarray):

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
    
        return [low_bounds, upper_bounds]



@ConstraintsCheck
def get_constraints(
        assets_dict: dict,
        assets_idlst: list,
        def_l_b: float = -1000.0,
        def_u_b: float = 1000.0
        ) -> t.List[t.Union[np.ndarray, None]]:
    '''
    input:
        1. assets_dict,  {'id': {'categ':, 'l_b', 'u_b', 'risk_r'} }
            key is asset_id, value is {'categ', 'l_b', 'u_b', 'risk_r'}
        2. assets_idlst, [ 'id1', 'id2', 'id3',... ]
    
    return:
        low_bounds
        upper_bounds
    np.array of low/upper bounds with the order of assets_idlst or None \n.
    only when all input lower bounds and upper bounds are all blank, returns Double None
    '''
    low_bounds, upper_bounds = [], []

    for asset_id in assets_idlst:
        l_b = assets_dict[asset_id]['l_b']
        u_b = assets_dict[asset_id]['u_b']

        try:
            low_bounds.append( float(l_b) )
        except ValueError as err:
            low_bounds.append( def_l_b )
        except TypeError:
            raise TypeError(
                f"wrong lower bound type as {l_b}"
                )

        try:
            upper_bounds.append( float(u_b) )
        except ValueError as err:
            upper_bounds.append( def_u_b )
        except TypeError:
            raise TypeError(
                f"wrong upper bound type as {u_b}"
                )
    
    low_bounds = np.array(low_bounds)
    upper_bounds = np.array(upper_bounds)

    if np.mean(low_bounds) <= -1000.0 and np.mean(upper_bounds) >= 1000.0:
        # 如果下限都是 -1000.0, 且上限都是1000.0，说明没有输入任何上下限
        low_bounds, upper_bounds = None, None

    return [low_bounds, upper_bounds]


class LinearAllocationCheck:

    def __init__(self, func: Callable) -> None:
        self.func = func


    def __call__(self, *args, **kwargs) -> Any:
        
        ratios = self.func(*args, **kwargs)

        if isinstance(ratios, np.ndarray):

            assert np.sum(ratios) <= 1.01, \
                f"sum of ratios {np.sum(ratios)} must be <= 1"
            
            assert any( ratios >= 0.0 ), \
                'ratios must be all >= 0'
            
    
        return ratios



@LinearAllocationCheck
def get_linear_ratios(
        assets_dict: dict,
        assets_idlst: list,
        key_name: str
        ) -> t.List[t.Union[np.ndarray, None]]:
    '''
    input:
        1. assets_dict,  {'id': {'categ':, 'l_b', 'u_b', 'ratio'} }\n.
            key is asset_id, value is {'categ', 'l_b', 'u_b', 'ratio'}
        
        2. assets_idlst, [ 'id1', 'id2', 'id3',... ]
    
    return:
        ratios consists of ratio of every asset.

    np.array of ratios with the order of assets_idlst or None \n.
    only when all ratios are all blank, returns None \n.
    if only some ratios are blank, use average to fill
    '''
    ratios = []
    
    num_blank = 0
    ratio_sum = 0
    for asset_id in assets_idlst:
        val = assets_dict[asset_id][key_name]

        try:
            ratios.append( float(val) )
            ratio_sum += float(val)

        except ValueError as err:
            num_blank += 1
            ratios.append( None )
    
    if num_blank == len(ratios):
        ratios = None
    elif num_blank > 0:
        fill_val = (1-ratio_sum)/num_blank
        ratios = [i if i is not None else fill_val for i in ratios]

    
    ratios = np.array(ratios)
    
    return ratios





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

    return: \n.
        1. assets_dict, {'id': {'categ':, 'l_b', 'u_b', 'risk_r', 'fixed_w'} }
            key is asset_id, value is {'categ', 'l_b', 'u_b', 'risk_r', 'fixed_w'} \n.

        2. src_tbl_dict, {'tbl_name': ['id1', 'id2'] }
            key is table name, value is [asset ids from table]
    
    duplicated asset ID will only keep the last record
    '''
    assets_dict, src_tbl_dict = {}, {}

    for asset in assets_info_lst:
        # asset中，各个key可能存在，也可能不存在
        # 存在的key可以是空输入，得到 ''
        asset_id, categ = asset.get('id', ''), asset.get('category', '')
        l_b, u_b = asset.get('lower_bound', ''), asset.get('upper_bound', '')

        risk_r = asset.get('asset_risk_ratio', '')
        fixed_w = asset.get('fixed_wght', '')

        assets_dict[asset_id] =\
            {
            'categ':categ,
            'l_b':l_b, 'u_b':u_b,
            'risk_r': risk_r,
            'fixed_w': fixed_w
            }

        tbl = get_tbl_asset(asset_id, categ)

        if tbl not in src_tbl_dict:
            src_tbl_dict[tbl] = []
        src_tbl_dict[tbl].append( asset_id )
    
    return assets_dict, src_tbl_dict
    





































if __name__ == "__main__":
    assets_dict = {
        '00001': {
            'categ':'1', 'l_b':0, 'u_b':1
        },
        '00002': {
            'categ':'1', 'l_b':0, 'u_b':1
        },
    }
    assets_idlst = ['00001', '00002']
    print( get_constraints(assets_dict, assets_idlst ) )