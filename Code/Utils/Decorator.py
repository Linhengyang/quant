from typing import Any, Callable
import typing as t
from datetime import datetime
from functools import wraps
import numpy as np


class funcTagger:
    '''
    Tag a function with attribute {attr_name} as True
    '''
    def __init__(self,
                 attr_name: str
                 ) -> None:
        self.__need_attr = attr_name

    def __call__(self,
                 fc: Callable
                 ) -> Callable:
        
        @wraps(fc)
        def wrapper(*args, **kwargs):
            return fc(*args, **kwargs)
        
        setattr(wrapper, self.__need_attr, True)

        return wrapper


class deDilate:
    '''
    decorator for solving result
    Decorating function must return a dict.
    This decorator modify all values with key ending with "rtn|std|var" by dilate
    '''
    def __init__(self,
                 rtn_dilate: int
                 ) -> None:
        
        self.__rtn_dilate = rtn_dilate


    def __call__(self,
                 fc: Callable
                 ) -> Callable:
        
        @wraps(fc)
        def wrapper(*args, **kwargs):
            bt_res = fc(*args, **kwargs)
            keys = bt_res.keys()
            for key in keys:
                if key.endswith('rtn'): # 所有rtn结尾的，都要 /dilate
                    bt_res[key] /= self.__rtn_dilate
                elif key.endswith('std'): # 所有std结尾的，都要 / dilate
                    bt_res[key] /= self.__rtn_dilate
                elif key.endswith('var'): # 所有var结尾的，都要 / dilate^2
                    bt_res[key] /= np.power(self.__rtn_dilate, 2)
            return bt_res
        
        setattr(wrapper, 'dedilated', True) # add dedilated flag

        return wrapper




class addAnnual:
    '''
    decorator for for de-dilated BackTest result
    Decorating function must return a dict.
    This decorator add 'annual_{rtn_argname}' based on {rtn_argname}
    '''
    def __init__(self,
                 rtn_argname: str,
                 begindate: str,
                 termidate: str
                 ) -> None:

        self.__rtn_argname = rtn_argname
        self.__delta_year = ( datetime.strptime(termidate, '%Y%m%d') - \
                              datetime.strptime(begindate, '%Y%m%d')
                            ).days / 365


    def __call__(self,
                 backtest_fc: Callable
                 ) -> Callable:
        
        @wraps(backtest_fc)
        def wrapper(*args, **kwargs):
            bt_res = backtest_fc(*args, **kwargs)

            # 检查 return 是否已经 de-dilate. 年化利率必须使用 de-dialted return 计算
            assert getattr(backtest_fc, 'dedilated', False),\
                "must de-dilated before adding annual return"

            _rtn = bt_res[self.__rtn_argname]
            # 添加 annual_{rtn_argname}
            bt_res['annual_' + self.__rtn_argname] =\
                np.power( 1 + _rtn, 1/self.__delta_year ) - 1
            
            return bt_res
        
        return wrapper
    


class addSTD:
    '''
    decorator for BackTest/solving result
    Decorating function must return a dict.
    This decorator add '{suffix}_std' based on '{suffix}_var'
    '''
    def __init__(self,
                 var_argname: str,
                 ) -> None:

        self.__var_argname = var_argname


    def __call__(self,
                 backtest_fc: Callable
                 ) -> Callable:
        
        @wraps(backtest_fc)
        def wrapper(*args, **kwargs):
            bt_res = backtest_fc(*args, **kwargs)

            # 添加 suffix_std 项
            std_argname = self.__var_argname.replace('var', 'std')
            try:
                bt_res[std_argname] = np.sqrt( bt_res[self.__var_argname] )
            except:
                bt_res[std_argname] = np.float32(-1)
            
            return bt_res
        
        return wrapper
    


    
    

class addSharpe:
    '''
    decorator for de-dilated BackTest/solving result
    Decorating function must return a dict.
    This decorator add 'sharpe' based on '{rtn_argname}' '{var_argname}' 
    and risk free rate.
    '''
    def __init__(self,
                 r_f: np.float32,
                 rtn_argname: str,
                 var_argname: str,
                 ) -> None:
        self.__rf = r_f
        self.__rtn_argname = rtn_argname
        self.__var_argname = var_argname


    def __call__(self,
                 backtest_fc: Callable
                 ) -> Callable:
        
        @wraps(backtest_fc)
        def wrapper(*args, **kwargs):
            bt_res = backtest_fc(*args, **kwargs)

            # 检查 return 是否已经 de-dilate. 年化利率必须使用 de-dialted return 计算
            assert getattr(backtest_fc, 'dedilated', False),\
                "must de-dilated before adding annual return"
            
            _rtn = bt_res[self.__rtn_argname]
            _var = bt_res[self.__var_argname]
            # 添加 sharpe 项
            bt_res['sharpe'] = ( _rtn - self.__rf )/np.sqrt( _var )

            return bt_res
        
        return wrapper