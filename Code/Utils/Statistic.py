import numpy as np
import typing as t



def cov2corr_mat(
        covariance: np.ndarray
        ) -> np.ndarray:
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0

    return correlation




def multiCoLinear(
        cov_mat: np.ndarray,
        id_lst: t.Union[t.List[str], None] = None
        ) -> np.ndarray:
    '''
    given covariance matrix, and corresponding id list
    return:
        id pairs which are in co-linearity
    e.g,
        corr_mat = [[1.0, 0.8, 1.0, 0.6]
                    [0.8, 1.0, 0.2, 1.0]
                    [1.0, 0.2, 1.0, 0.5]
                    [0.6, 1.0, 0.5, 1.0]]

    if not input id_lst:
        return: [0, 2], [1, 3] as 2-d nparray
    if input id_lst as ['a', 'b', 'c', 'd']:
        return: ['a', 'c'], ['b', 'd'] as 2-d nparray
    '''
    corr_mat = cov2corr_mat(cov_mat)
    corr_triu_mat = np.abs( np.triu(corr_mat, k=1) )
    cmp_mat = np.isclose( np.abs( corr_triu_mat ), 1.)

    colinear_ords_pair = np.array( np.where(cmp_mat) ).T

    if not id_lst:
        return colinear_ords_pair
    else:
        return np.array(id_lst)[colinear_ords_pair]