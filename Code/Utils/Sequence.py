import numpy as np

# 1-D array slices to 2-D array with window length L and stride S  ---> 2-D array with shape(nrows, L)
# nrows=max step number through 1-D array with stride S
# e.g
# [1,2,3,4,5,6,7,8,9,10] --> window size 3, stride 2 ---> [1,2,3], [3,4,5], [5,6,7], [7,8,9]
# residuals ----> [9, 10]

def strided_slice_1darr(arr, window_size, stride):  # Window len = window_size, Stride len/stepsize = stride
    assert len(arr) >= window_size, '1-D array length must be larger or equal to window_size'
    nrows = ((arr.size-window_size)//stride)+1
    n = arr.strides[0]
    return np.lib.stride_tricks.as_strided(arr, shape=(nrows,window_size), strides=(stride*n,n))


# given length = 10, window_size = 3, stride = 3
# return strided_slices = [ [0,1,2], [3,4,5], [6,7,8] ], rsd_slices = [9,]
# given length = 10, window_size = 2, stride = 2
# return strided_slices = [ [0,1], [2, 3], [4,5], [6,7], [8, 9] ], rsd_slices = None
def strided_slicing_w_residual(length, window_size, stride):
    strided_slices = strided_slice_1darr(np.arange(length), window_size, stride)
    rsd_slice = range( stride * ((length - window_size) // stride + 1), length ) \
        if stride * ((length - window_size) // stride + 1) < length else None
    return strided_slices, rsd_slice


# given length = 10, windows_size =3, stride = 3
# return strided_indices = [0, 3, 6], rsd_indx = 9
# given length = 10, window_size = 2, stride = 2
# return strided_indices = [ 0, 2, 4, 6, 8 ], rsd_indx = None
def strided_indexing_w_residual(length, window_size, stride):
    strided_indices = list( range(0, length-window_size, stride) )
    next_indx = strided_indices[-1] + stride
    rsd_indx = next_indx if next_indx < length else None
    return strided_indices, rsd_indx











if __name__ == "__main__":
    print( strided_slicing_w_residual(16, 5, 5) )