import numpy as np

# 1-D array slices to 2-D array with window length L and stride S  ---> 2-D array with shape(nrows, L)
# nrows=max step number through 1-D array with stride S
# e.g
# [1,2,3,4,5,6,7,8,9,10] --> window size 3, stride 2 ---> [1,2,3], [3,4,5], [5,6,7], [7,8,9]
# residuals ----> [10]

def strided_slice_1darr(arr, window_size, stride):  # Window len = window_size, Stride len/stepsize = stride
    assert len(arr) >= window_size, '1-D array length must be larger or equal to window_size'
    nrows = ((arr.size-window_size)//stride)+1
    n = arr.strides[0]
    return np.lib.stride_tricks.as_strided(arr, shape=(nrows,window_size), strides=(stride*n,n))


# given length = 10, window_size = 3, stride = 3
# return [0,1,2], [3,4,5], [6,7,8], [9,]
def strided_slicing_w_residual(length, window_size, stride):
    strided_slices = strided_slice_1darr(np.arange(length), window_size, stride)
    rsd_slices = range( stride * ((length - window_size) // stride + 1), length ) \
        if stride * ((length - window_size) // stride + 1) < length else None
    return strided_slices, rsd_slices


# given length = 10, windows_size =3, stride = 3
# return [0, 3, 6, 9]
def strided_indexing_w_residual(length, window_size, stride):
    indices = list( range(0, length-window_size, stride) )
    next_indx = indices[-1] + stride
    if next_indx < length:
        indices.append( next_indx )
    return indices











if __name__ == "__main__":
    print( strided_indexing_w_residual(10, 2, 5) )