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
    # strided_slices: 2-d array with shape (num_slices = num_jump + 1, window_size)
    # strided_slices[-1][0] 开始到 strided_slices[-1][-1], 长度是window_size, 但是 strided_slices[-1][-1]到末尾，不足window_size

    residual_range = range(strided_slices[-1][-1]+1, length) # 从strided_slices[-1][-1] + 1 到末尾, 可为空

    last_short_range = range(strided_slices[-1][0] + stride, length) # 从strided_slices[-1][0] + stride 到末尾, 可为空
    # 当 window_size = stride时，residual_slice == last_short_slice
    # last_incomlete_slice = range( stride * ((length - window_size) // stride + 1), length ) \
    #     if stride * ((length - window_size) // stride + 1) < length else None
    return strided_slices, residual_range, last_short_range


# given length = 10, windows_size =3, stride = 3
# return strided_indices = [0, 3, 6], rsd_indx = 9
# given length = 10, window_size = 2, stride = 2
# return strided_indices = [ 0, 2, 4, 6, 8 ], rsd_indx = None
def strided_indexing_w_residual(length, window_size, stride):
    strided_indices = list( range(0, length-window_size, stride) )
    next_jump_indx = strided_indices[-1] + stride
    next_jump_indx = next_jump_indx if next_jump_indx < length else None

    residual_index = strided_indices[-1] + window_size + 1
    residual_index = residual_index if residual_index < length else None
    return strided_indices, next_jump_indx, residual_index











if __name__ == "__main__":
    # print( strided_slicing_w_residual(16, 5, 3) )
    x = range(5, 5)
    # a = np.array( [0, 1 ,2 ,3 ,4 ,5 ,6, 7, 8, 9, 10] )
    if list(x):
        print('true')
    else:
        print('false')