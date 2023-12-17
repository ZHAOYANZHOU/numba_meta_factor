from numba_basic import *

import numpy as np
import numba as nb 
import pandas as pd
import time

def generate_arrays():
    # 生成测试用例
    global arr0, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9

    # 单个数据
    arr0 = np.random.randn(1)
    arr1 = np.full(1, np.nan)
    arr2 = np.full(1, np.inf)

    # 多个数据，有nan或者inf
    arr3 = np.random.randn(4000)

    arr4 = np.random.randn(4000)
    arr4[0:len(arr4):10] = np.nan

    arr5 = np.random.randn(4000)
    arr5[0:len(arr5):10] = np.inf

    # 多个数据，全是nan或者inf
    arr6 = np.full(4000, np.nan)
    arr7 = np.full(4000, np.inf)

    # 既有nan又有inf
    arr8 = np.random.randn(4000)
    arr8[0:len(arr8):10] = np.inf
    arr8[0:len(arr8):15] = np.nan

    # 只有nan和inf
    arr9 = np.full(4000, np.nan)
    arr9[0:len(arr9):10] = np.inf

def main():

    generate_arrays()
    
    n_iter = 10000

    for arr in [eval(f'arr{i}') for i in range(3, 10)]:

        time1 = time.time()
        series = pd.Series(arr)
        for _ in range(n_iter):
            a = series.rolling(5, 5).mean()
        print(f'{time.time() - time1:.6f}')

        time1 = time.time()
        for _ in range(n_iter):
            b = nb_rolling_mean(arr, 5)
        print(f'{time.time() - time1:.6f}','\n')

        print(a, b, nb_nansum(abs(a.values-b)), '\n')



if __name__ == '__main__':
    main()