import numba as nb
import numpy as np

# ============================== #
#          Math_Operators        #
# ============================== #
@nb.njit
def nb_add(arr1, arr2):
    '''
    arr1 + arr2
    '''
    return arr1 + arr2

@nb.njit
def nb_sub(arr1, arr2):
    '''
    arr1 - arr2
    '''
    return arr1 - arr2

@nb.njit
def nb_mult(arr1, arr2):
    '''
    arr1 * arr2
    '''
    return arr1 * arr2

@nb.njit
def nb_div(arr1, arr2):
    '''
    arr1 / (arr2 + 1e-12)
    这里尽量保证不会出现inf
    '''
    return arr1 / (arr2 + 1e-12)

@nb.njit
def nb_sum(arr):
    '''
    计算arr的sum, 有nan就返回nan
    这个和np.sum计算的一模一样
    '''
    sum_val = 0.0
    for val in arr:
        if not np.isnan(val):
            sum_val += val
        else:
            return np.nan
    return sum_val 

@nb.njit
def nb_nansum(arr):
    '''
    计算arr的sum, 有nan就跳过
    全为nan返回nan,
    这个跟np.nanmean计算不一样, 它返回的是0
    '''
    sum_val = 0.0
    count = 0
    for val in arr:
        if not np.isnan(val):
            sum_val += val
            count += 1
    # count==0说明全是nan，返回nan
    return sum_val if count > 0 else np.nan

@nb.njit
def nb_mean(arr):
    '''
    计算arr的mean, 有nan就返回nan
    这个和np.mean计算的一模一样
    '''
    sum_val = 0.0
    count = 0
    for val in arr:
        if not np.isnan(val):
            sum_val += val
            count += 1
        else:
            return np.nan
    return nb_div(sum_val, count) 

@nb.njit
def nb_nanmean(arr):
    '''
    计算arr的mean, 有nan就跳过
    全为nan返回nan
    这个和np.nanmean计算的一模一样
    '''
    sum_val = 0.0
    count = 0
    for val in arr:
        if not np.isnan(val):
            sum_val += val
            count += 1
    # count==0说明全是nan，返回nan
    return nb_div(sum_val, count) if count > 0 else np.nan

@nb.njit
def nb_median(arr):
    '''
    计算arr的median, 有nan就返回nan
    在numba中直接调用np.median, 就算有nan也不会返回nan
    它没有完全遵循原生numpy函数的行为
    所以我加了个判断条件
    '''
    if np.isnan(arr).any():  # 检查数组中是否有 NaN 值
        return np.nan
    return np.median(arr)

@nb.njit
def nb_nanmedian(arr):
    '''
    计算arr的median, 有nan就跳过
    全为nan返回nan
    这个和np.nanmedian计算的一模一样
    '''
    return np.nanmedian(arr)

@nb.njit
def nb_std(arr, ddof=1):
    '''
    计算arr的std, 有nan就返回nan
    '''
    n = len(arr)
    if n <= ddof:
        raise ValueError("Sample size must be larger than ddof.")
    
    # 计算均值，如果返回nan，说明有nan，直接返回nan
    mean = nb_mean(arr)
    if np.isnan(mean):
        return np.nan
    # 注意，这个mean可能是np.inf
    # 后面np.inf - np.inf = np.nan，就没有inf了
    # 计算方差
    squared_diff_sum = 0.0
    for i in range(n):
        squared_diff_sum += (arr[i] - mean) ** 2
    variance = nb_div(squared_diff_sum, (n - ddof))    
    return np.sqrt(variance)

@nb.njit
def nb_nanstd(arr, ddof=1):
    '''
    计算arr的std, 有nan就跳过
    全为nan返回nan
    '''
    n = len(arr)
    if n <= ddof:
        raise ValueError("Sample size must be larger than ddof.")
    
    # 计算均值，如果返回nan说明全是nan, 那就返回nan
    mean = nb_nanmean(arr)
    if np.isnan(mean):
        return np.nan
    # 注意，这个mean可能是np.inf
    # 后面np.inf - np.inf = np.nan，就没有inf了
    # 计算方差
    squared_diff_sum = 0.0
    count = 0
    for val in arr:
        if not np.isnan(val):
            squared_diff_sum += (val - mean) ** 2
            count += 1
    variance = nb_div(squared_diff_sum, (n - ddof))
    return np.sqrt(variance)

# ============================== #
#         Rolling_Operators      #
# ============================== #
@nb.njit
def nb_rolling_sum(arr, timeperiod):
    '''
    计算rolling的sum, min_periods=timeperiod
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = nb_sum(window)  
    return result

@nb.njit
def nb_rolling_nansum(arr, timeperiod):
    '''
    计算rolling的nansum, min_periods=timeperiod
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = nb_nansum(window)  
    return result
    
@nb.njit
def nb_rolling_mean(arr, timeperiod):
    '''
    计算rolling的mean, min_periods=timeperiod
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = nb_mean(window)  
    return result

@nb.njit
def nb_rolling_nanmean(arr, timeperiod):
    '''
    计算rolling的nanmean, min_periods=timeperiod
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = nb_nanmean(window)  
    return result

@nb.njit
def nb_rolling_std(arr, timeperiod, ddof=1):
    '''
    计算rolling的std, min_periods=timeperiod, 
    因为rolling窗口一般不大, 所以视作样本, 默认ddof=1
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = nb_std(window, ddof=ddof)  
    return result

@nb.njit
def nb_rolling_nanstd(arr, timeperiod, ddof=1):
    '''
    计算rolling的nanstd, min_periods=timeperiod, 
    因为rolling窗口一般不大, 所以视作样本, 默认ddof=1
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = nb_nanstd(window, ddof=ddof)  
    return result
    
@nb.njit
def nb_rolling_max(arr, timeperiod):
    '''
    计算rolling的max, min_periods=timeperiod, 
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = np.max(window) 
    return result

@nb.njit
def nb_rolling_argmax(arr, timeperiod):
    '''
    计算rolling的argmax, min_periods=timeperiod, 
    需要专门做nan处理
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        if np.isnan(window).any():  # 检查窗口内是否有 NaN
            result[i] = np.nan
        else:
            result[i] = np.argmax(window)  # 计算窗口内的argmax
    return result
    
@nb.njit
def nb_rolling_min(arr, timeperiod):
    '''
    计算rolling的min, min_periods=timeperiod, 
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = np.min(window)  # 计算窗口内的和
    return result
    
@nb.njit
def nb_rolling_argmin(arr, timeperiod):
    '''
    计算rolling的argmin, min_periods=timeperiod, 
    需要专门做nan处理
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        if np.isnan(window).any():  # 检查窗口内是否有 NaN
            result[i] = np.nan
        else:
            result[i] = np.argmin(window)  # 计算窗口内的argmin
    return result

# ============================== #
#          Other_Operators       #
# ============================== #
@nb.njit
def nb_shift(arr, periods=1):
    '''
    如果是整数的话，会变成float
    因为是用np.nan铺了层底
    '''
    n = len(arr)
    result = np.full(n, np.nan)
    if periods >= 0:
        result[periods:] = arr[:-periods]
    else:
        result[:periods] = arr[-periods:]
    return result

@nb.njit
def nb_fillinf(arr, fill_value):
    '''
    把所有的inf替换为其他值
    '''
    arr = arr.copy()
    for i in range(len(arr)):
        if np.isinf(arr[i]):
            arr[i] = fill_value
    return arr

@nb.njit
def nb_fillna(arr, fill_value):
    '''
    把所有的nan替换为其他值
    '''
    arr = arr.copy()
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = fill_value
    return arr