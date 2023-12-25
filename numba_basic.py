import numba as nb
import numpy as np

# ============================== #
#           基础数学运算          #
# ============================== #
@nb.njit
def ADD(a, b):
    '''
    a + b
    '''
    return a + b

@nb.njit
def SUB(a, b):
    '''
    a - b
    '''
    return a - b

@nb.njit
def MUL(a, b):
    '''
    a * b
    '''
    return a * b

@nb.njit
def DIV(a, b):
    '''
    a / b
    '''
    return a / b

@nb.njit
def LOG(arr):
    '''
    np.log(arr)
    arr大于0, 等于0是-inf, 小于0是nan
    '''
    return np.log(arr)

@nb.njit
def EXP(arr):
    '''
    np.exp(arr)
    '''
    return np.exp(arr)

@nb.njit
def SQUARE(arr):
    '''
    arr ** 2
    '''
    return np.square(arr)

@nb.njit
def SQRT(arr):
    '''
    arr ** 0.5, arr大于0
    '''
    return np.sqrt(arr)

@nb.njit
def POWER(arr, n):
    '''
    arr ** n
    '''
    return np.power(arr, n)

@nb.njit
def ROOT(arr, n):
    '''
    arr ** (1/n), arr大于0
    '''
    return np.power(arr, 1 / n)

@nb.njit
def ABS(arr):
    '''
    np.abs(arr)
    '''
    return np.abs(arr)

@nb.njit
def SIN(arr):
    '''
    np.sin(arr)
    '''
    return np.sin(arr)

@nb.njit
def COS(arr):
    '''
    np.cos(arr)
    '''
    return np.cos(arr)

@nb.njit
def TAN(arr):
    '''
    np.tan(arr)
    '''
    return np.tan(arr)

@nb.njit
def ARCSIN(arr):
    '''
    np.arcsin(arr), -1到1
    '''
    return np.arcsin(arr)

@nb.njit
def ARCCOS(arr):
    '''
    np.arccos(arr), -1到1
    '''
    return np.arccos(arr)

@nb.njit
def ARCTAN(arr):
    '''
    np.arctan(arr)
    '''
    return np.arctan(arr)

@nb.njit
def SIGN(arr):
    '''
    np.sign(arr)
    正数1, 负数-1, 0就是0
    '''
    return np.sign(arr)

@nb.njit
def ROUND(arr, n):
    '''
    四舍五入, 保留小数点后n位
    '''
    return np.round(arr, decimals=n)

# ============================== #
#             统计函数            #
# ============================== #
@nb.njit
def SUM(arr, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 效果同np.nansum()
    ignore_nan=False, 效果同np.sum(), 但是如果全为nan的话, 返回nan而不是0
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    if ignore_nan:
        sum_val = 0.0
        count = 0
        for val in arr:
            if not np.isnan(val):
                sum_val += val
                count += 1
        # count==0说明全是nan，返回nan
        return sum_val if count > 0 else np.nan
    else:
        sum_val = 0.0
        for val in arr:
            if not np.isnan(val):
                sum_val += val
            else:
                return np.nan
        return sum_val 

@nb.njit
def MEAN(arr, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 效果同np.nanmean()
    ignore_nan=False, 效果同np.mean()
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    if ignore_nan:
        sum_val = 0.0
        count = 0
        for val in arr:
            if not np.isnan(val):
                sum_val += val
                count += 1
        # count==0说明全是nan，返回nan
        return DIV(sum_val, count) if count > 0 else np.nan
    else:
        sum_val = 0.0
        count = 0
        for val in arr:
            if not np.isnan(val):
                sum_val += val
                count += 1
            else:
                return np.nan
        return DIV(sum_val, count) 

@nb.njit
def MEDIAN(arr, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，效果同 np.nanmedian()
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    
    if ignore_nan:
        # 过滤出非 nan 的值进行计算
        valid_arr = arr[~np.isnan(arr)]
        if valid_arr.size == 0:
            return np.nan
        else:
            # 由于 Numba 不能直接计算中位数，此处退出 njit 模式
            return np.median(valid_arr)
    else:
        if np.isnan(arr).any():
            # 如果存在 nan，则返回 nan
            return np.nan
        else:
            # 由于 Numba 不能直接计算中位数，此处退出 njit 模式
            return np.median(arr)

@nb.njit
def MAX(arr, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，效果同 np.nanmax()
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    if ignore_nan:
        valid_arr = arr[~np.isnan(arr)]
        return np.max(valid_arr) if valid_arr.size > 0 else np.nan
    else:
        if np.isnan(arr).any():
            return np.nan
        else:
            return np.max(arr)
        
@nb.njit
def MIN(arr, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，效果同 np.nanmin()
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    if ignore_nan:
        valid_arr = arr[~np.isnan(arr)]
        return np.min(valid_arr) if valid_arr.size > 0 else np.nan
    else:
        if np.isnan(arr).any():
            return np.nan
        else:
            return np.min(arr)

@nb.njit
def RANGE(arr, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，返回有效数据的范围
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    return SUB(MAX(arr, ignore_nan), MIN(arr, ignore_nan))

@nb.njit
def QUANTILE(arr, q, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，计算 n 分位数
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    q: 分位数，范围 0-1 之间
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    if ignore_nan:
        valid_arr = arr[np.isfinite(arr)]
        return np.percentile(valid_arr, q * 100) if valid_arr.size > 0 else np.nan
    else:
        if np.isinf(arr).any():
            return np.nan
        return np.percentile(arr, q * 100)

@nb.njit
def VAR(arr, ddof=0, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，计算方差
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    ddof: 自由度调整
    '''
    arr = np.where(np.isinf(arr), np.nan, arr)
    if ignore_nan:
        n = len(arr)
        if n <= ddof:
            return np.nan
            # raise ValueError("Sample size must be larger than ddof.")
        # 计算均值，如果返回nan说明全是nan, 那就返回nan
        mean = MEAN(arr, ignore_nan)
        if np.isnan(mean):
            return np.nan
        # 计算方差
        squared_diff_sum = 0.0
        count = 0
        for val in arr:
            if not np.isnan(val):
                squared_diff_sum += SQUARE(val - mean)
                count += 1
        if count <= ddof:
            return np.nan
            # aise ValueError("Valid sample size must be larger than ddof.")
        variance = DIV(squared_diff_sum, (count - ddof))
        return variance

    else:
        n = len(arr)
        if n <= ddof:
            return np.nan
            # raise ValueError("Sample size must be larger than ddof.")
        
        # 计算均值，如果返回nan，说明有nan，直接返回nan
        mean = MEAN(arr, ignore_nan)
        if np.isnan(mean):
            return np.nan
        # 计算方差
        squared_diff_sum = 0.0
        for i in range(n):
            squared_diff_sum += SQUARE(arr[i] - mean)
        variance = DIV(squared_diff_sum, (n - ddof))    
        return variance

@nb.njit
def STD(arr, ddof=0, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，计算标准差
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    ddof: 自由度调整 
    '''
    return SQRT(VAR(arr=arr, ddof=ddof, ignore_nan=ignore_nan))

@nb.njit
def COV(x, y, ddof=0, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，计算协方差
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    ddof: 自由度调整
    '''
    x = np.where(np.isinf(x), np.nan, x)
    y = np.where(np.isinf(y), np.nan, y)
    # mask表示x和y都是 非nan 的值
    mask = np.where(np.isnan(x), False, True) & np.where(np.isnan(y), False, True)
    valid_num = np.sum(mask)
    if ignore_nan:
        # 有效值长度不大于ddof
        if valid_num <= ddof:
            return np.nan
        else:
            part1 = MEAN(MUL(x[mask], y[mask]), ignore_nan) 
            part2 = MUL(MEAN(x[mask], ignore_nan), MEAN(y[mask], ignore_nan))
            return SUB(part1, part2) * valid_num / (valid_num-ddof)
    else:
        n = len(x)
        # 有nan值 或者 有效值长度不大于ddof
        if (valid_num < n) or (valid_num <= ddof):
            return np.nan
        else:
            part1 = MEAN(MUL(x[mask], y[mask]), ignore_nan)
            part2 = MUL(MEAN(x[mask], ignore_nan), MEAN(y[mask], ignore_nan))
            return SUB(part1, part2) * valid_num / (valid_num-ddof)

@nb.njit
def CORR(x, y, ddof=0, ignore_nan=True):
    '''
    inf 视作 nan
    ignore_nan=True, 忽略 nan 和 inf，计算相关系数
    ignore_nan=False, 一旦有 nan 或 inf，结果为 nan
    ddof: 自由度调整
    其实算corr的时候, ddof没啥用, 默认0
    '''
    x = np.where(np.isinf(x), np.nan, x)
    y = np.where(np.isinf(y), np.nan, y)
    # mask表示x和y都是非nan的值
    mask = np.where(np.isnan(x), False, True) & np.where(np.isnan(y), False, True)
    valid_num = np.sum(mask)

    if ignore_nan:
        if valid_num <= ddof:
            return np.nan
        else:
            std_x = STD(x[mask], ddof, ignore_nan)
            std_y = STD(y[mask], ddof, ignore_nan)
            if std_x > 0 and std_y > 0:
                return COV(x[mask], y[mask], ddof, ignore_nan) / (std_x * std_y)
            else:
                return np.nan
    else:
        n = len(x)
        if (valid_num < n) or (valid_num <= ddof):
            return np.nan
        else:
            std_x = STD(x[mask], ddof, ignore_nan)
            std_y = STD(y[mask], ddof, ignore_nan)
            if std_x > 0 and std_y > 0:
                return COV(x[mask], y[mask], ddof, ignore_nan) / (std_x * std_y)
            else:
                return np.nan

# ============================== #
#           时间序列函数          #
# ============================== #
@nb.njit
def SHIFT(arr, periods=1):
    '''
    正数为往下移动，负数为往上移动
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
def ROLL_SUM(arr, timeperiod, ignore_nan=True):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = SUM(window, ignore_nan)  
    return result

@nb.njit
def ROLL_MEAN(arr, timeperiod, ignore_nan=True):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = MEAN(window, ignore_nan)  
    return result

@nb.njit
def ROLL_MEDIAN(arr, timeperiod, ignore_nan=True):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = MEDIAN(window, ignore_nan)  
    return result

@nb.njit
def ROLL_MAX(arr, timeperiod, ignore_nan=True):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = MAX(window, ignore_nan)  
    return result

@nb.njit
def ROLL_MIN(arr, timeperiod, ignore_nan=True):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = MIN(window, ignore_nan)  
    return result

@nb.njit
def ROLL_VAR(arr, timeperiod, ddof=0, ignore_nan=True):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = VAR(window, ddof, ignore_nan)  
    return result

@nb.njit
def ROLL_STD(arr, timeperiod, ddof=0, ignore_nan=True):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window = arr[i - timeperiod + 1: i + 1]
        result[i] = STD(window, ddof, ignore_nan)  
    return result

@nb.njit
def ROLL_COV(x, y, timeperiod, ddof=0, ignore_nan=True):
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window_x = x[i - timeperiod + 1: i + 1]
        window_y = y[i - timeperiod + 1: i + 1]
        result[i] = COV(window_x, window_y, ddof, ignore_nan)  
    return result

@nb.njit
def ROLL_CORR(x, y, timeperiod, ddof=0, ignore_nan=True):
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(timeperiod - 1, n):
        window_x = x[i - timeperiod + 1: i + 1]
        window_y = y[i - timeperiod + 1: i + 1]
        result[i] = CORR(window_x, window_y, ddof, ignore_nan)  
    return result


if __name__ == '__main__':
    # import time

    
    import pandas as pd

    arr1 = np.array([1,2,3,4,5])
    arr2 = np.array([np.nan, 1,2,np.nan,3])
    arr3 = np.array([1, np.nan])
    arr4 = np.array([1,2,np.inf,4,5])

    # for i in range(1,5): 
    #     arr = eval(f'arr{i}')
    #     print(ROLL_STD(arr, 2, ddof=1, ignore_nan=True))
    #     print(pd.Series(arr).rolling(2,2).apply(lambda x: np.nanstd(x, ddof=1)), '\n')
    #     # print(np.min(arr), '\n')

    arr = np.random.normal(0, 1, (2, 20))
    # arr[1,3] = np.nan
    print(ROLL_COV(arr[0,:], arr[1,:], 3, ddof=0, ignore_nan=False))
    print(pd.DataFrame(arr).T)
    print(pd.DataFrame(arr).T.rolling(3,3).apply(lambda x: x.().iloc[0, 1]))


# @nb.njit
# def nb_sum(arr):
#     '''
#     计算arr的sum, 有nan就返回nan
#     这个和np.sum计算的一模一样
#     '''
#     sum_val = 0.0
#     for val in arr:
#         if not np.isnan(val):
#             sum_val += val
#         else:
#             return np.nan
#     return sum_val 

# @nb.njit
# def nb_nansum(arr):
#     '''
#     计算arr的sum, 有nan就跳过
#     全为nan返回nan,
#     这个跟np.nanmean计算不一样, 它返回的是0
#     '''
#     sum_val = 0.0
#     count = 0
#     for val in arr:
#         if not np.isnan(val):
#             sum_val += val
#             count += 1
#     # count==0说明全是nan，返回nan
#     return sum_val if count > 0 else np.nan

# @nb.njit
# def nb_mean(arr):
#     '''
#     计算arr的mean, 有nan就返回nan
#     这个和np.mean计算的一模一样
#     '''
#     sum_val = 0.0
#     count = 0
#     for val in arr:
#         if not np.isnan(val):
#             sum_val += val
#             count += 1
#         else:
#             return np.nan
#     return nb_div(sum_val, count) 

# @nb.njit
# def nb_nanmean(arr):
#     '''
#     计算arr的mean, 有nan就跳过
#     全为nan返回nan
#     这个和np.nanmean计算的一模一样
#     '''
#     sum_val = 0.0
#     count = 0
#     for val in arr:
#         if not np.isnan(val):
#             sum_val += val
#             count += 1
#     # count==0说明全是nan，返回nan
#     return nb_div(sum_val, count) if count > 0 else np.nan

# @nb.njit
# def nb_median(arr):
#     '''
#     计算arr的median, 有nan就返回nan
#     在numba中直接调用np.median, 就算有nan也不会返回nan
#     它没有完全遵循原生numpy函数的行为
#     所以我加了个判断条件
#     '''
#     if np.isnan(arr).any():  # 检查数组中是否有 NaN 值
#         return np.nan
#     return np.median(arr)

# @nb.njit
# def nb_nanmedian(arr):
#     '''
#     计算arr的median, 有nan就跳过
#     全为nan返回nan
#     这个和np.nanmedian计算的一模一样
#     '''
#     return np.nanmedian(arr)

# @nb.njit
# def nb_std(arr, ddof=1):
#     '''
#     计算arr的std, 有nan就返回nan
#     '''
#     n = len(arr)
#     if n <= ddof:
#         raise ValueError("Sample size must be larger than ddof.")
    
#     # 计算均值，如果返回nan，说明有nan，直接返回nan
#     mean = nb_mean(arr)
#     if np.isnan(mean):
#         return np.nan
#     # 注意，这个mean可能是np.inf
#     # 后面np.inf - np.inf = np.nan，就没有inf了
#     # 计算方差
#     squared_diff_sum = 0.0
#     for i in range(n):
#         squared_diff_sum += (arr[i] - mean) ** 2
#     variance = nb_div(squared_diff_sum, (n - ddof))    
#     return np.sqrt(variance)

# @nb.njit
# def nb_nanstd(arr, ddof=1):
#     '''
#     计算arr的std, 有nan就跳过
#     全为nan返回nan
#     '''
#     n = len(arr)
#     if n <= ddof:
#         raise ValueError("Sample size must be larger than ddof.")
    
#     # 计算均值，如果返回nan说明全是nan, 那就返回nan
#     mean = nb_nanmean(arr)
#     if np.isnan(mean):
#         return np.nan
#     # 注意，这个mean可能是np.inf
#     # 后面np.inf - np.inf = np.nan，就没有inf了
#     # 计算方差
#     squared_diff_sum = 0.0
#     count = 0
#     for val in arr:
#         if not np.isnan(val):
#             squared_diff_sum += (val - mean) ** 2
#             count += 1
#     variance = nb_div(squared_diff_sum, (n - ddof))
#     return np.sqrt(variance)

# # ============================== #
# #         Rolling_Operators      #
# # ============================== #
# @nb.njit
# def nb_rolling_sum(arr, timeperiod):
#     '''
#     计算rolling的sum, min_periods=timeperiod
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = nb_sum(window)  
#     return result

# @nb.njit
# def nb_rolling_nansum(arr, timeperiod):
#     '''
#     计算rolling的nansum, min_periods=timeperiod
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = nb_nansum(window)  
#     return result
    
# @nb.njit
# def nb_rolling_mean(arr, timeperiod):
#     '''
#     计算rolling的mean, min_periods=timeperiod
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = nb_mean(window)  
#     return result

# @nb.njit
# def nb_rolling_nanmean(arr, timeperiod):
#     '''
#     计算rolling的nanmean, min_periods=timeperiod
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = nb_nanmean(window)  
#     return result

# @nb.njit
# def nb_rolling_std(arr, timeperiod, ddof=1):
#     '''
#     计算rolling的std, min_periods=timeperiod, 
#     因为rolling窗口一般不大, 所以视作样本, 默认ddof=1
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = nb_std(window, ddof=ddof)  
#     return result

# @nb.njit
# def nb_rolling_nanstd(arr, timeperiod, ddof=1):
#     '''
#     计算rolling的nanstd, min_periods=timeperiod, 
#     因为rolling窗口一般不大, 所以视作样本, 默认ddof=1
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = nb_nanstd(window, ddof=ddof)  
#     return result
    
# @nb.njit
# def nb_rolling_max(arr, timeperiod):
#     '''
#     计算rolling的max, min_periods=timeperiod, 
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = np.max(window) 
#     return result

# @nb.njit
# def nb_rolling_argmax(arr, timeperiod):
#     '''
#     计算rolling的argmax, min_periods=timeperiod, 
#     需要专门做nan处理
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         if np.isnan(window).any():  # 检查窗口内是否有 NaN
#             result[i] = np.nan
#         else:
#             result[i] = np.argmax(window)  # 计算窗口内的argmax
#     return result
    
# @nb.njit
# def nb_rolling_min(arr, timeperiod):
#     '''
#     计算rolling的min, min_periods=timeperiod, 
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         result[i] = np.min(window)  # 计算窗口内的和
#     return result
    
# @nb.njit
# def nb_rolling_argmin(arr, timeperiod):
#     '''
#     计算rolling的argmin, min_periods=timeperiod, 
#     需要专门做nan处理
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     for i in range(timeperiod - 1, n):
#         window = arr[i - timeperiod + 1: i + 1]
#         if np.isnan(window).any():  # 检查窗口内是否有 NaN
#             result[i] = np.nan
#         else:
#             result[i] = np.argmin(window)  # 计算窗口内的argmin
#     return result

# # ============================== #
# #          Other_Operators       #
# # ============================== #
# @nb.njit
# def nb_shift(arr, periods=1):
#     '''
#     如果是整数的话，会变成float
#     因为是用np.nan铺了层底
#     '''
#     n = len(arr)
#     result = np.full(n, np.nan)
#     if periods >= 0:
#         result[periods:] = arr[:-periods]
#     else:
#         result[:periods] = arr[-periods:]
#     return result

# @nb.njit
# def nb_fillinf(arr, fill_value):
#     '''
#     把所有的inf替换为其他值
#     '''
#     arr = arr.copy()
#     for i in range(len(arr)):
#         if np.isinf(arr[i]):
#             arr[i] = fill_value
#     return arr

# @nb.njit
# def nb_fillna(arr, fill_value):
#     '''
#     把所有的nan替换为其他值
#     '''
#     arr = arr.copy()
#     for i in range(len(arr)):
#         if np.isnan(arr[i]):
#             arr[i] = fill_value
#     return arr