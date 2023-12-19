# numba_meta_factor
用numba写基础算子的工具箱

## 一些处理的说明
1，对于rolling窗口，默认min_periods=windows。因为我觉得前面的数据既然不够，那就不能当作有效的数据来使用。  
2，对于inf，我默认我所有的function的输入都是没有inf的，所有function最后的输出也是不会有inf的。  
3，因为我认为inf是个无效的数据点，所以有个专门把inf转化为nan的函数，nb_fillinf()。  
