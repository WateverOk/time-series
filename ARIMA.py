import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA
from scipy.io import loadmat
from time import *

# 构建时间序列数据
Serial_number = 1000
test_data_size = 20
data = loadmat("D:/py/pytorch_test/DeepLearning_TempPredict.mat")
all_data_sum = data['caoyuan']
all_data = all_data_sum[:, :].T
train_data = all_data[:-test_data_size,:]

time_series = pd.Series(train_data[:, Serial_number])
time_series.index = pd.Index(sm.tsa.datetools.dates_from_range('1978','2010'))
dat1 = time_series
time_series.plot()
plt.show()

# 由于其指数性质，可先将其log进行线性化
time_series = np.log(time_series)
dat = time_series
time_series.plot()
plt.show()

# 为了确定其稳定性，对取对数后的数据进行 adf 检验
t = sm.tsa.stattools.adfuller(time_series, )
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)


time_series = time_series.diff(1)
time_series = time_series.dropna(how=any)
time_series.plot()
plt.show()
t=sm.tsa.stattools.adfuller(time_series)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

plot_acf(time_series)
plot_pacf(time_series)
plt.show()

r,rac,Q = sm.tsa.acf(time_series, qstat=True)
prac = pacf(time_series,method='ywmle')
table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])
print(table)


p,d,q = (1,1,2)
arma_mod = ARMA(time_series,(p,d,q)).fit(disp=-1,method='mle')
summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))
print(summary)

(p, q) = (sm.tsa.arma_order_select_ic(time_series,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
#这里需要设定自动取阶的 p和q 的最大值，即函数里面的max_ar,和max_ma。ic 参数表示选用的选取标准，这里设置的为aic,当然也可以用bic。然后函数会算出每个 p和q 组合(这里是(0,0)~(3,3)的AIC的值，取其中最小的,这里的结果是(p=0,q=1)。

# 残差和白噪声检验
arma_mod = ARMA(time_series,(0,1,1)).fit(disp=-1,method='mle')
resid = arma_mod.resid
t=sm.tsa.stattools.adfuller(resid)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

arma_model = sm.tsa.ARMA(time_series,(0,1)).fit(disp=-1,maxiter=100)

predict_data = arma_model.predict(start=str(1979), end=str(2010+3), dynamic = False)
predict_data = pd.Series([dat[0]], index=[dat.index[0]]).append(predict_data).cumsum()
predict_data = np.exp(predict_data)
plt.ion()
dat1.plot(legend=True,label='actual value')
predict_data.plot(legend=True,label='predict value')
plt.ioff()
plt.show()