import pandas as pd
#pandas_datareader is a dataset lib to get data from the web
#run the following command to install in anaconda: conda install -c anaconda pandas-datareader
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_colwidth', 200)

#Financial pandas reader
import pandas_datareader as pdr
#History price initial date and final date 
#Start of a five year exemple of traditional analysis
start_date = pd.to_datetime('2010-01-01')
end_date = pd.to_datetime('2015-01-01')

spy = pdr.data.get_data_yahoo('PETR4.SA', start_date, end_date)

print(spy)

spy_c = spy['Close']

print(spy_c)

fig, ax = plt.subplots(figsize=(15,10))
spy_c.plot(color='b')
plt.title("PETR4", fontsize=20)
plt.savefig('stockgraphex.png')

first_open = spy['Open'].iloc[0]
print(first_open)

last_close = spy['Close'].iloc[-1]
print(last_close)

print(last_close - first_open)

#Price change during the day
spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])
print(spy['Daily Change'])

#Sums price during the day
spy['Daily Change'].sum()

np.std(spy['Daily Change'])

#Price change during the night
spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))
np.std(spy['Overnight Change'])

print(spy['Overnight Change'])

spy[spy['Daily Change'] > 0]['Daily Change'].mean()

spy[spy['Overnight Change'] > 0]['Overnight Change'].mean()

#Calculates the possible return of each type of deal
daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
id_rtn = ((spy['Close'] - spy['Open'])/spy['Open'])*100
on_rtn = ((spy['Open'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100

print(daily_rtn)
print(id_rtn)
print(on_rtn)


def get_stats(s, n=252):
    s = s.dropna()
    wins = len(s[s>0])
    losses = len(s[s<0])
    evens = len(s[s==0])
    mean_w = round(s[s>0].mean(), 3)
    mean_l = round(s[s<0].mean(), 3)
    win_r = round(wins/losses, 3)
    mean_trd = round(s.mean(), 3)
    sd = round(np.std(s), 3)
    max_l = round(s.min(), 3)
    max_w = round(s.max(), 3)
    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
    cnt = len(s)
    print('Trades:', cnt,\
        '\nWins:', wins,\
        '\nLosses:', losses,\
        '\nBreakeven:', evens,\
        '\nWin/Loss Ratio', win_r,\
        '\nMean Win:', mean_w,\
        '\nMean Loss:', mean_l,\
        '\nMean', mean_trd,\
        '\nStd Dev:', sd,\
        '\nMax Loss:', max_l,\
        '\nMax Win:', max_w,\
        '\nSharpe Ratio:', sharpe_r)

get_stats(daily_rtn)
get_stats(id_rtn)
get_stats(on_rtn)
#End of exemple

#Starting training
start_date = pd.to_datetime('2000-01-01')
stop_date = pd.to_datetime('2018-06-01')
sp = pdr.data.get_data_yahoo('PETR4.SA', start_date, stop_date) #The Dataset

print(sp)

fig, ax = plt.subplots(figsize=(15,20))
sp['Close'].plot(color='b')
plt.title("PETR4", fontsize=20)
plt.savefig('stockgraphtrain.png')

longday_rtn = ((sp['Close'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100
longid_rtn = ((sp['Close'] - sp['Open'])/sp['Open'])*100
longon_rtn = ((sp['Open'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100

#Check if the analysis is right
print('Long Day: ', (sp['Close'] - sp['Close'].shift(1)).sum())
print('Long Intra Day: ', (sp['Close'] - sp['Open']).sum())
print('Long OVernight: ', (sp['Open'] - sp['Close'].shift(1)).sum())

get_stats(longday_rtn)
get_stats(longid_rtn)
get_stats(longon_rtn)

for i in range(1, 21, 1):
        sp.loc[:, 'Close Minus' + str(i)] = sp['Close'].shift(i)
        sp20 = sp[[x for x in sp.columns if 'Close Minus' in x or x == 'Close']].iloc[20:,]
print(sp20)

#Scikit-learn is a machine learning lib for python
#Run the following command to install the lib in anaconda: conda install -c anaconda scikit-learn

#SVMs are supervised learning methods - for classification, regression and outliners detection
from sklearn.svm import SVR
clf = SVR(kernel='linear', verbose=1)
x_train = sp20[:-1000]
y_train = sp20['Close'].shift(-1)[:-1000]
x_test = sp20[-1000:]
y_test = sp20['Close'].shift(-1)[-1000:]

model = clf.fit(x_train, y_train)
preds = model.predict(x_test)

tf = pd.DataFrame(list(zip(y_test, preds)), columns=['Next Day Close', 'Predicted Next Close'], index=y_test.index)
print(tf)

cdc = sp[['Close']].iloc[-1000:]
ndo = sp[['Open']].iloc[-1000:].shift(-1)
tf1 = pd.merge(tf, cdc, left_index=True, right_index=True)
tf2 = pd.merge(tf1, ndo, left_index=True, right_index=True)
tf2.columns = ['Next Day Close', 'Predicted Next Close', 'Current Day Close', 'Next Day Open']
print(tf2)

def get_signal(r):
        if r['Predicted Next Close'] > r['Next Day Open']:
                return 1
        else:
                return 0

def get_ret(r):
        if r['Signal'] == 1:
                return ((r['Next Day Close'] - r['Next Day Open'])/r['Next Day Open']) * 100
        else:
                return 0

tf2 = tf2.assign(Signal = tf2.apply(get_signal, axis=1))
tf2 = tf2.assign(PnL = tf2.apply(get_ret, axis=1))
print(tf2)

print((tf2[tf2['Signal']==1]['Next Day Close'] - tf2[tf2['Signal']==1]['Next Day Open']).sum())
print((sp['Close'].iloc[-1000:] - sp['Open'].iloc[-1000:]).sum())

get_stats((sp['Close'].iloc[-1000:] - sp['Open'].iloc[-1000:])/sp['Open'].iloc[-1000:] * 100)
get_stats(tf2['PnL'])

def get_signal(r):
    if r['Predicted Next Close'] > r['Next Day Open'] + 1:
        return 1
    else:
        return 0
    
def get_ret(r):
    if r['Signal'] == 1:
        return ((r['Next Day Close'] - r['Next Day Open'])/r['Next Day Open']) * 100
    else:
        return 0
    
tf2 = tf2.assign(Signal = tf2.apply(get_signal, axis=1))
tf2 = tf2.assign(PnL = tf2.apply(get_ret, axis=1))
print(tf2)

print((tf2[tf2['Signal']==1]['Next Day Close'] - tf2[tf2['Signal']==1]['Next Day Open']).sum())
get_stats(tf2['PnL'])

X_train = sp20[:-2000]
y_train = sp20['Close'].shift(-1)[:-2000]
X_test = sp20[-2000:-1000]
y_test = sp20['Close'].shift(-1)[-2000:-1000]

model = clf.fit(X_train, y_train)
preds = model.predict(X_test)

tf = pd.DataFrame(list(zip(y_test, preds)), columns=['Next Day Close','Predicted Next Close'], index=y_test.index)
cdc = sp[['Close']].iloc[-2000:-1000]
ndo = sp[['Open']].iloc[-2000:-1000].shift(-1)
tf1 = pd.merge(tf, cdc, left_index=True, right_index=True)
tf2 = pd.merge(tf1, ndo, left_index=True, right_index=True)
tf2.columns = ['Next Day Close', 'Predicted Next Close', 'Current Day Close', 'Next Day Open']

def get_signal(r):
    if r['Predicted Next Close'] > r['Next Day Open'] + 1:
        return 0
    else:
        return 1
def get_ret(r):
    if r['Signal'] == 1:
        return ((r['Next Day Close'] - r['Next Day Open'])/r['Next Day Open']) * 100
    else:
        return 0
    
tf2 = tf2.assign(Signal = tf2.apply(get_signal, axis=1))
tf2 = tf2.assign(PnL = tf2.apply(get_ret, axis=1))

print((tf2[tf2['Signal']==1]['Next Day Close'] - tf2[tf2['Signal']==1]['Next Day Open']).sum())
print((sp['Close'].iloc[-2000:-1000] - sp['Open'].iloc[-2000:-1000]).sum())

from scipy.spatial.distance import euclideany
#Python lib for Dynamic Time Warping
#Run the following command to install in anaconda: conda install -c bioconda fastdtw 
from fastdtw import fastdtw

def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance

tseries = []
tlen = 5
for i in range(tlen, len(sp), tlen):
    pctc = sp['Close'].iloc[i-tlen:i].pct_change()[1:].values * 100
    res = sp['Close'].iloc[i-tlen:i+1].pct_change()[-1] * 100
    tseries.append((pctc, res))

print(tseries[0])

dist_pairs = []
for i in range(len(tseries)):
    for j in range(len(tseries)):
        dist = dtw_dist(tseries[i][0], tseries[j][0])
        dist_pairs.append((i,j,dist,tseries[i][1], tseries[j][1]))

dist_frame = pd.DataFrame(dist_pairs, columns=['A','B','Dist', 'A Ret', 'B Ret'])
sf = dist_frame[dist_frame['Dist']>0].sort_values(['A','B']).reset_index(drop=1)
sfe = sf[sf['A']<sf['B']]

winf = sfe[(sfe['Dist']<=1)&(sfe['A Ret']>0)]
print(winf)

plt.plot(np.arange(4), tseries[6][0])
plt.savefig('tranedreturn.png')

plt.plot(np.arange(4), tseries[598][0])
plt.savefig('tradereturn2.png')

excluded = {}
return_list = []
def get_returns(r):
    if excluded.get(r['A']) is None:
        return_list.append(r['B Ret'])
        if r['B Ret'] < 0:
            excluded.update({r['A']:1})
winf.apply(get_returns, axis=1);

get_stats(pd.Series(return_list))

#Holly shit, that's some code
