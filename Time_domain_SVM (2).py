#!/usr/bin/env python
# coding: utf-8

# In[87]:


import os
import numpy as np
import scipy.io
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# dataクラス
class O_Data:
    def __init__(self, eeg, index, label):
        self.eeg = eeg
        self.index = index
        self.label = label
        self.stack = []

    def push(self, item):
        self.stack.append(item)

o_data65 = list()
R_data = list()  #  実運動のデータを格納
RI_data = list()  # 実運動＋想起運動のデータを格納
All_data = list()  # 全データを格納

def Standardization(data): #標準化
    after_data = scipy.stats.zscore(data)
    b = np.average(after_data)
    c = np.var(after_data)
    return after_data

def resampleing(data, fs1, fs2):
    sec = len(data) / fs1
    n = int(fs2 * sec)
    resample_data = signal.resample(data, n)  # アンチエイリアス・リサンプリング
    return resample_data

def butter_lowpass(cutfreq, fs, order=4):
    nyq = fs / 2.0
    low = cutfreq / nyq
    b, a = signal.butter(order, low, btype='low', fs=fs, output='ba')
    return b, a

def butter_lowpass_filt(data, cutfreq, fs, order=4):
    b, a = butter_lowpass(cutfreq, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# FFT変換
def FFT(data):
    F = np.fft.fft(data)
    F = F/max(F)
    F = abs(F)
    return F

# 積分
def calc_integral(start, stop, x, y):
    sum = 0.0
    j = start
    for i in range(start, stop):
        sum = sum + (y[j] + y[j + 1]) * (x[j + 1] - x[j]) / 2
        j += 1
    return sum


# In[88]:


file_name = list()  # すべての.matファイルの名前
for file in os.listdir():
    base, ext = os.path.splitext(file)
    if ext == '.mat':
        file_name = sorted(file_name)
        file_name.append(file)
print('Filelist')
print(file_name)

# 変数など
data_directry = ''
file_num = len(file_name)
trial_num = 30
all_trial = file_num * trial_num
all_d = 0
CH = 64

A = 0  # トライアルカウント
B = 0  # セッションカウント
for s in range(file_num):
    Dictionary = scipy.io.loadmat(file_name[s])
    for t in range(trial_num):
        for i, key in enumerate(Dictionary.keys()):
            if i > 2:
                a = Dictionary[key]
                b = a[0, t][0][0]
                eeg = b[0][0:64, :]
                index = b[1]
                samplerate = b[2]
                if s < 3:
                    label = 1

                else:
                    label = 2

                #label = b[5][0, 0]

                o_data = O_Data(eeg, index, label)
                o_data65.append(o_data)

R_num = 0
RI_num = 0

print("data_loading")


# In[89]:


for n in range(all_trial):  # all_trial
    number = list()  # restのindex
    time = list()
    all_d = sum(len(v) for v in o_data65[n].index) - 1
    for i in range(all_d + 1):
        time.append(i)
    eeg = o_data65[n].eeg  # トライアルnのeeg
    index = o_data65[n].index  # トライアルnのindex
    label = o_data65[n].label  # トライアルnのlabel
    for i in range(all_d):
        if index[:, i] == 0:
            number.append(i)
    rest_time = len(number)  # rest時間のindex数



    # すべてのチャンネルで行う



    for j in range(CH):
        onech_eeg = eeg[j, :]  # 一つのチャンネルのeeg
        number = np.array(number)
        time = np.array(time)

        detrend = signal.detrend(onech_eeg)  # ベースライン補正
        detrend_128 = resampleing(detrend, 1024, 128)  # ダウンサンプリング
        standard = Standardization(detrend_128)[256:]  # 標準・2秒分(rest)カット
        filtered = butter_lowpass_filt(standard, 50, 128)  # LPF

    max_data = len(filtered)
    del filtered[max_data:669]
    All_data.append(filtered)
        
   

    if label == 1:
            R_data.append(0)


    else:
            R_data.append(1)

R_data = np.array(R_data)
All_data = np.array(All_data)

print(np.shape(R_data))
print(np.shape(All_data))


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(All_data, R_data, test_size=0.25)



model =SVC(kernel='linear')
model.fit(X_train,Y_train)

SVC(C=1.0, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

print(model.score(X_train, Y_train))
print(model.score(X_test, Y_test))


# In[ ]:


params = {
    "C":np.arange(0.1,1,0.05),
    "kernel":["linear", "poly", "rbf", "sigmoid"],
    "gamma":np.arange(0.0001,0.1,0.05)
}
grid = GridSearchCV(model, params,scoring="accuracy", cv=5)


# In[ ]:


grid.fit(X_train, Y_train)


# In[ ]:


print(grid.best_score_)
print(grid.best_params_)


# In[ ]:


pred = grid.predict(X_test)
print(classification_report(Y_test, pred))


# In[ ]:




