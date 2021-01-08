


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import iqr
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pandas as import pd


dataset = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv' , low_memory = False)
dataset['timestamp'] = pd.to_datetime(dataset['timestamp']) # change the type of timestamp column
dataset.plot(x='timestamp', y='value', figsize=(10,5),label='Datasets')
plt.xlabel('Timestamp', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.legend()
dataset['Days'] = dataset['timestamp'].dt.dayofweek
dataset['Weekday'] = (dataset['Days'] < 5).astype(int)
dataset['time_epoch'] = (dataset['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
dataset.head(10)

#========================================================================== IQR

def Interquartile_Range(dataset):
    sorted(dataset['value'])
    Q1,Q2, Q3= np.percentile(dataset['value'],[25,50,75])
    IQR = iqr(dataset['value'])
    band_low = Q1 -(1.5 * IQR) 
    band_up = Q3 +(1.5 * IQR)    
    n = 0
    for row in dataset.iterrows():
        if row[1]['value'] > band_up or row[1]['value'] < band_low:
            n = n + 1            
    print("Number of Anomalies is", n)
    print(Q1, Q2, Q3)
Interquartile_Range(dataset)

#============================================================= Isolation Forest
X_train = dataset.loc[:7000, ['value','Days']]
X_valid = dataset.loc[7000:, ['value','Days']]
model =  IsolationForest(n_estimators=1000, contamination = 0.05)
model.fit(X_train)
y_pred_train = model.predict(X_train)
dataset['anomaly_IF']=pd.Series(y_pred_train)
print(dataset['anomaly_IF'].value_counts())
print("Normal, Anomalies\n",dataset['anomaly_IF'].value_counts().values)
scores = model.decision_function(X_valid)
plt.figure(figsize=(12, 8))
plt.hist(scores, bins=50)
#Visualisation of anomaly
fig, ax = plt.subplots(figsize=(15,10))
dataset['time_epoch'] = (dataset['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
anomaly_data = dataset.loc[dataset['anomaly_IF'] == -1, ['time_epoch', 'value']] #anomaly
ax.plot(dataset['time_epoch'], dataset['value'], color='green',label='Normal')
ax.scatter(anomaly_data['time_epoch'],anomaly_data['value'], color='blue',label='Anomalies')
plt.title("Isolation Forest")
plt.legend()
plt.show()

#========================================================================= SVM
X = dataset[['value', 'Days']]
min_max_scaler = preprocessing.StandardScaler()
X_scale = min_max_scaler.fit_transform(X)
model_SVM =  OneClassSVM(nu=0.95 * 0.05)
X_SVM = pd.DataFrame(X_scale)
model_SVM.fit(X_SVM)

dataset['anomaly_SVM'] = pd.Series(model_SVM.predict(X_SVM))
print(dataset['anomaly_SVM'].value_counts())
fig, ax = plt.subplots(2,sharex= True , figsize=(15,10))
fig.suptitle("One Class Support Vector Machine")
param = dataset.loc[dataset['anomaly_SVM'] == -1, ['time_epoch', 'value']] #anomaly
param_1 = dataset.loc[dataset['anomaly_SVM'] == -1, ['time_epoch', 'Days']] #anomaly
ax[0].plot(dataset['time_epoch'], dataset['value'], color='green',label='Normal')
ax[0].scatter(param['time_epoch'],param['value'], color='blue',label='Anomaly')
ax[1].plot(dataset['time_epoch'], dataset['Days'], color='green',label='Normal')
ax[1].scatter(param_1['time_epoch'],param_1['Days'], color='blue',label='Anomaly')
plt.legend()
plt.show()