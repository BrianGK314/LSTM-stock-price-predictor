import math
import pandas as pd
import pandas_datareader as web
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

df = web.DataReader('Dis', data_source='yahoo', start = '2017-01-01', end = '2021-02-18')

#see the graph
close_price = df.filter(['Close'])
plt.figure(figsize=(16,8))
plt.plot(close_price)
plt.title('Disney Stock Price',fontsize=18)
plt.xlabel('Time',fontsize=18)
plt.ylabel('Stock price ($)',fontsize=18)
plt.show()

#configure data

data_set=close_price.values

training_data_len = math.ceil(len(data_set)*0.8)
training_data_len

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set)

training_data = scaled_data[:training_data_len,:]
x_train = []
y_train = []

for i in range(60,len(training_data)):
  x_train.append(training_data[i-60:i,0])
  y_train.append(training_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Build Model

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',
              loss='mse')
              
              
#train model

model.fit(x_train,y_train,epochs=1,batch_size=1)


#generate test data

test_data = scaled_data[training_data_len-60:,:]

x_test=[]
y_test=scaled_data[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape

#predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(y_test-predictions)**2)
#rmse

#viewing prediction
trainig = close_price[:training_data_len]
valid = close_price[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Apple stock prediction')
plt.xlabel('Years')
plt.ylabel('Stock Price ($)')
plt.plot(trainig['Close'])
plt.axis('off') 
plt.plot(valid[['Close','Predictions']])

#predict tomorrow

get_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-02-22')

new_df = get_quote.filter(['Close'])

last_60_days = new_df[-60:].values

last_60_days_scaled = scaler.fit_transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#X_test.shape

prediction = model.predict(X_test)
prediction = scaler.inverse_transform(prediction)

print(prediction)

#actual tomorrow
apple_quote2= web.DataReader('AAPL',data_source='yahoo',start='2021-02-23',end='2021-02-23')
print(apple_quote2['Close'])
