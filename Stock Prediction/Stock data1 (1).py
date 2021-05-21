#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import investpy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[2]:


stocks = investpy.get_stock_historical_data(stock='GOOGL',country='United States',
                                            from_date='01/01/2010',
                                            to_date='01/12/2020')


# In[3]:


stocks


# In[4]:


closing = stocks.filter(["Close"])


# In[5]:


closing = pd.DataFrame (stocks,columns=['Close'])


# In[6]:


plt.figure(figsize=(15, 7))
plt.plot(closing)
plt.title("GOOGLE Closing Price")
plt.xlabel("DATE")
plt.ylabel("Stock Price")
plt.show()


# In[7]:


closing


# In[8]:


closing.to_csv('close.csv')


# In[9]:


closing1=pd.read_csv('close.csv')


# In[10]:


closing1


# In[11]:


closing1['Date'] = pd.to_datetime(closing1['Date'])
closing1 = closing1.set_index(closing1['Date'])
closing1 = closing1.sort_index()


# In[12]:


train = closing1['2010,04,01':'2019,31,12']
test  = closing1['2020-01-01':]
print('Train Dataset:',train.shape)
print('Test Dataset:',test.shape)
train1= train.to_csv('train.csv')
test1=test.to_csv('test.csv')
train1=pd.read_csv('train.csv')
test1=pd.read_csv('test.csv')


# In[13]:


training_set = train.iloc[:, 1:2].values
testing_set = test.iloc[:, 1:2].values


# In[14]:


scaler = MinMaxScaler(feature_range=(0,1))


# In[15]:


scaled_data =scaler.fit_transform(training_set)


# In[16]:


X_train = []
y_train = []


# In[17]:


for i in range(60, 2035):
    X_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])


# In[18]:


X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[ ]:





# LSTM model

# In[ ]:





# In[19]:


regressor = Sequential()


# In[20]:


regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


# In[21]:


regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# In[22]:


regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# In[23]:


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[24]:


regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[26]:


dataset_total = pd.concat((train1['Close'], test1['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test1) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)


# In[27]:


X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])


# In[28]:


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[29]:


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# In[ ]:





# In[30]:


plt.figure(figsize=(15, 7))
plt.plot(testing_set, color = 'black', label = 'GOOGLE STOCK PRICE')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted GOOGLE Stock Price')
plt.title('GOOGLE Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('GOOGLE Stock Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




