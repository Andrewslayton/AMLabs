from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping

import math
import matplotlib.pyplot as plt

def create_RNN(hidden_units,dense_units,input_shape, activation, ):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units = dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
demo_model = create_RNN(2,1,(3,1), activation = ['linear', 'linear'])
wx=demo_model.get_weights()[0]
wh=demo_model.get_weights()[1]
bh=demo_model.get_weights()[2]
wy=demo_model.get_weights()[3]
by=demo_model.get_weights()[4]

x = np.array([1, 2, 3])

x_input = np.reshape(x, (1, 3, 1))

y_pred_model = demo_model.predict(x_input)

m = 2
h0 = np.zeros(m)


h1 = np.dot(x[0], wx) + h0 + bh
h2 = np.dot(x[1], wx) + np.dot(h1, wh) + bh
h3 = np.dot(x[2], wx) + np.dot(h2, wh) + bh
o3 = np.dot(h3, wy) + by

def get_train_test(url,split_percent=0.8):
    df=read_csv(url,usecols=[1],engine='python')
    data=np.array(df.values.astype('float32')) 
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    n=len(data)
    split = int(n*split_percent)
    train_data=data[range(split)]
    test_data=data[split:]
    return train_data,test_data,data
sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
train_data,test_data,data=get_train_test(sunspots_url)

def get_XY(dat,time_steps):
    Y_ind = np.arange(time_steps,len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
    x=dat[range(time_steps*rows_x)]
    X = np.reshape(x, (rows_x, time_steps, 1))
    return X,Y
time_steps = 12
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model = create_RNN(hidden_units=100,dense_units=1,input_shape=(time_steps,1), activation=['tanh', 'tanh'])
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stop], validation_split=0.2)

def print_error(trainY,testY,train_predict,test_predict):
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    print('Train RMSE: %.3f' % (train_rmse))
    print ('Test RMSE: %.3f' % (test_rmse))
train_predict = model.predict(trainX)
test_predict = model.predict(testX)
print_error(trainY,testY,train_predict,test_predict)

def plot_result(trainY,testY,train_predict,test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize = (15,6), dpi = 80)
    plt.plot(range(rows),actual)
    plt.plot(range(rows),predictions)
    plt.axvline(x=len(trainY),color='r')
    plt.legend(['Actual','Predicted',])
    plt.xlabel("Observation nmber after given time steps")
    plt.ylabel("sunspots scaled")
    plt.title("actual and predicted ")
    plt.show()
plot_result(trainY,testY,train_predict,test_predict)