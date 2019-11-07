import random as rd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# original get data
# def getData(path):
#     df = pd.read_csv(path)
#     Y = np.array([])
#     dataset = np.array(df.values)
#     if "price" in df.columns:
#         Y = dataset[:, 1]
#         dataset = np.delete(dataset, 1, 1)
#     dataset = np.delete(dataset, 0, 1)
#     return dataset, Y

# transfer zipcode to one hot encoding
def getData(path):
    df = pd.read_csv(path)
    Y = np.array([])
    zip = pd.get_dummies(df['zipcode'])
    df = df.join(zip)
    df.drop(columns=['id', 'zipcode', 'sale_yr', 'sale_month', 'sale_day'])
    dataset = np.array(df.values)
    if "price" in df.columns:
        Y = dataset[:, 1]
        dataset = np.delete(dataset, 1, 1)
    dataset = np.delete(dataset, 0, 1)
    return dataset, Y

# Read training dataset into X and Y
X_train, Y_train = getData('./train-v3.csv')

# Read validation dataset into X and Y
X_valid, Y_valid = getData('./valid-v3.csv')

# Read test dataset into X
X_test, _ = getData('./test-v3.csv')


def normalize(train,valid,test):
	tmp=train
	mean=tmp.mean(axis=0)
	std=tmp.std(axis=0)
	train=(train-mean)/std
	valid=(valid-mean)/std
	test=(test-mean)/std
	return train,valid,test

X_train,X_valid,X_test=normalize(X_train,X_valid,X_test)

from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(1000, activation='relu', input_dim=X_train.shape[1]),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dropout(0.3),
	keras.layers.Dense(200, activation='relu'),
     keras.layers.Dropout(0.6),
	 keras.layers.Dense(100, activation='relu'),
	  keras.layers.Dropout(0.3),
    keras.layers.Dense(5, activation='relu'),
	
    keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mae')
history = model.fit(X_train, Y_train, batch_size=30, epochs=150, validation_data=(X_valid, Y_valid))


Y_predict = model.predict(X_test)

Y = pd.DataFrame({'id': [i for i in range(1,len(Y_predict)+1)], 'price': Y_predict[:,0]}).astype('int')
Y.to_csv('./test.csv', index= False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='best')
plt.show()
