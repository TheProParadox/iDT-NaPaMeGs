import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras import layers


path = "/scratch/proj/napamegs/mat_out/wavel_random/"
df = pd.read_csv(path+'wavel_random1.csv')
df2 = pd.read_csv(path+'wavel_random2.csv')
df = df.append(df2, ignore_index=True) #append by maintaining the continuity of indexing

dataset = df.values #df to np_array
X = dataset[:,:6]
y = dataset[:,6:]
x_train , x_test ,y_train ,y_test = train_test_split(X,y, test_size = 0.3,random_state=50)



input = Input(shape=(6,))
x1 = Dense(64, activation='relu')(input)
x2 = Dense(128, activation='relu')(x1)
x3 = Dense(256, activation='relu')(x2)
x4 = Dense(128, activation='relu')(x3)
x5 = Dense(64, activation='relu')(x4)
x6 = layers.add([x5, x1])
output = Dense(4, activation='relu')(x6)
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
print(model.summary())

history = model.fit(x_train, y_train, epochs=100, batch_size=400000, verbose=1)
prediction = model.evaluate(x_test, y_test, verbose=1)
print("prediction: %f, actual: %f" %prediction %y_test)
model.save("mlp_symmetric.h5")

'''
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE LOSS VARIATION')
plt.ylabel('value')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('MSE LOSS VARIATION')
plt.ylabel('value')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()'''