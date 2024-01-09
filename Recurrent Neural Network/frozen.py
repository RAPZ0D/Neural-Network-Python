import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('Frozen_Dessert_Production.csv',index_col='DATE',parse_dates=True)
df.head()

df.columns = ['Production']
df.head()

df.plot(figsize=(10,10));

test_size = 24
test_ind = len(df)- test_size
train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length = 18
n_features=1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=1)

early_stop = EarlyStopping(monitor='loss',patience=2, mode='min')

model.fit_generator(generator,epochs=20,
                    validation_data=validation_generator,
                   callbacks=[early_stop])

# Accessing loss and validation loss from model history
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

# Plotting loss and validation loss with different colors (green and purple)
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'orange', label='Training Loss')  
plt.plot(epochs, val_loss, 'red', label='Validation Loss')  
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import seaborn as sns

# Accessing loss and validation loss from model history
loss_values = model.history.history['loss']
val_loss_values = model.history.history['val_loss']

# Creating separate plots for training loss and validation loss
plt.figure(figsize=(10, 5))  # Adjust figure size if needed

plt.subplot(1, 2, 1)  # Subplot for training loss
sns.histplot(loss_values, color='purple', label='Training Loss', kde=True)
plt.title('Distribution of Training Loss')
plt.xlabel('Loss')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)  # Subplot for validation loss
sns.histplot(val_loss_values, color='gold', label='Validation Loss', kde=True)
plt.title('Distribution of Validation Loss')
plt.xlabel('Loss')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()  # Adjust spacing between subplots
plt.show()



test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions

test

test.plot();

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test['Production'],test['Predictions']))



