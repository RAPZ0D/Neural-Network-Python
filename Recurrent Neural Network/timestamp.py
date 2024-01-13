import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import pandas as pd 

df = pd.read_csv('RSCCASN.csv',index_col='DATE',parse_dates=True)

df.head()

df.columns = ['Sales'] 

df.plot(figsize=(12,8));

test_size = 18
test_ind = len(df)- test_size
train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length = 12
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
X,y = generator[0]

print(f'Given the Array: {X.flatten()}')
print(f'Predict this y: {y}')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

n_features = 1

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

early_stop = EarlyStopping(monitor='loss',patience=2,mode='min')
validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=1)

model.fit_generator(generator,epochs=30,
                    validation_data=validation_generator,
                   callbacks=[early_stop])

model.history.history

# Accessing loss and validation loss from model history
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

# Plotting loss and validation loss with different colors (green and purple)
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training Loss')  # 'g' for green
plt.plot(epochs, val_loss, 'purple', label='Validation Loss')  # 'purple' for purple
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accessing loss and validation loss from model history
loss_values = model.history.history['loss']
val_loss_values = model.history.history['val_loss']

# Creating separate plots for training loss and validation loss
plt.figure(figsize=(10, 5))  # Adjust figure size if needed

plt.subplot(1, 2, 1)  # Subplot for training loss
sns.histplot(loss_values, color='blue', label='Training Loss', kde=True)
plt.title('Distribution of Training Loss')
plt.xlabel('Loss')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)  # Subplot for validation loss
sns.histplot(val_loss_values, color='red', label='Validation Loss', kde=True)
plt.title('Distribution of Validation Loss')
plt.xlabel('Loss')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


