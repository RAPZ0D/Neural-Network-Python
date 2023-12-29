import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

df=pd.read_csv('cancer_classification.csv')
df.head()

df.info()

df.describe()

# Select columns for pairplot
columns_for_pairplot = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'benign_0__mal_1']

# Create pairplot
sns.pairplot(df[columns_for_pairplot], hue='benign_0__mal_1', diag_kind='kde')


# Calculate correlation matrix
correlation_matrix = df.corr()

# Create larger heatmap
plt.figure(figsize=(14, 12))  # Adjust the width and height here
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Select columns for distribution plot
columns_for_distribution = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

# Create distribution plot
plt.figure(figsize=(12, 6))
for col in columns_for_distribution:
    sns.histplot(df[col], kde=True, bins=30, alpha=0.5, label=col)
plt.legend()
plt.title('Distribution of Features')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Select columns for boxplot
columns_for_boxplot = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[columns_for_boxplot])
plt.title('Boxplot of Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout

model = Sequential()
model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))


model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(x=X_train, 
          y=y_train, 
          epochs=300,
          validation_data=(X_test, y_test), verbose=1
          )

# Get training history
train_accuracy = model.history.history['accuracy']
val_accuracy =model. history.history['val_accuracy']
train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

# Plotting accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label='Training Accuracy', color='blue')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss', color='green')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

help(EarlyStopping)

early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=25)

model.fit(x=X_train, 
          y=y_train, 
          epochs=500,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

train_accuracy = model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'gold', label='Training Loss')
plt.plot(epochs, val_loss, 'purple', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


from tensorflow.keras.layers import Dropout

help(Dropout)

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

train_accuracy = model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, 'g', label='Training Accuracy')  # Green color for training accuracy
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')   # Blue color for validation accuracy
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'm', label='Training Loss')   # Magenta color for training loss
plt.plot(epochs, val_loss, 'c', label='Validation Loss')  # Cyan color for validation loss
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

