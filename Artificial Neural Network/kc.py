import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('kc_house_data.csv')

df.isnull().sum()


df.describe()


df.info()

df.columns

plt.figure(figsize=(8, 6))
plt.hist(df['price'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()


# Selecting numeric columns for pairplot
numeric_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

sns.pairplot(df[numeric_columns])
plt.suptitle('Pairplot of Numeric Features', y=1.02)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.title('Price Variation with Bedrooms')
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df['sqft_living'], df['price'], alpha=0.5)
plt.xlabel('Sqft Living Area')
plt.ylabel('Price')
plt.title('Price vs Sqft Living Area')
plt.show()


# Selecting only numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64'])

# Removing non-numeric columns like 'id'
numeric_columns = numeric_columns.drop(columns=['id'])

# Calculating correlation
corr = numeric_columns.corr()

# Plotting heatmap with a larger size (e.g., figsize=(12, 10))
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Creating a displot for the 'price' column with customized colors
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()

df = df.drop(['id','date','zipcode'],axis=1)

X = df.drop('price',axis=1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=300)

loss = pd.DataFrame(model.history.history)

loss.plot()
plt.show()

loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'b', label='Training Loss')  # Blue line for training loss
plt.plot(epochs, val_loss, 'r', label='Validation Loss')  # Red line for validation loss
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

predictions = model.predict(X_test)

mean_absolute_error(y_test,predictions)

np.sqrt(mean_squared_error(y_test,predictions))

explained_variance_score(y_test,predictions)

plt.figure(figsize=(8, 6))

# Scatter plot for actual vs. predicted values
plt.scatter(y_test, predictions, alpha=0.6, color='blue', label='Actual vs. Predicted')

# Perfect predictions line
plt.plot(y_test, y_test, 'r', label='Perfect Prediction')

plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()


errors = y_test.values.reshape(6480, 1) - predictions
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Errors')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()


