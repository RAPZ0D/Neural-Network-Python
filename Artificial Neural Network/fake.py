import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('fake_reg.csv')

df.head()
df.info

df.shape

df.isnull().sum()

# Histogram for 'price'
plt.hist(df['price'], bins=20, color='skyblue')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Price')
plt.show()

# Scatter plot for 'feature1' vs 'price'
plt.scatter(df['feature1'], df['price'], color='orange')
plt.xlabel('Feature 1')
plt.ylabel('Price')
plt.title('Scatter Plot: Feature 1 vs Price')
plt.show()

# Box plot for 'feature2'
plt.boxplot(df['feature2'])
plt.ylabel('Feature 2')
plt.title('Box Plot: Feature 2')
plt.show()


# Pairplot for multiple features
sns.pairplot(df)
plt.title('Pairplot of Features')
plt.show()


# Correlation heatmap
corr = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

from sklearn.model_selection import train_test_split
# Features
X = df[['feature1','feature2']].values

# Label
y = df['price'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
help(StandardScaler)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

# Final output node for prediction
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')

model.fit(X_train,y_train,epochs=100)

model.history.history

loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)),y=loss)
plt.title("Training Loss per Epoch")
plt.show()

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.distplot(a=loss)
plt.title("Training Loss per Epoch")
plt.show()

training_score = model.evaluate(X_train,y_train,verbose=0)
test_score = model.evaluate(X_test,y_test,verbose=0)

training_score

test_score

test_predictions = model.predict(X_test)
test_predictions

pred_df = pd.DataFrame(y_test,columns=['y_test'])
pred_df

test_predictions = pd.Series(test_predictions.reshape(300,))
test_predictions

pred_df = pd.concat([pred_df,test_predictions],axis=1)

pred_df.columns = ['y_test','model_predictions']

pred_df

sns.scatterplot(x='y_test',y='model_predictions',data=pred_df)

