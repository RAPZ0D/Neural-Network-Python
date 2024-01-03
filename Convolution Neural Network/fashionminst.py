import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train[1]

y_train[1]

plt.imshow(x_train[1])
plt.axis('off');

print("The max value is: ",x_train.max())
print("The min value is: ",x_train.min())

x_train = x_train/255
x_test = x_test/255

x_train.shape

x_test.shape

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_train

from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))

# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN LAYER
model.add(Flatten())

# DENSE LAYER
model.add(Dense(128, activation='relu'))

# FINAL LAYER
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

model.fit(x_train,y_cat_train,epochs=20)

model.history.history

# Accessing training history
history = model.history.history

# Plotting accuracy
plt.figure(figsize=(8, 6))
plt.plot(history['accuracy'], label='Training Accuracy', color='gold')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss
plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label='Training Loss', color='purple')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Accessing training history
history = model.history.history

# Creating a histogram for accuracy distribution
plt.figure(figsize=(8, 6))
sns.histplot(history['accuracy'], kde=True, color='orange')
plt.title('Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

# Creating a histogram for loss distribution
plt.figure(figsize=(8, 6))
sns.histplot(history['loss'], kde=True, color='purple')
plt.title('Loss Distribution')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.show()


model.evaluate(x_test,y_cat_test)
