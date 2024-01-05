import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape

single_image = x_train[0]

single_image

single_image.shape

plt.imshow(single_image)
plt.axis('off');

y_train


y_test

from tensorflow.keras.utils import to_categorical

y_train.shape

y_example = to_categorical(y_train)
y_example

y_example[0]

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

print("The Max value of the image is: ",single_image.max())
print("The min value of the image is: ",single_image.min())

x_train = x_train/255
x_test = x_test/255

scaled_single = x_train[0]
scaled_single.max()

plt.imshow(scaled_single)
plt.axis('off');

x_train.shape

x_train = x_train.reshape(60000, 28, 28, 1)
x_train

x_train.shape

x_test = x_test.reshape(10000,28,28,1)
x_test.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu'))

# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN LAYER 
model.add(Flatten())

# DENSE HIDDEN LAYER
model.add(Dense(128, activation='relu'))

# LAST LAYER
model.add(Dense(10, activation='softmax'))

#COMPILING THE MODEL
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train,y_cat_train,epochs=25,validation_data=(x_test,y_cat_test))

print("The first metric is: ",model.metrics_names[0])
print("The second metric is: ",model.metrics_names[1])

new_metrics = pd.DataFrame(model.history.history)

new_metrics.head()

# Plotting accuracy and validation accuracy
plt.figure(figsize=(10, 5))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(new_metrics['accuracy'], label='Training Accuracy', color='blue')
plt.plot(new_metrics['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss and validation loss
plt.subplot(1, 2, 2)
plt.plot(new_metrics['loss'], label='Training Loss', color='green')
plt.plot(new_metrics['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Set the style
sns.set(style="whitegrid")

# Create subplots for each metric
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.histplot(new_metrics['accuracy'], color='blue', kde=True)
plt.title('Training Accuracy Distribution')

plt.subplot(2, 2, 2)
sns.histplot(new_metrics['val_accuracy'], color='orange', kde=True)
plt.title('Validation Accuracy Distribution')

plt.subplot(2, 2, 3)
sns.histplot(new_metrics['loss'], color='green', kde=True)
plt.title('Training Loss Distribution')

plt.subplot(2, 2, 4)
sns.histplot(new_metrics['val_loss'], color='red', kde=True)
plt.title('Validation Loss Distribution')

plt.tight_layout()
plt.show()

print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))