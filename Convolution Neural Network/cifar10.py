import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape

y_train.shape

plt.imshow(x_train[1])
plt.axis('off');

plt.imshow(x_train[12])
plt.axis('off');

x_train[0]

x_train = x_train/255
x_test = x_test/255

from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))

# MAX POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))

# MAXPOOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN LAYER
model.add(Flatten())

# DENSE LAYER
model.add(Dense(256, activation='relu'))

# FINAL LAYER
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy',patience=3,mode='max')

model.fit(x_train,y_cat_train,epochs=30,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

new_metrics = pd.DataFrame(model.history.history)

new_metrics.head()

# Plotting accuracy and validation accuracy with new colors
plt.figure(figsize=(10, 5))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(new_metrics['accuracy'], label='Training Accuracy', color='skyblue')
plt.plot(new_metrics['val_accuracy'], label='Validation Accuracy', color='salmon')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss and validation loss with new colors
plt.subplot(1, 2, 2)
plt.plot(new_metrics['loss'], label='Training Loss', color='limegreen')
plt.plot(new_metrics['val_loss'], label='Validation Loss', color='tomato')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Set the style
sns.set(style="whitegrid")

# Create subplots for each metric with new colors
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.histplot(new_metrics['accuracy'], color='skyblue', kde=True)
plt.title('Training Accuracy Distribution')

plt.subplot(2, 2, 2)
sns.histplot(new_metrics['val_accuracy'], color='salmon', kde=True)
plt.title('Validation Accuracy Distribution')

plt.subplot(2, 2, 3)
sns.histplot(new_metrics['loss'], color='limegreen', kde=True)
plt.title('Training Loss Distribution')

plt.subplot(2, 2, 4)
sns.histplot(new_metrics['val_loss'], color='tomato', kde=True)
plt.title('Validation Loss Distribution')

plt.tight_layout()
plt.show()

new_metrics[['accuracy','val_accuracy']].plot()

new_metrics[['loss','val_loss']].plot()

print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))

my_image = x_test[16]

plt.imshow(my_image)
plt.axis('off');

model.predict(my_image.reshape(1,32,32,3))

