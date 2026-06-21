import tensorflow as tf
import numpy as np 
from keras import layers, models

(x_train,y_train) , (x_test,y_test) = tf.keras.datasets.mnist.load_data()

# 1. Cast integer pixel values to float32 and normalize down to a [0.0, 1.0] range
x_train = x_train.astype('float32') /255.0
x_test = x_test.astype('float32') /255.0

#Changing dimention to 4D
x_train = x_train[...,np.newaxis]
x_test = x_test[...,np.newaxis]

# MODEL ARCHITETURE 
model = models.Sequential([
    # First convolutional block 
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPool2D(pool_size=(2,2)),

    # Second convolutional block
    layers.Conv2D(filters=64,kernel_size=(3,3)),
    layers.MaxPool2D(pool_size=(2,2)),

    #flattening
    layers.Flatten(),
    layers.Dropout(0.5), # prevents underfitting 
    layers.Dense(10,activation='softmax')
])

# Compiling the model (Setting rules)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Model training
model.fit(x_train,y_train, epochs=5 , batch_size=64 ,validation_split=0.1)

# Model testing 
test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)

print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

model.save('mnist_cnn_model.keras')
print("Model saved successfully as 'mnist_cnn_model.keras'!")
print("Model summary",model.summary())