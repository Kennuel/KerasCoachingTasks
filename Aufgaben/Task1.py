import tensorflow as tf
 
#Load data 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# data preperation, what needs to be done? Have you heard of normalization before?


# Model aufbauen
model = tf.keras.models.Sequential()

## Eingabeschicht
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
 

# TODO: Add one or more dense layers with the standard activation function: tf.nn.relu


## Ausgabeschicht
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Tensorboard
# Load with: "tensorboard --logdir ./Graph" inside the same directory
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Modeltrainieren
model.fit(x_train,y_train, epochs=3, callbacks=[tbCallBack])

# Auswerten
val_loss, val_acc = model.evaluate(x_test, y_test)
print("")
print("val_loss:", val_loss)
print("val_acc:", val_acc)