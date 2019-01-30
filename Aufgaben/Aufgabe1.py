import tensorflow as tf
 
#Datenladen
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Datenvorbereiten, kann man hier was verbesern?


# Model aufbauen
model = tf.keras.models.Sequential()

## Eingabeschicht
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
 

# TODO: Füge eine oder mehrere Denselayer hinzu mit bspw. activation=tf.nn.relu


## Ausgabeschicht
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy',tf.keras.metrics.mean_squared_error])

# Tensorboard
# Load with: "tensorboard --logdir ./Graph" inside the same directory
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Modeltrainieren
model.fit(x_train,y_train, epochs=3, callbacks=[tbCallBack])

# Auswerten
val_loss, val_acc, val_mse = model.evaluate(x_test, y_test)
print("")
print("val_loss:", val_loss)
print("val_mse:", val_mse)
print("val_acc:", val_acc)