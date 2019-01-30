import tensorflow as tf
 
#Datenladen
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Datenvorbereiten
x_train = tf.keras.utils.normalize(x_train, 1)
x_test = tf.keras.utils.normalize(x_test, 1)

CHANNELS = 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, CHANNELS)
x_test = x_test.reshape(x_test.shape[0], 28, 28, CHANNELS)

y_train =tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
 
# Modelaufbauen

model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Conv2D(28, (4, 4), activation='relu', input_shape=(28, 28, CHANNELS)))

# TODO: Füge ein max pooling layer hinzu

# TODO: Füge eine oder mehrere Conv2D layer hinzu mit bspw. activation=tf.nn.relu

# TODO: Füge eine Dropoutlayer layer hinzu mit bspw. activation=tf.nn.relu

# TODO: Was fehlt hier, damit es auf das dense ausgabe layer passt?

# TODO: Was kann noch verbessert werden?

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'] )

# Tensorboard
# Load with tensorboard --logdir ./Graph
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

#Modeltrainieren
model.fit(x_train,y_train, epochs=3, callbacks=[tbCallBack])

# Auswerten

val_loss, val_acc = model.evaluate(x_test, y_test)
print("")
print("val_loss:", val_loss)
print("val_acc:", val_acc)