import tensorflow as tf
 
# Load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize your data
x_train = tf.keras.utils.normalize(x_train, 1)
x_test = tf.keras.utils.normalize(x_test, 1)

# Reshabe to fit into Convolutional Layers
CHANNELS = 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, CHANNELS)
x_test = x_test.reshape(x_test.shape[0], 28, 28, CHANNELS)
 
# Build Model

model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Conv2D(28, (4, 4), activation='relu', input_shape=(28, 28, CHANNELS)))

# TODO: Add a max-pooling layer

# TODO: Add additional Conv2D layers activation=tf.nn.relu

# TODO: Add a dropout layer. activation=tf.nn.relu

# TODO: Whats needed to get a Conv2D Output to a fit to the output layer?

# TODO: What could be added here additionally?

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

# Tensorboard
# Load with tensorboard --logdir ./Graph
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

#Train your model
model.fit(x_train,y_train, epochs=3, callbacks=[tbCallBack])

# evaluate
val_loss, val_acc = model.evaluate(x_test, y_test)
print("")
print("val_loss:", val_loss)
print("val_acc:", val_acc)
