from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
#print("\n\nhihi\n\n")
# Try to make a dML MODEL to predict an outcome from an input

#training data
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))


#create model and compile( ? 0.1: learning rate, adam is optimizer)
io = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([io])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

#train(epoch:how many times?, verbose:how much)
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training")

#visualize
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#use model for backtesting
print(model.predict([100.0]))#211.7406 near 212
print(model.predict([45.0]))#112.8290 near 113

#layer weight
print("These are the layer variables: {}".format(io.get_weights()))

