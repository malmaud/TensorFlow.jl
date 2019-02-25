tf=TensorFlow
m = tf.Sequential()

tf.add(m, tf.Dense(3,10))
tf.add(m, tf.ReluLayer())
tf.add(m, tf.Dense(10, 3))

x=constant(randn(5,3))
y=3x+5
tf.compile(m, optimizer=tf.SGD(lr=1e-3), loss=tf.mse)
tf.fit(m, x, y, n_epochs=1000)
