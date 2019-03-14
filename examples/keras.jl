using TensorFlow
tf=TensorFlow
tf.enable_eager_execution()
m = tf.Sequential()

tf.add(m, tf.Dense(3,10))
tf.add(m, tf.Relu())
tf.add(m, tf.Dense(10, 3))

x=constant(randn(5,3))
y=3x+5
tf.compile(m, optimizer=tf.SGD(lr=1e-4), loss=tf.mse)
tf.fit(m, x, y, n_epochs=200)
