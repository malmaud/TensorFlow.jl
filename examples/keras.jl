m = tf.Model()
layer = tf.dense(3,3)
tf.add(m, layer)

x=constant(randn(5,3))
y=3x
tf.compile(m, optimizer=.01, loss=tf.mse)
tf.fit(m, x, y, n_epochs=100)
