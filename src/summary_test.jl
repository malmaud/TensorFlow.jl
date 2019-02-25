tf = TensorFlow
summary = tf.summary
writer = summary.FileWriter("/Users/malmaud/tmp")
summary.set_default(writer)
summary.scalar("x", 3.2, step=0)
summary.scalar("x", 5.0, step=1)
summary.scalar("x", -2.5, step=2)
