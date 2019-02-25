using TensorFlow
using DifferentialEquations

model = tf.Sequential([tf.Dense(2, 1)])
f(u, p, t) = model(u)
problem = ODEProblem(f, u0=[0.5, 0.5], tspan=(0.0, 1.0))
tf.compile(model, optimizer=tf.Adam(), loss=tf.diffeq_loss(problem, t=[0.0, 0.5, 1.0]))
tf.fit(m, [1.0, 2.0, 5.0], n_epochs=100)
