using Base.Test
using TensorFlow

sess = Session(Graph())
x = get_variable("x", (50,10), Float64)
x3n5 = gather(x, [3, 5])
cost = reduce_sum(x3n5)
optimizer = train.minimize(train.AdamOptimizer(0.1), cost)
run(sess, global_variables_initializer())
@test size(run(sess, x3n5)) == (2, 10)
