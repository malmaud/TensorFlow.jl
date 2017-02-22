using Base.Test

sess = Session(Graph())

first = TensorFlow.constant(collect(1:16))
second = run(sess, TensorFlow.identity(first))
@test collect(1:16) == second
third = run(sess, TensorFlow.make_tuple([TensorFlow.constant(collect(1:16)), TensorFlow.constant(collect(1:16))]))
@test [collect(1:16), collect(1:16)] == third

x = TensorFlow.constant(2)
y = TensorFlow.constant(5)
f1 = ()->17x
f2 = ()->y+23
result = run(sess, TensorFlow.cond(x.<y, f1, f2))
@test 17*2 == result
inc = constant(1)
i = constant(1)
w = while_loop((i,s)->i≤5, (i,s)->[i+inc, s+i], [i, 0])
@test run(sess, w)[2] == sum(1:5)
grad = gradients(w[1], i)
@test run(sess, grad) == 1
