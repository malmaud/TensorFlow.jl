sess = Session(Graph())

first = TensorFlow.constant(collect(1:16))
second = run(sess, TensorFlow.identity(first))
@test collect(1:16) == second
third = run(sess, TensorFlow.make_tuple([TensorFlow.constant(collect(1:16)), TensorFlow.constant(collect(1:16))]))
@test [collect(1:16), collect(1:16)] == third
