using TensorFlow
using Base.Test

sess = TensorFlow.Session(TensorFlow.Graph())

@test [1, 2] == run(sess, cast(constant([1.8, 2.2]), Int))

one_tens = ones(Tensor, (5,5))

@test ones(25) == run(sess, reshape(one_tens, 25))

@test ones(Float32, 5).' == run(sess, slice(one_tens, [0, 0], [1, -1]))

@test Int32[5,5,1] == run(sess, TensorFlow.shape(pack(split(2, 5, one_tens), axis=1)))

@test ones(5,5,1) == run(sess, expand_dims(one_tens, 2))

@test 2 == run(sess, rank(one_tens))

@test ones(10,5) == run(sess, tile(one_tens, [2; 1]))

@test ones(Float32, 4,3) == run(sess, transpose(ones(Tensor, (3, 4))))
