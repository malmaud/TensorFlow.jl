using TensorFlow
using Base.Test

sess = Session(Graph())

@test run(sess, ones(Tensor, 2, 3)) == ones(2, 3)
@test run(sess, zeros(Tensor{Float64}, 2, 3)) == zeros(2, 3)
