using TensorFlow
using Test

k = placeholder(Float32; shape=[10, 20, -1])
m = placeholder(Float32; shape=[10, 20, 30])
n = placeholder(Float32)
i = placeholder(Int32; shape=[])


@test get_shape(k) == TensorShape([10, 20, missing])
@test get_shape(m) == TensorShape([10, 20, 30])
@test get_shape(n) == TensorShape(missing)
@test get_shape(i) == TensorShape([])

@test get_shape(k,2) == 20
@test_throws ErrorException get_shape(k, 4)
@test_throws ErrorException get_shape(n, 1)


# https://github.com/malmaud/TensorFlow.jl/issues/466
@test get_shape(TensorFlow.Ops.no_op()) == TensorShape([])
@test get_shape(TensorFlow.group(Tensor(1))) ==  TensorShape([])
