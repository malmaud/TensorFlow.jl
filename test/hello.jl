using TensorFlow
using Base.Test

# Try your first TensorFlow program
# https://github.com/tensorflow/tensorflow#try-your-first-tensorflow-program

hello = TensorFlow.constant("Hello, TensorFlow!")
@test isa(hello, TensorFlow.Tensor)

sess = TensorFlow.Session()
result = run(sess, hello)
@test isa(result, String)
@test "Hello, TensorFlow!" == result

a = TensorFlow.constant(10)
b = TensorFlow.constant(32)
result = run(sess, a+b)
@test isa(result, Int)
@test 42 == result

#basic tensor facts
tens = TensorFlow.ones((5,5))
@test size(tens) == (5,5)
@test length(tens) == 25


@test fill(9, (2, 3)) == run(sess, fill(Tensor, 9, [2, 3]))
@test [2 1; 4 3] == run(sess, reverse(constant([1 2; 3 4]), constant([false, true])))

let
  x = placeholder(Float32, shape = [1,28,28,1])
  y = nn.conv2d(x, Variable(randn(Float32, 5,5,1,32)), [1, 1, 1, 1], "VALID")
  @test get.(get_shape(y).dims) == [1,23,23,32]
  z = nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
  @test get.(get_shape(z).dims) == [1, 12, 12, 32]
end

let
  x = placeholder(Float32, shape = [1,28,28,1])
  sh = TensorFlow.shape(x)
  @test get.(get_shape(sh).dims) == [4]
end
