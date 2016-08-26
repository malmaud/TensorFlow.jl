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
