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

c = TensorFlow.constant(complex(Float32(3.2), Float32(5.0)))
result_r = run(sess, real(c))
@test Float32(3.2) == result_r
result_i = run(sess, imag(c))
@test Float32(5.) == result_i

d_raw = 1. + rand(10)
d = TensorFlow.constant(d_raw)
for unary_func in [erf, erfc, lgamma, tanh, tan, sin, cos, abs, exp, log]
                   result = run(sess, unary_func(d))
                   @test unary_func.(d_raw) ≈ result
end
result = run(sess, -d)
@test -d_raw == result

f_raw = rand(10)./2
f = TensorFlow.constant(f_raw)
for func in [acos, asin, atan]
    result = run(sess, func(f))
    @test func.(f_raw) ≈ result
end

x_r_raw = Vector{Float32}(rand(10))
x_i_raw = Vector{Float32}(rand(10))
x_r = Tensor(x_r_raw)
x_i = Tensor(x_i_raw)

result = run(sess, complex(x_r, x_i))
@test complex(x_r_raw, x_i_raw) == result
