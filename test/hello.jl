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

#basic tensor facts
tens = TensorFlow.ones((5,5))
@test size(tens) == (5,5)
@test length(tens) == 25

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


mat_raw = rand(10, 10)
mat = TensorFlow.constant(mat_raw)
result = run(sess, inv(mat))
@test inv(mat_raw) ≈ result
result = run(sess, det(mat))
@test det(mat_raw) ≈ result
result = run(sess, diag(mat))
@test diag(mat_raw) ≈ result

diag_raw = rand(5)
diag_mat = TensorFlow.constant(diag_raw)
result = run(sess, det(diagm(diag_mat)))
@test prod(diag_raw) ≈ result

a_raw = rand(3)
b_raw = rand(3)
a = TensorFlow.constant(a_raw)
b = TensorFlow.constant(b_raw)
result = run(sess, a+b)
@test a_raw + b_raw ≈ result
result = run(sess, a-b)
@test a_raw - b_raw ≈ result
result = run(sess, a.*b)
@test a_raw.*b_raw ≈ result
result = run(sess, a × b)
@test a_raw × b_raw ≈ result

a_raw = rand(10)
b_raw = rand(10)
a = TensorFlow.constant(a_raw)
b = TensorFlow.constant(b_raw)
result = run(sess, max(a,b))
@test max(a_raw, b_raw) == result
result = run(sess, min(a,b))
@test min(a_raw, b_raw) == result

a_raw = rand(10)
a = TensorFlow.constant(a_raw)
result = run(sess, indmin(a, 1))
@test indmin(a_raw) == result+1
result = run(sess, indmax(a, 1))
@test indmax(a_raw) == result+1

a_raw = Vector{Bool}(bitrand(10))
b_raw = Vector{Bool}(bitrand(10))
a = TensorFlow.constant(a_raw)
b = TensorFlow.constant(b_raw)
for (op, func) in [(&, logical_and), (|, logical_or), (^, logical_xor)]
    result = run(sess, func(a,b))
    @test op(a_raw, b_raw) == result
end
result = run(sess, logical_not(a))
@test ~a_raw == result
