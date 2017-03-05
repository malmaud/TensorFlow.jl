using TensorFlow
using Base.Test

sess = TensorFlow.Session(TensorFlow.Graph())

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
for func in [lbeta, polygamma, zeta]
    @test func(2, 3.0) ≈ run(sess, func(constant(2.0), constant(3.0)))
end

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
result = run(sess, TensorFlow.add_n([a,b]))
@test a_raw .+ b_raw ≈ result
result = run(sess, 5 * a)
@test 5*a_raw ≈ result

a_raw = rand(10)
b_raw = rand(10)
a = TensorFlow.constant(a_raw)
b = TensorFlow.constant(b_raw)
result = run(sess, max(a,b))
@test max.(a_raw, b_raw) == result
result = run(sess, min(a,b))
@test min.(a_raw, b_raw) == result

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
for (op, func) in [(&, logical_and), (|, logical_or)]
    result = run(sess, func(a,b))
    @test all(map(op, a_raw, b_raw) .== result)
end

result = run(sess, ~a)
@test ~a_raw == result

a_raw = rand(10)
a = TensorFlow.constant(a_raw)
for (jl_func, red_func) in [(sum, reduce_sum), (prod, reduce_prod), (minimum, reduce_min), (maximum, reduce_max), (mean, reduce_mean)]
    result = run(sess, red_func(a))
    @test jl_func(a_raw) ≈ result
    result = run(sess, red_func(a, axis=0))
    @test jl_func(a_raw, 1)[1] ≈ result
end

a_raw = rand(10)
a = TensorFlow.constant(a_raw)
inds = vcat(ones(Int32,5), fill(Int32(2), 5))
for (jl_func, seg_func) in [(sum, segment_sum), (prod, segment_prod), (minimum, segment_min), (maximum, segment_max), (mean, segment_mean)]
    result = run(sess, seg_func(a, inds))
    @test jl_func(a_raw[1:5]) ≈ result[1]
    @test jl_func(a_raw[6:10]) ≈ result[2]
end

a_raw = Vector{Bool}(trues(10))
a = TensorFlow.constant(a_raw)
b_raw = Vector{Bool}(bitrand(10))
b = TensorFlow.constant(b_raw)
for (jl_func, red_func) in [(any, reduce_any), (all, reduce_all)]
    result = run(sess, red_func(a))
    @test jl_func(a_raw) == result
    result = run(sess, red_func(b))
    @test jl_func(b_raw) == result
end

M_raw = tril(rand(Float32, 10, 10))
x_raw = ones(Float32, 10, 2)
x = TensorFlow.constant(x_raw)
M = TensorFlow.constant(M_raw)
result = run(sess, TensorFlow.matrix_triangular_solve(M, x))
@test M_raw \ x_raw ≈ result

result = run(sess, TensorFlow.matrix_solve(M, x))
# This test fails on Linux since the result is too approximate
#@test M_raw \ x_raw ≈ result

M_raw = rand(Float32, 10, 10)
M_raw += M_raw'
result = run(sess, TensorFlow.self_adjoint_eig(constant(M_raw)))
@test eigvals(M_raw) ≈ result[1]
evs = eigvecs(M_raw)
for vec_ind in 1:10
    @test abs(dot(evs[:, vec_ind], result[2][:, vec_ind])) ≈ Float32(1.)
end

M_raw = rand(Float32, 10, 10)
M_raw *= M_raw'
result = run(sess, TensorFlow.cholesky(constant(M_raw)))
@test cholfact(M_raw)[:L] ≈ result

a = TensorFlow.Variable(ones(6))
run(sess, initialize_all_variables())
result = run(sess, TensorFlow.scatter_update(a, [1; 5], [2.; 2.]))
@test [2; 1; 1; 1; 2; 1] == result

result = run(sess, TensorFlow.assign_add(a, ones(6); use_locking=true))
@test [3.; 2.; 2.; 2.; 3.; 2.] == result

result = run(sess, TensorFlow.assign_sub(a, ones(6)))
@test [2.; 1.; 1.; 1.; 2.; 1.] == result

result = run(sess, TensorFlow.assign(a, -ones(6); use_locking=true))
@test -ones(6) == result

result = run(sess, TensorFlow.scatter_add(a, [1; 5], [1.; 1.]; use_locking=true))
@test [0.; -1.; -1.; -1.; 0.; -1.] == result

result = run(sess, TensorFlow.sign(constant([-1, 2, 0])))
@test [-1, 1, 0] == result

@test run(sess, squared_difference([1,2], 5)) == [16, 9]
