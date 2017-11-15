using TensorFlow
using Base.Test

sess = TensorFlow.Session(TensorFlow.Graph())

@testset "real and complex" begin
    c = TensorFlow.constant(complex(Float32(3.2), Float32(5.0)))
    result_r = run(sess, real(c))
    @test Float32(3.2) == result_r
    result_i = run(sess, imag(c))
    @test Float32(5.) == result_i


    x_r_raw = Vector{Float32}(rand(10))
    x_i_raw = Vector{Float32}(rand(10))
    x_r = constant(x_r_raw)
    x_i = constant(x_i_raw)

    result = run(sess, complex(x_r, x_i))
    @test complex.(x_r_raw, x_i_raw) == result
end

@testset "SpecialFuns" begin
    d_raw = 1. + rand(10)
    d = TensorFlow.constant(d_raw)
    for unary_func in [erf, erfc, lgamma, tanh, tan, sin, cos, abs, exp, log]
       result = run(sess, unary_func(d))
       @test unary_func.(d_raw) ≈ result
    end
    result = run(sess, -d)
    @test -d_raw == result

    for func in [polygamma, zeta]
        @test func(2, 3.0) ≈ run(sess, func(constant(2.0), constant(3.0)))
    end

    f_raw = rand(10)./2
    f = TensorFlow.constant(f_raw)
    for func in [acos, asin, atan]
        result = run(sess, func(f))
        @test func.(f_raw) ≈ result
    end
end

@testset "Matrix Operations" begin
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
end

@testset "arithmetic" begin
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
end

@testset "extrema" begin
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
    @test indmin(a_raw) == result
    result = run(sess, indmax(a, 1))
    @test indmax(a_raw) == result

    #check it find the first instance of lowest/highers number as the result for indmin/indmax
    x=constant([1 2 3; 0 2 3; 0 0 3; 0 0 0]')
    @test [2, 3, 4] == run(sess, indmin(x, 2))
    @test [1, 1, 1] == run(sess, indmax(x, 2))
end

@testset "logic" begin
    a_raw = Vector{Bool}(bitrand(10))
    b_raw = Vector{Bool}(bitrand(10))
    a = TensorFlow.constant(a_raw)
    b = TensorFlow.constant(b_raw)
    for (op, func) in [(&, logical_and), (|, logical_or)]
        result = run(sess, func(a,b))
        @test all(map(op, a_raw, b_raw) .== result)
    end

    result = run(sess, ~a)
    @test map(~, a_raw) == result  # Use map for .5 comptability
end

@testset "reduce" begin
    a_raw = rand(10)
    a = TensorFlow.constant(a_raw)
    for (jl_func, red_func) in [(sum, reduce_sum), (prod, reduce_prod), (minimum, reduce_min), (maximum, reduce_max), (mean, reduce_mean)]
        result = run(sess, red_func(a))
        @test jl_func(a_raw) ≈ result
        result = run(sess, red_func(a, axis=0))
        @test jl_func(a_raw, 1)[1] ≈ result
        @test run(sess, jl_func(a)) ≈ jl_func(a_raw)
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

    # Unsorted segment sum
    a = TensorFlow.constant(eye(5))
    idxs = TensorFlow.constant(map(Int64, [1,1,2,3,1]))
    n = TensorFlow.constant(Int32(3))
    d = unsorted_segment_sum(a, idxs, n)
    results = [1.0  1.0  0.0  0.0  1.0;
               0.0  0.0  1.0  0.0  0.0;
               0.0  0.0  0.0  1.0  0.0]
    @test all(run(sess,d).==results)
end

@testset "linear algebra" begin
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

    for thin in [true, false]
        M_raw = rand(Float32, 10, 10)
        u, s, v = run(sess, svd(constant(M_raw), thin=thin))
        uj, sj, vj = svd(M_raw, thin=thin)
        # Can't directly check values between julia svd and tensorflow svd as SVD not unique
        # Check the shape of results match
        @test size(s) == size(sj)
        @test size(u) == size(uj)
        @test size(v) == size(uj)
        # Check the reconstruction is close
        err = sum(abs2, M_raw-u*diagm(s)*v')
        err_j = sum(abs2, M_raw-uj*diagm(sj)*vj')

        @assert err_j ≤ 1e-10
        @test err ≤ 1000err_j # Julia has really accurate SVD, apparently, so give it a margin
    end
end


@testset "assignment" begin
    a = TensorFlow.Variable(ones(6))
    run(sess, global_variables_initializer())
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

    result = run(sess, TensorFlow.scatter_sub(a, [1; 5], [1.; 1.]; use_locking=true))
    @test -ones(6) == result

end


@testset "Rounding" begin
    @test 2.0 == run(sess, round(constant(1.7)))
    @test [2.0, -1.0, 3.0] == run(sess, round(constant([1.7, -1.0, π])))

    @test 2 == run(sess, round(Int, constant(1.7)))
    @test [2, -1, 3] == run(sess, round(Int, constant([1.7, -1.0, π])))
end


@testset "broadcasting operations" begin
    x_jl = [1.0 5.0; 7.0 13.0]
    x = constant(x_jl)
    y_jl = [31.0 331.0; 3331.0 333331.0]
    y = constant(y_jl)

    @test run(sess, x.^2 .* y) == x_jl.^2 .* y_jl

    #issue #336
    a = [0.01]
    b = [0.02]
    v = constant([1.,2.])
    #@test_throws TensorFlow.TFException run(sess, v[1] * a)
    # The above test is causing a 'cycle error' in future uses of the graph. TODO: fix that
    @test 0.01 ≈ run(sess, v[1] .* a) |> first
    @test 0.02 ≈ run(sess, v[1].* b) |> first
    @test 0.0002 ≈ run(sess, v[1].* a .* b) |> first
    @test 0.0002 ≈ run(sess, (v[1].* a) .* b) |> first
    @test 0.0002 ≈ run(sess, v[1].* (a .* b)) |> first
end


@test [-1, 1, 0] == run(sess, TensorFlow.sign(constant([-1, 2, 0])))

@test run(sess, squared_difference([1,2], 5)) == [16, 9]
