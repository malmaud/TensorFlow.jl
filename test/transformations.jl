using TensorFlow
using Base.Test


sess = TensorFlow.Session(TensorFlow.Graph())

one_tens = ones(Tensor, (5,5))

@test [1, 2] == run(sess, cast(constant([1.8, 2.2]), Int))
@test ones(25) == run(sess, reshape(one_tens, 25))
@test 2 == run(sess, rank(one_tens))
@test ones(10,5) == run(sess, tile(one_tens, [2; 1]))
@test hcat(ones(Float32, 5,5), zeros(Float32, 5)) == run(sess, pad(one_tens, [0 0; 0 1]))
@test Float32[1.; 0.; 0.; 0.; 0.] == run(sess, one_hot(1, 5))


@testset "Stack/Unstack" begin
    @test Int32[5,5,1] == run(sess, TensorFlow.shape(stack(split(2, 5, one_tens), axis=1)))

    @test ones(Float32, 5,5) == run(sess, stack(unstack(one_tens, num=5)))
end

@testset "ExpandDims" begin
    @test ones(1,5,5) == run(sess, expand_dims(one_tens, 1))
    @test ones(5,1,5) == run(sess, expand_dims(one_tens, 2))
    @test ones(5,1,5) == run(sess, expand_dims(one_tens, -1))
    @test ones(5,5,1) == run(sess, expand_dims(one_tens, 0))
end



@testset "Permute Dims" begin
    @test ones(Float32, 4,3) == run(sess, transpose(ones(Tensor, (3, 4))))
    @test ones(Float32, 4,3,2) == run(sess, permutedims(ones(Tensor, (4, 2, 3)), [1, 3, 2]))
end


@testset "Shuffle" begin
    a = Tensor(collect(1:5))
    result = run(sess, shuffle(a))
    for i in 1:5
        @test i ∈ result
    end
end


@testset "Squeeze" begin
    # Test `squeeze()` works when given explicit dimensions, fails on incorrect explicit dimensions,
    # and works when given no explicit dimension
    sq_ones = ones(Tensor, (10, 1, 5, 1))
    @test size(run(sess, squeeze(sq_ones))) == (10,5)
    @test size(run(sess, squeeze(sq_ones,[2,4]))) == (10,5)
    @test size(run(sess, squeeze(sq_ones,[2]))) == (10,5,1)
    @test_throws TensorFlow.TFException run(sess, squeeze(sq_ones,[1]))
end



#######################################################################
# getindex related methods (getindex overload and the methods behind it)

# Test values
x_jl = [10x+y for x in 1:5, y in 1:7]
x = constant(x_jl)
w_jl = [100x+10y+z for x in 1:5, y in 1:7, z in 1:3]
w = constant(w_jl)
y_jl = Int32[1,2,3,4]
y = constant(y_jl)

@testset "Mask (bool array)" begin
    mask_jl=[true, false, true,false]
    mask = constant(mask_jl)
    @test run(sess, boolean_mask(y,mask)) == [1, 3]
    @test run(sess, y[mask]) == [1, 3]
    @test run(sess, boolean_mask(y,mask_jl)) == [1, 3]
    @test run(sess, y[mask_jl]) == [1, 3]
end

@testset "Gather (int/ int array) / Index" begin
    @test ones(Float32, 2, 5) == run(sess, gather(one_tens, [1, 2]))
    @test run(sess, y[[1, 3]]) == [1, 3]
    @test run(sess, y[2]) == 2

    @test y_jl[end] == run(sess, y[end])
    @test y_jl[end-1] == run(sess, y[end-1])
    @test y_jl[end-2] == run(sess, y[end-2])
    @test y_jl[end÷2] == run(sess, y[end/2])
    @test y_jl[end-y_jl[1]] == run(sess, y[end-y[1]])

end

@testset "Gather-nd / Cartean Index/Slice" begin
    @test run(sess, gather_nd(x, [2, 3])) == x_jl[2,3]
    @test run(sess, x[2,3]) == x_jl[2,3]

    @test run(sess, gather_nd(x, [3])) == x_jl[3,:]

    @test run(sess, gather_nd(x, [1 1; 2 3])) == [x_jl[1,1], x_jl[2,3]]
    @test run(sess, gather_nd(x, [1 2]')) == [x_jl[1,:]'; x_jl[2,:]']
end

@testset "Slice" begin
    # to do make sure we slice the right indices
    @test ones(Float32, 5).' == run(sess, TensorFlow.slice(one_tens, [1, 1], [1, -1]))

    @test y_jl[2:3] ==  run(sess, y[2:3])
    @test y_jl[2:end] ==  run(sess, y[Int32(2):end])
    @test y_jl[2:end] ==  run(sess, y[2:end])
    @test y_jl[:] ==  run(sess, y[:])

    @test x_jl[2:3, :] ==  run(sess, x[Int32(2):Int32(3), :])
    @test x_jl[2:3, :] ==  run(sess, x[2:3, :])
    @test x_jl[2:end, :] ==  run(sess, x[Int32(2):end, :])

    @test w_jl[:,:,:] ==  run(sess, w[:, :, :])
end


@testset "Mixed slice with index" begin
    @test x_jl[2, :] ==  run(sess, x[2, :])
    @test x_jl[2, :] ==  run(sess, x[constant(2), :])

    @test w_jl[2:4, 3, :] ==  run(sess, w[2:4, 3, :])
    @test w_jl[:, 3, :] ==  run(sess, w[:, 3, :])
    @test w_jl[:, end, :] ==  run(sess, w[:, end, :])
    @test w_jl[1:1, :, :] ==  run(sess, w[1:1, :, :])
end

@testset "Invalid GetIndex" begin
    @test_throws MethodError x[]
    @test_throws MethodError x[1.0:0.5:end]
    @test_throws MethodError x[1f0]
end

@testset "ScatterNd" begin
    @test run(sess, TensorFlow.scatter_nd([2], [6], [4])) == [0, 6, 0, 0]
    @test run(sess, TensorFlow.scatter_nd([5 4 2 8]', [9, 10, 11, 12], [8])) == [0, 11, 0, 10, 9, 0, 0, 12]
    @test run(sess, TensorFlow.scatter_nd([5 3]', [9 9; 10 10], [6,2])) == [0 0; 0 0; 10 10; 0 0; 9 9; 0 0]
end




###################################################################
# Tests after this point must provide their own sessions and graphs

@testset "Concatenation Syntax" begin
    srand(37)
    sess4 = Session(Graph())

    a_jl = rand(10,5); a = constant(a_jl);
    b_jl = rand(10,5); b = constant(b_jl);
    c_jl = rand(5)  ; c = constant(c_jl);
    d_jl = rand(10)  ; d = constant(d_jl);

    s_jl = rand(); s = constant(s_jl);

    @testset "vcat" begin
        @test c_jl == run(sess4, vcat(c))

        @test [c_jl; d_jl] == run(sess4, [c; d])
        @test [c_jl; d_jl; c_jl] == run(sess4, [c; d; c])

        @test [a_jl; b_jl] == run(sess4, [a; b])

        @test [s_jl; s_jl] == run(sess4, [s; s])

        s_jl3 = [s_jl; s_jl; s_jl]
        @test s_jl3 == run(sess4, [s; s; s])
        @test s_jl3 == run(sess4, [s; [s; s]])
        # Promotion
        @test s_jl3 == run(sess4, [s_jl; s; s])
        @test s_jl3 == run(sess4, [s; s_jl; s])
        @test s_jl3 == run(sess4, [s; s; s_jl])

  end

    @testset "hcat" begin
        @test a_jl == run(sess4, hcat(a))
        @test hcat(c_jl) == run(sess4, hcat(c)) #hcat 1D makes it 2D

        @test [c_jl c_jl] == run(sess4, [c c])

        @test [a_jl b_jl] == run(sess4, [a b])
        @test [a_jl d_jl] == run(sess4, [a d])
        @test [a_jl b_jl d_jl] == run(sess4, [a b d])

        @test [s_jl s_jl] == run(sess4, [s s])

         s_jl3 = [s_jl s_jl s_jl]
        @test s_jl3 == run(sess4, [s s s])
        @test s_jl3 == run(sess4, [s [s s]])
        # Promotion
        @test s_jl3 == run(sess4, [s_jl s s])
        @test s_jl3 == run(sess4, [s s_jl s])
        @test s_jl3 == run(sess4, [s s s_jl])

   end

    @testset "nonconcatentating" begin
        @test [s_jl, s_jl] == run(sess4, [s, s])
        @test [a_jl, b_jl, c_jl, d_jl] == run(sess4, [a, b, c, d])

    end

end
