using TensorFlow
using Test

sess = Session(Graph())

@test run(sess, ones(Tensor, 2, 3)) == ones(2, 3)
@test run(sess, zeros(Tensor{Float64}, 2, 3)) == zeros(2, 3)

@testset "random" begin
    for dtype in [Float32, Float64]
        shape = [3, 5]
        x = random_normal(shape, dtype=dtype)
        @test get_shape(x) == TensorShape(shape)
        result = run(sess, x)
        @test size(result) == (shape...,)
        @test eltype(result) == dtype
    end
end
