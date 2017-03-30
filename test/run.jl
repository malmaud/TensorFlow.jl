using Base.Test
using TensorFlow

@testset "Placeholder Size Matching" begin
    sess = Session(Graph())
    k = placeholder(Float32; shape=[10, 20, 30])
    m = placeholder(Float32; shape=[10, 20, -1])
    n = placeholder(Float32)
    i = placeholder(Int32; shape=[])

    kk=2k
    mm=2m
    nn=2n
    ii=2i


    #Should work fine:
    run(sess, kk, Dict(k=>ones(10, 20, 30)))

    run(sess, mm, Dict(m=>ones(10, 20, 30)))
    run(sess, mm, Dict(m=>ones(10, 20, 50)))

    run(sess, nn, Dict(n=>ones(10, 20, 30)))
    run(sess, nn, Dict(n=>ones(10, 20)))
    run(sess, nn, Dict(n=>1f0))

    run(sess, ii, Dict(i=>1))

    #Should not be allowed
    @test_throws DimensionMismatch run(sess, kk, Dict(k=>ones(10, 20, 31)))
    @test_throws DimensionMismatch run(sess, kk, Dict(k=>ones(10, 20)))

    @test_throws DimensionMismatch run(sess, mm, Dict(m=>ones(10, 21, 30)))
    @test_throws DimensionMismatch run(sess, mm, Dict(m=>ones(10, 20)))

    @test_throws DimensionMismatch run(sess, ii, Dict(i=>[1,2]))
end
