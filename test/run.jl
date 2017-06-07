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
    nn_=2n  # To not conflict with TensorFlow.nn during debugging
    ii=2i


    #Should work fine:
    run(sess, kk, Dict(k=>ones(10, 20, 30)))

    run(sess, mm, Dict(m=>ones(10, 20, 30)))
    run(sess, mm, Dict(m=>ones(10, 20, 50)))

    run(sess, nn_, Dict(n=>ones(10, 20, 30)))
    run(sess, nn_, Dict(n=>ones(10, 20)))
    run(sess, nn_, Dict(n=>1f0))

    run(sess, ii, Dict(i=>1))

    #Should not be allowed
    @test_throws DimensionMismatch run(sess, kk, Dict(k=>ones(10, 20, 31)))
    @test_throws DimensionMismatch run(sess, kk, Dict(k=>ones(10, 20)))

    @test_throws DimensionMismatch run(sess, mm, Dict(m=>ones(10, 21, 30)))
    @test_throws DimensionMismatch run(sess, mm, Dict(m=>ones(10, 20)))

    @test_throws DimensionMismatch run(sess, ii, Dict(i=>[1,2]))
end

@testset begin
    srand(1)
    data = rand(Int64.(1:10), 3,4)
    sess = Session(Graph())

    x = placeholder(Int32)
    y = Int32(2)*x

    # Should work fine
    run(sess, y, Dict(x=>data))
    run(sess, y, Dict(x=>@view data[1:2,:]))

    data32 = Int32.(data)

    run(sess, y, Dict(x=>data32))
    run(sess, y, Dict(x=>@view data32[1:2, :]))

end
