using Base.Test
using TensorFlow


@testset "identity and make_tuple" begin
    sess = Session(Graph())
    first = constant(collect(1:16))
    second = run(sess, identity(first))
    @test collect(1:16) == second
    third = run(sess, TensorFlow.make_tuple([constant(collect(1:16)), constant(collect(1:16))]))
    @test [collect(1:16), collect(1:16)] == third
end

@testset "cond and while_loop" begin
    sess = Session(Graph())
    x = constant(2)
    y = constant(5)
    f1 = ()->17x
    f2 = ()->y+23
    result = run(sess, cond(x<y, f1, f2))
    @test 17*2 == result

    s=Session(Graph())
    @tf i = constant(1)
    @tf weight = placeholder(Float32)
    w=while_loop((i,res)->i<5, (i,res)->[i+1, res.*weight], [i, 2.0])


    g=gradients(w[2], weight)
end
