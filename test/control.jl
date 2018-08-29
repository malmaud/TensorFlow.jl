using Test
using TensorFlow
import LinearAlgebra

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
    result = run(sess, LinearAlgebra.cond(x<y, f1, f2))
    @test 17*2 == result
    @test_broken begin
        inc = constant(1)
        i = constant(1)
        w = TensorFlow.while_loop((i,s)->i≤5, (i,s)->[i+inc, s+i], [i, 0])
        @test run(sess, w)[2] == sum(1:5)
        grad = gradients(w[1], i)
        @test run(sess, grad) == 1
    end
end
