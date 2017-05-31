using Base.Test
using TensorFlow


@testset "slice" begin
    let
        sess = Session(Graph())
        x = get_variable("x", (50,10), Float64)
        x1 = TensorFlow.slice(x, [1,1],[-1,1])
        cost = reduce_sum(x1)
        optimizer = train.minimize(train.AdamOptimizer(), cost)
        run(sess, global_variables_initializer())
        @test size(run(sess, x1)) == (50, 1)
    end
end
