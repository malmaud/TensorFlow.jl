using Base.Test
using TensorFlow

@testset "gather" begin
    let
        sess = Session(Graph())
        x = get_variable("x", (50,10), Float64)
        x3n5 = gather(x, [3, 5])
        cost = reduce_sum(x3n5)
        optimizer = train.minimize(train.AdamOptimizer(0.1), cost)
        run(sess, global_variables_initializer())
        @test size(run(sess, x3n5)) == (2, 10)
    end
end

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
