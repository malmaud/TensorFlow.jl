using Test
using TensorFlow

@testset "Training with 'gather' nodes" begin
    sess = Session(Graph())
    x = get_variable("x", (50,10), Float64)
    x3n5 = gather(x, [3, 5])
    cost = reduce_sum(x3n5)
    optimizer = train.minimize(train.AdamOptimizer(0.1), cost)
    run(sess, global_variables_initializer())
    @test size(run(sess, x3n5)) == (2, 10)
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

@testset "Making a functioning network: Issue #147" begin
    # concat
    let
        sess3 = Session(Graph())
        x1 = constant(rand(20,10))
        x2 = get_variable("x2", (50,10), Float64)
        xs = concat([x1,x2], 1)
        cost = reduce_sum(xs)
        optimizer = train.minimize(train.AdamOptimizer(0.1), cost)
        run(sess3, global_variables_initializer())
        @test size(run(sess3, xs)) == (70, 10)
    end

    # gather_nd
    let
        sess2 = Session(Graph())
        embs = get_variable("tt2", (10,10), Float64)
        vals = gather_nd(embs,[2])
        cost = reduce_sum(vals)
        optimizer = train.minimize(train.AdamOptimizer(0.1), cost)
        run(sess2, global_variables_initializer())
        @test length(run(sess2, vals)) == 10
    end

end
