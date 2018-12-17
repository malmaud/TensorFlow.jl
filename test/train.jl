using TensorFlow
using Test

@testset "save and resore" begin
    try
        let
            session = Session(Graph())
            x = get_variable("x", [], Float32)
            run(session, assign(x, 5.f0))
            saver = train.Saver()
            train.save(saver, session, "weights.jld")
        end

        let
            session = Session(Graph())
            @tf x = get_variable([], Float32)
            saver = train.Saver()
            train.restore(saver, session, "weights.jld")
            @test run(session, x) == 5.0f0
        end
   finally
        rm("weights.jld"; force=true)
   end
end


@testset "save and restore max_to_keep" begin
    try
        let
            session = Session(Graph())
            x = get_variable("x", [], Float32)
            run(session, assign(x, 5.f0))
            saver = train.Saver(max_to_keep=5)
            for i in 1:12
                train.save(saver, session, "weights.jld", global_step=i)
            end
        end

        let
            session = Session(Graph())
            @tf x = get_variable([], Float32)
            saver = train.Saver()
            for i in 1:7
                @test_throws SystemError train.restore(saver, session, "weights.jld-$i")
            end
            for i in 8:12
                train.restore(saver, session, "weights.jld-$i")
                @test run(session, x) == 5.0f0
            end
        end
    finally
        for i in 1:12
            rm("weights.jld-$i"; force=true)
        end
    end
end


@testset "metagraph importing and exporting" begin
    mktempdir() do tmppath
        modelfile = joinpath(tmppath, "my_model")
        let
            session = Session(Graph())
            x = constant(1)
            @tf y = x+1
            z = Variable(3, name="z")
            train.export_meta_graph(modelfile)
        end

        let
            session = Session(Graph())
            train.import_meta_graph(modelfile)
            y = get_tensor_by_name("y")
            @test run(session, y) == 2
            run(session, global_variables_initializer())
            z = get_tensor_by_name("z")
            @test run(session, z) == 3
        end
    end
end


@test "optimizers" begin
    using Distributions
    # Generate some synthetic data
    x = randn(100, 50)
    w = randn(50, 10)
    y_prob = exp.(x*w)
    y_prob ./= sum(y_prob,dims=2)

    function draw(probs)
        y = zeros(size(probs))
        for i in 1:size(probs, 1)
            idx = rand(Categorical(probs[i, :]))
            y[i, idx] = 1
        end
        return y
    end

    y = draw(y_prob)

    # Build the model
    sess = Session(Graph())

    X = placeholder(Float64, shape=[-1, 50])
    Y_obs = placeholder(Float64, shape=[-1, 10])

    variable_scope("logisitic_model"; initializer=Normal(0, .001)) do
        global W = get_variable("W", [50, 10], Float64)
        global B = get_variable("B", [10], Float64)
    end

    Y=nn.softmax(X*W + B)

    


    Loss = -reduce_sum(log(Y).*Y_obs)
    
    ### NadamOptimizer
    optimizer = train.NadamOptimizer()
    minimize_op = train.minimize(optimizer, Loss)
    # Run training
    run(sess, global_variables_initializer())
    for epoch in 1:100
        cur_loss, _ = run(sess, [Loss, minimize_op], Dict(X=>x, Y_obs=>y))
        println(@sprintf("[NadamOptimizer]Current loss is %.2f.", cur_loss))
    end

    ### AMSGradOptimizer
    optimizer = train.AMSGradOptimizer()
    minimize_op = train.minimize(optimizer, Loss)
    # Run training
    run(sess, global_variables_initializer())
    for epoch in 1:100
        cur_loss, _ = run(sess, [Loss, minimize_op], Dict(X=>x, Y_obs=>y))
        println(@sprintf("[AMSGradOptimizer]Current loss is %.2f.", cur_loss))
    end

    function mycallback(handle)
        res = run(sess, Loss, Dict(X=>x, Y_obs=>y))
        println("[$m]iter \$(handle.iteration): \$(res)")
        return false # so it do not stop 
    end

    for m in ["AGD", "CG", "BFGS", "LBFGS"]
        run(sess, global_variables_initializer())
        options = Optim.Options(show_trace = false, iterations=1000, callback = mycallback, allow_f_increases=true)
        OptimMinimize(sess, Loss, feed_dict = Dict(X=>x, Y_obs=>y), options=options, method=m)
    end

end