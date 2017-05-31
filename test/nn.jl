using TensorFlow
using Base.Test


@testset "conv2d_transpose" begin
    let
        sess = Session(Graph())
        value = placeholder(Float32, shape=[32, 10, 10, 3])
        filter = placeholder(Float32, shape=[3, 3, 5, 3])
        shape_ = placeholder(Int32, shape=[4])
        y = nn.conv2d_transpose(value, filter, shape_, [1, 1, 1, 1])
        run(sess, y, Dict(value=>randn(Float32, 32, 10, 10, 3),
                          filter=>randn(Float32, 3, 3, 5, 3),
                          shape_=>[32,10,10,5]))
    end
end


@testset "Cross Entropy Loss" begin
    srand(1)
    let
        sess = Session(Graph())
        targets = constant(collect(1:10))
        targets_hot = constant(Float64.(eye(10)))
        logits_unscaled = constant(rand(10,10))


        logits = nn.softmax(logits_unscaled)
        loss_direct = -reduce_sum(log(logits).*targets_hot; axis=1)
        loss_sparse = nn.sparse_softmax_cross_entropy_with_logits(logits=logits_unscaled, labels=targets)
        loss_nonsparse = nn.softmax_cross_entropy_with_logits(logits=logits_unscaled, labels=targets_hot)
        res_direct, res_sparse, res_nonsparse =  run(sess, [loss_direct, loss_sparse, loss_nonsparse])

        @test res_direct ≈ res_sparse
        @test res_direct ≈ res_nonsparse
    end
end

@testset "rnn_cell zero state" begin
    let
        sess = Session(Graph())
        
        x = placeholder(Float32, shape=[-1, 1, 5])
        x1 = x[:,1,:]
        gru_state = nn.rnn_cell.zero_state(nn.rnn_cell.GRUCell(7), x1, nothing)
        lstm_state = nn.rnn_cell.zero_state(nn.rnn_cell.LSTMCell(7), x1, nothing)

        @test eltype(lstm_state) == Float32

        gru_state_jl, lstm_state_jl = run(sess, [gru_state, lstm_state], Dict(x=>rand(19, 1, 5)))

        @test size(gru_state_jl) == (19, 7) #batchsize, hidden_size
        @test all(gru_state_jl .== 0f0)
        @test size(lstm_state_jl.c) == (19, 7) #batchsize, hidden_size
        @test all(lstm_state_jl.c .== 0f0)
        @test size(lstm_state_jl.h) == (19, 7) #batchsize, hidden_size
        @test all(lstm_state_jl.h .== 0f0)
    end
end


for (rnn_fun, post_proc_outputs) in ((nn.dynamic_rnn, identity), (nn.rnn, last))
    testname = split(string(rnn_fun), ".")[end]

     @testset "$testname len 1" begin
        let
            sess = Session(Graph())
            data = constant(ones(1, 1, 1))
            cell = nn.rnn_cell.BasicRNNCell(1)
            local y
            variable_scope("rnn", initializer=ConstantInitializer(.1)) do
                y = rnn_fun(cell, data)
            end

            run(sess, global_variables_initializer())
            outputs = run(sess, y)[1]
            output = post_proc_outputs(outputs)

            expected_output = tanh(1*.1+.1)
            @test output[1,1] ≈ expected_output
        end
    end

    @testset "$testname len2" begin
        let
            sess = Session(Graph())
            data = constant(ones(1, 2, 1))
            cell = nn.rnn_cell.BasicRNNCell(1)
            s0 = nn.zero_state(cell, data[:,1,:], Float64)
            local y
            variable_scope("rnn", initializer=ConstantInitializer(.1)) do
                y = rnn_fun(cell, data, initial_state=s0)
            end


            run(sess, global_variables_initializer())
            outputs = run(sess, y)[1]
            output = post_proc_outputs(outputs)
            expected_output = tanh(1*.1+tanh(1*.1+.1)*.1+.1)
            @test output[1,1] ≈ expected_output

        end
    end


    @testset "$testname sequence_length" begin
        let
            sess = Session(Graph())
            data_jl = Float32[100*x+10y+z for x in 1:10, y in 1:20, z in 1:5]
            data = constant(data_jl)
            lens_jl = collect(1:2:20) #1 for each element in the batch (x) saying how far to go down the time (y)
            lens = constant(lens_jl)
            cell = nn.rnn_cell.IdentityRNNCell(5)
            y, s_last = rnn_fun(cell, data, lens; dtype=Float32)

            run(sess, global_variables_initializer())
            outputs = run(sess, y)
            output = post_proc_outputs(outputs)
            @test output == [data_jl[xi, lens_jl[xi], zi] for xi in 1:10, zi in 1:5]
        end
    end

    @testset "$testname sequence_length time-major" begin
        let
            sess = Session(Graph())
            data_jl = Float32[100*x+10y+z for x in 1:20, y in 1:10, z in 1:5] #Time first dim, batch second
            data = constant(data_jl)
            lens_jl = collect(1:2:20) #1 for each element in the batch (y) saying how far to go down the time (y)
            lens = constant(lens_jl)
            cell = nn.rnn_cell.IdentityRNNCell(5)
            y, s_last = rnn_fun(cell, data, lens; dtype=Float32, time_major=true)

            run(sess, global_variables_initializer())
            outputs = run(sess, y)
            output = post_proc_outputs(outputs)
            @test output == [data_jl[lens_jl[xi], xi, zi] for xi in 1:10, zi in 1:5]
        end
    end

    @testset "$testname dynamic batch-size" begin
        let
            sess = Session(Graph())
            
            x = placeholder(Float32, shape=[-1, 1, 5])
            cell = nn.rnn_cell.GRUCell(7)
            out, state = rnn_fun(cell, x)
            
            run(sess, global_variables_initializer())
            outs_jl, state_jl = run(sess, [out, state], Dict(x=>rand(19, 1, 5)))
            
            @test size(state_jl) == (19, 7) #batchsize, hidden_size
        end
    end


end

@testset "rnn gradients" begin
    sess = Session(Graph())
    cell = nn.rnn_cell.LSTMCell(10)
    inputs = constant(randn(Float32, 5, 32, 5))
    out = nn.dynamic_rnn(cell, inputs)
    loss = reduce_sum(out[1]).^2
    minimizer = train.GradientDescentOptimizer(.01)
    minimize_op = train.minimize(minimizer, loss)
    run(sess, global_variables_initializer())
    run(sess, minimize_op)
end

@testset "dropout" begin
    sess = Session(Graph())
    inputs = constant(randn(Float32, 5, 32, 5))
    drop_prob = placeholder(Float32)

    dd =  nn.dropout(inputs, drop_prob)  # Check that this can actually be created

end
