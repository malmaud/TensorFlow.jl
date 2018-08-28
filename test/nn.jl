using TensorFlow
using Test
using StatsFuns
using Random
import LinearAlgebra

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
    Random.seed!(1)
    let
        sess = Session(Graph())
        targets = constant(collect(1:10))
        #targets_hot = constant(Float64.(LinearAlgebra.eye(10)))
        targets_hot = constant(Matrix{Float64}(LinearAlgebra.I, 10, 10))
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


#for (rnn_fun, post_proc_outputs) in ((nn.dynamic_rnn, identity), (nn.rnn, last))
for (rnn_fun, post_proc_outputs) in ((nn.rnn, last),)
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

            @test size(outs_jl) == (1,) # time steps in x
            @test size(outs_jl[1]) == (19, 7) #batchsize, hidden_size
            @test size(state_jl) == (19, 7) #batchsize, hidden_size
        end
    end

    @testset "$testname dynamic length" begin
        hiddenstates(x) = (x, )
        hiddenstates(x::nn.rnn_cell.LSTMStateTuple) = (x.h, x.c)

        @testset "$celltype" for celltype in (nn.rnn_cell.GRUCell, nn.rnn_cell.LSTMCell)
            let
                sess = Session(Graph())

                batch_size = 4
                hidden_size = 7
                time_steps = 3
                input_dim = 5

                x = placeholder(Float32, shape=[-1, time_steps, input_dim])
                xlen = placeholder(Float32, shape=[-1])
                cell = celltype(hidden_size)
                out, state = rnn_fun(cell, x, xlen)

                run(sess, global_variables_initializer())

                xdata = ones(batch_size, time_steps, input_dim)
                xlendata = [1, 2, 3, 2]

                outs_jl, state_jl = run(sess, [out, state], Dict(x=>xdata, xlen=>xlendata))

                @test size(outs_jl) == (time_steps,)
                @test all(size.(outs_jl) .== [(batch_size, hidden_size)])

                # the output from the first sequence is repeated 3 times since the length is 1
                # the second output from the second and fourth sequence is repeated twice
                # the third sequence has some new output from each of the 3 time steps
                @test all(outs_jl[1] .== outs_jl[2], dims=2) == [true false false false]'
                @test all(outs_jl[2] .== outs_jl[3], dims=2) == [true true false true]'

                # since xdata is the same for all sequences the hidden state will be the same
                # if the sequences have equal length. Sequence number 2 and 4 are both of length 2.
                for s in hiddenstates(state_jl)
                    @test s[1,:] != s[2,:]
                    @test s[1,:] != s[3,:]
                    @test s[2,:] != s[3,:]
                    @test s[2,:] == s[4,:]
                end
            end
        end
    end
end

@testset "rnn gradients" begin
    @test_broken begin
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
end


@testset "LSTMcell biases" begin
    sess = Session(Graph())
    inputs = constant(randn(Float32, 5, 32, 5))

    cell1 = nn.rnn_cell.LSTMCell(10)
    out1 = nn.rnn(cell1, inputs; scope="RNN1")

    cell2 = nn.rnn_cell.LSTMCell(10; forget_bias=2f0)
    out2 = nn.rnn(cell2, inputs; scope="RNN2")

    run(sess, global_variables_initializer())

    #default should be 1
    run(sess, sess.graph["RNN1/Bias/Bf"])
    @test run(sess, sess.graph["RNN1/Bias/Bf"]) ≈ ones(10)

    run(sess, sess.graph["RNN2/Bias/Bf"])
    @test run(sess, sess.graph["RNN2/Bias/Bf"]) ≈ fill(2f0, 10)
end






@testset "dropout" begin
    sess = Session(Graph())
    inputs = constant(randn(Float32, 5, 32, 5))
    drop_prob = placeholder(Float32)

    dd =  nn.dropout(inputs, drop_prob)  # Check that this can actually be created

end

@testset "l2loss" begin
    sess = Session(Graph())
    x = [3.0, 4.0]
    loss = nn.l2_loss(constant(x))
    @test run(sess, loss) ≈ sum(x.^2)/2
end

@testset "top_k" begin
    sess = Session(Graph())
    inputs = constant([1,10,2,7,3])
    topk_values, topk_indices = run(sess, nn.top_k(inputs,1))
    @test topk_values == [10]
    @test topk_indices == [2]
    topk_values, topk_indices = run(sess, nn.top_k(inputs,2))
    @test topk_values == [10,7]
    @test topk_indices == [2,4]
end

@testset "Activation Functions" begin
    sess = Session(Graph())
    run_op(op, val)=run(sess, op(constant(val)))

    x = 0.5
    xs = [-0.5, 0.0, 0.5, 0.6]'

    @test logistic(x) == run_op(logistic, x) == run_op(nn.sigmoid, x)
    @test logistic.(xs) == run_op(logistic, xs) == run_op(nn.sigmoid, xs)

    @test softmax(xs) == run_op(nn.softmax, xs)

    @test softplus(x) == run_op(log1pexp, x) == run_op(nn.softplus, x)
    @test softplus.(xs) == run_op(log1pexp, xs) == run_op(nn.softplus, xs)
end

@testset "Pooling" begin
    sess = Session(Graph())
    input = ones(32, 4, 4, 3)
    for pool_fn in [nn.max_pool, nn.avg_pool]
        op = pool_fn(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        output = run(sess, op)
        @test size(output) == (32, 2, 2, 3)
    end
end
