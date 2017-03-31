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

for (rnn_fun, post_proc_outputs) in ((nn.dynamic_rnn, identity), (nn.rnn, last))
    testname = split(string(rnn_fun), ".")[end]
    @testset "$testname" begin
        let
            sess = Session(Graph())
            data = constant(ones(1, 2, 1))
            cell = nn.rnn_cell.BasicRNNCell(1)
            s0 = nn.zero_state(cell, 1, Float64)
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
            data_jl = Float32[100*x+10y+z for x in 1:10, y in 1:20, z in 1:10]
            data = constant(data_jl)
            lens_jl = collect(1:2:20) #1 for each element in the batch (x) saying how far to go down the time (y)
            lens = constant(lens_jl)
            cell = nn.rnn_cell.IdentityRNNCell(10)
            y, s_last = rnn_fun(cell, data, lens; dtype=Float32)

            run(sess, global_variables_initializer())
            outputs = run(sess, y)
            output = post_proc_outputs(outputs)
            @test output == [data_jl[xi, lens_jl[xi], zi] for xi in 1:10, zi in 1:10]

        end

        let
            sess = Session(Graph())
            data_jl = Float32[100*x+10y+z for x in 1:20, y in 1:10, z in 1:10] #Time first dim, batch second
            data = constant(data_jl)
            lens_jl = collect(1:2:20) #1 for each element in the batch (x) saying how far to go down the time (y)
            lens = constant(lens_jl)
            cell = nn.rnn_cell.IdentityRNNCell(10)
            y, s_last = rnn_fun(cell, data, lens; dtype=Float32, time_major=true)

            run(sess, global_variables_initializer())
            outputs = run(sess, y)
            output = post_proc_outputs(outputs)
            @test output == [data_jl[lens_jl[xi], xi, zi] for xi in 1:10, zi in 1:10]
        end
    end
end

@testset "rnn gradients" begin
    sess = Session(Graph())
    cell = nn.rnn_cell.LSTMCell(10)
    s0 = nn.zero_state(cell, 5, Float32)
    inputs = constant(randn(Float32, 5, 32, 5))
    out = nn.dynamic_rnn(cell, inputs, initial_state=s0)
    loss = reduce_sum(out[1]).^2
    minimizer = train.GradientDescentOptimizer(.01)
    minimize_op = train.minimize(minimizer, loss)
    run(sess, global_variables_initializer())
    run(sess, minimize_op)
end
