using TensorFlow
using Base.Test

#TODO workout why these tests sometimes break
#Remove this hack that stops them running in CI
const FLAKEY_TESTS_ENABLED = !haskey(ENV,"TRAVIS_JULIA_VERSION")

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

@testset "dynamic_rnn" begin
    let
        sess = Session(Graph())
        data = constant(ones(1, 2, 1))
        cell = nn.rnn_cell.BasicRNNCell(1)
        s0 = nn.zero_state(cell, 1, Float64)
        local y
        variable_scope("rnn", initializer=ConstantInitializer(.1)) do
            y = nn.dynamic_rnn(cell, data, initial_state=s0)
        end

        if FLAKEY_TESTS_ENABLED
            run(sess, global_variables_initializer()) #This line is flakily breaking on Travis
            output = run(sess, y)[1]
            expected_output = tanh(1*.1+tanh(1*.1+.1)*.1+.1)
            @test output[1,1] ≈ expected_output
        end
    end
end




@testset "dynamic_rnn sequence_length" begin
    let
        sess = Session(Graph())
        data_jl = Float32[100*x+10y+z for x in 1:10, y in 1:20, z in 1:10]
        data = constant(data_jl)
        lens_jl = collect(1:2:20) #1 for each element in the batch (x) saying how far to go down the time (y)
        lens = constant(lens_jl)
        cell = nn.rnn_cell.IdentityRNNCell(10)
        y, s_last = nn.dynamic_rnn(cell, data, lens; dtype=Float32)

        if FLAKEY_TESTS_ENABLED
            run(sess, global_variables_initializer()) #This line is flakily breaking on Travis
            y_o = run(sess, y)
            @test y_o == [data_jl[xi, lens_jl[xi], zi] for xi in 1:10, zi in 1:10]
        end
    end

    let
        sess = Session(Graph())
        data_jl = Float32[100*x+10y+z for x in 1:20, y in 1:10, z in 1:10] #Time first dim, batch second
        data = constant(data_jl)
        lens_jl = collect(1:2:20) #1 for each element in the batch (x) saying how far to go down the time (y)
        lens = constant(lens_jl)
        cell = nn.rnn_cell.IdentityRNNCell(10)
        y, s_last = nn.dynamic_rnn(cell, data, lens; dtype=Float32, time_major=true)

        if FLAKEY_TESTS_ENABLED
            run(sess, global_variables_initializer()) #This line is flakily breaking on Travis
            y_o = run(sess, y)
            @test y_o == [data_jl[lens_jl[xi], xi, zi] for xi in 1:10, zi in 1:10]
        end
    end
end
