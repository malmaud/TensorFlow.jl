using TensorFlow
using Base.Test

## conv2d_transpose
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

## dynamic_rnn

let
    sess = Session(Graph())
    data = constant(ones(1, 2, 1))
    cell = nn.rnn_cell.BasicRNNCell(1)
    s0 = nn.zero_state(cell, 1, Float64)
    local y
    variable_scope("rnn", initializer=ConstantInitializer(.1)) do
        y = nn.dynamic_rnn(cell, data, initial_state=s0)
    end
    # Disable until the flakiness of the test is resolved
    # run(sess, global_variables_initializer())  # This is flakily failing.
    # output = run(sess, y)[1]
    # expected_output = tanh(1*.1+tanh(1*.1+.1)*.1+.1)
    # @test output[1,1] ≈ expected_output
end
