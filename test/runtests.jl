using TensorFlow
using Base.Test

include("hello.jl")
include("math.jl")
include("debug.jl")
include("image.jl")
include(joinpath(dirname(@__FILE__), "..", "examples", "logistic.jl"))

# RNNs

cell = nn.rnn_cell.BasicRNNCell(5)
s0 = nn.zero_state(cell, 10, Float64)
x = constant(randn(10, 3))
y, s1 = cell(x, s0)
