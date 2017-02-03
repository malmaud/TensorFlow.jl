using TensorFlow
using Base.Test

srand(0)  # Make tests deterministic

include(joinpath(dirname(@__FILE__), "..", "examples", "logistic.jl"))

# RNNs

cell = nn.rnn_cell.BasicRNNCell(5)
s0 = nn.zero_state(cell, 10, Float64)
x = constant(randn(10, 3))
y, s1 = cell(x, s0)

# Main tests

tests = [
    "hello.jl",
    "core.jl",
    "math.jl",
    "debug.jl",
    "comp.jl",
    "clipping.jl",
    "image.jl",
    "transformations.jl",
    "proto.jl"
]

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
