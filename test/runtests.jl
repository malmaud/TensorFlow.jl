using TensorFlow
using Test
import Random

Random.seed!(0)  # Make tests deterministic

include(joinpath(dirname(@__FILE__), "..", "examples", "logistic.jl"))

# Main tests

tests = [
"hello.jl",
"clipping.jl",
"comp.jl",
"control.jl",
"core.jl",
"debug.jl",
"image.jl",
"init_ops.jl",
"io.jl",
"math.jl",
"meta.jl",
"nn.jl",
"ops.jl",
"proto.jl",
"run.jl",
"sequences.jl",
"shape_inference.jl",
"show.jl",
"summary.jl",
"train.jl",
"training.jl",
"transformations.jl",
"summary_eager.jl"
]

tf_versioninfo() # Dump out all the info at start of the test, for easy debugging from logs. (also check `tf_versioninfo()` itself works)

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end

# TODO configure tests so they automatically set the appropriate graph or eager mode for themselves. For now,
# all the eager tests run at the end.
include(joinpath(dirname(@__FILE__), "..", "examples", "keras.jl"))
