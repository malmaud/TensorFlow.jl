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
]


tf_versioninfo() # Dump out all the info at start of the test, for easy debugging from logs. (also check `tf_versioninfo()` itself works)

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
