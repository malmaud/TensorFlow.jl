using TensorFlow
using Base.Test

srand(0)  # Make tests deterministic

include(joinpath(dirname(@__FILE__), "..", "examples", "logistic.jl"))


# Main tests

tests = [
    "hello.jl",
    "core.jl",
    "shape_inference.jl",
    "math.jl",
    "debug.jl",
    "comp.jl",
    "clipping.jl",
    "image.jl",
    "transformations.jl",
    "proto.jl",
    "meta.jl",
    "control.jl",
    "ops.jl",
    "summary.jl",
    "nn.jl",
    "sequences.jl",
    "run.jl",
    "training.jl"
]

# @test_nowarn was added in Julia 0.6.
# We make it a no-op on earlier versions.
if !isdefined(Base.Test, Symbol("@test_nowarn"))
    macro test_nowarn(ex)
        esc(ex)
    end
end

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
