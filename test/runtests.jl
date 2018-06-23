using TensorFlow
using Base.Test

srand(0)  # Make tests deterministic

include(joinpath(dirname(@__FILE__), "..", "examples", "logistic.jl"))

# Main tests

tests = [
"clipping.jl",
"comp.jl",
"control.jl",
"core.jl",
"debug.jl",
"hello.jl",
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

# @test_nowarn was added in Julia 0.6.
# We make it a no-op on earlier versions.
if !isdefined(Base.Test, Symbol("@test_nowarn"))
    macro test_nowarn(ex)
        esc(ex)
    end
end


tf_versioninfo() # Dump out all the info at start of the test, for easy debugging from logs. (also check `tf_versioninfo()` itself works)

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
