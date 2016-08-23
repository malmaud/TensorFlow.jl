using TensorFlow
using Base.Test

include("hello.jl")
include(joinpath(dirname(@__FILE__), "..", "examples", "logistic.jl"))
