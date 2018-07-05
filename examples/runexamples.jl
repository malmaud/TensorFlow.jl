using Base.Test

examples = [
"ae.jl"
"logistic.jl"
"mnist_full.jl"
"mnist_simple.jl"
]

for filename in examples
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
