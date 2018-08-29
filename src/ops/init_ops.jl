struct ConstantInitializer{T}
    value::T
end

function Base.rand(c::ConstantInitializer, shape...)
    fill(c.value, shape)
end

function Base.rand(c::ConstantInitializer, shape::Integer...)
    fill(c.value, shape)
end

#function rand(c::ConstantInitializer, shape::Integer, shapes::Integer...)
#	fill(c.value, [shape, shapes...]...)
#end

"""
    zeros_initializer(dtype=Float32)

Initializer that generates tensors initialized to 0.
"""
function zeros_initializer(dtype=Float32)
    function(size::Array{T, 1} where T<:Integer)
        zeros(Tensor{dtype}, size...)
    end
end

"""
    ones_initializer(dtype=Float32)

Initializer that generates tensors initialized to 1.
"""
function ones_initializer(dtype=Float32)
    function(size::Array{T, 1} where T<:Integer)
        ones(Tensor{dtype}, size...)
    end
end
