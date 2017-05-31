immutable ConstantInitializer{T}
    value::T
end

function Base.rand(c::ConstantInitializer, shape...)
    fill(c.value, shape)
end

"""
    zeros_initializer(dtype=Float32)

Initializer that generates tensors initialized to 0.
"""
function zeros_initializer(dtype=Float32)
    size->zeros(Tensor{dtype}, size...)
end

"""
    ones_initializer(dtype=Float32)

Initializer that generates tensors initialized to 1.
"""
function ones_initializer(dtype=Float32)
    size->ones(Tensor{dtype}, size...)
end
