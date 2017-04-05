import .Ops:
    random_uniform,
    random_standard_normal,
    random_shuffle

@op function constant(value; dtype=nothing, kwargs...)
    if dtype === nothing
        if isa(value, AbstractString)
            dtype = String
        else
            dtype = eltype(value)
        end
    end
    Ops.const_(; value=value, dtype=dtype, kwargs...)
end

for f in [:zeros, :ones]
    @eval Base.$f{T}(::Type{Tensor{T}}, shape...) = constant($f(T, shape...))
    @eval Base.$f(::Type{Tensor}, args...) = $f(Tensor{Float32}, args...)
end

function random_normal(shape; mean=0.0, stddev=1.0, name=nothing, kwargs...)
    local out
    with_op_name(name, "random_normal") do
        standard = Ops.random_standard_normal(shape; name=name, kwargs...)
        out = standard.*stddev + mean
    end
    out
end

@op function Base.shuffle(t::AbstractTensor; kwargs...)
    Ops.random_shuffle(t; kwargs...)
end

@op function Base.linspace(start::AbstractTensor, stop, num; kwargs...)
    Ops.lin_space(start, stop, num; kwargs...)
end

@op function Base.range(::Type{Tensor}, start; limit=nothing, delta=1, name=nothing)
    if limit == nothing
        limit = start
        start = 0
    end
    local desc
    with_op_name(name, "Range") do
        desc = NodeDescription("Range")
        add_input(desc, convert(Tensor{Int32}, start))
        add_input(desc, convert(Tensor{Int32}, limit))
        add_input(desc, convert(Tensor{Int32}, delta))
    end
    Tensor(Operation(desc), 1)
end

@op function Base.fill(n::AbstractTensor, dims; kwargs...)
    Ops.fill(n, dims; kwargs...)
end

@op function Base.reverse(x::AbstractTensor, indices; kwargs...)
    Ops.reverse_v2(x, indices; kwargs...)
end
