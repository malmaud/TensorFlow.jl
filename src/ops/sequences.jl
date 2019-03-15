import .Ops:
    random_uniform,
    random_standard_normal,
    random_shuffle

import Random # shuffle

function convert_eltype(x::Array, dtype)
    convert(Array{dtype}, x)
end

function convert_eltype(x::Number, dtype)
    convert(dtype, x)
end

convert_eltype(x, dtype) = x

@op function constant(value; dtype = nothing, kwargs...)
    if dtype === nothing
        if isa(value, AbstractString)
            dtype = String
        else
            dtype = eltype(value)
        end
    else
        value = convert_eltype(value, dtype)
    end
    if in_eager_mode()
        EagerTensor(value)
    else
        Ops.const_(; value = value, dtype = dtype, kwargs...)
    end
end

for f in [:zeros, :ones]
    @eval Base.$f(::Type{Tensor}, args::Integer...) = $f(Tensor{Float32}, args...)
    @eval Base.$f(::Type{Tensor}, args::NTuple{N,Integer}) where N = $f(Tensor, args...)
    @eval Base.$f(::Type{Tensor{T}}, args::Integer...) where {T} = constant($f(T, args...))
    @eval Base.$f(::Type{Tensor{T}}, args::NTuple{N,Integer}) where {T,N} = constant($f(T, args))
end

@op function random_normal(shape; mean = 0.0, stddev = 1.0, dtype = Float32, name = nothing, kwargs...)
    local out
    with_op_name(name, "random_normal") do
        mean = convert(Tensor{dtype}, mean)
        stddev = convert(Tensor{dtype}, stddev)
        standard = Ops.random_standard_normal(shape; name = name, dtype = dtype, kwargs...)
        out = standard .* stddev + mean
    end
    out
end

"""
Outputs random values from a uniform distribution.
The generated values follow a uniform distribution in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound `maxval` is excluded.
In the integer case, the random integers are slightly biased unless
`maxval - minval` is an exact power of two.
The bias is small for values of `maxval - minval` significantly smaller than the
range of the output (either 2**32 or 2**64).
Args:
* `shape`: A one dimensional `Tensor` or array containing the shape of the output `Tensor`.
* `minval`: Lower bound on random values.
* `maxval`: Upper bound on random values.
* `seed`: An integer to seed the RNG with. Defaults to `0`, which results in random seed.
* `dtype`: Optional datatype of random values generated. Default is `Float32`.
Returns:
A `Tensor` of the specified `shape` and `dtype` containing random values.
"""
@op function random_uniform(shape, minval, maxval; name = nothing, seed = 0, dtype = Float32)
    local out
    with_op_name(name, "RandomUniformScaled") do
        seed1 = 0
        # TODO use global seed
        seed2 = seed
        minval = convert(Tensor{dtype}, minval)
        maxval = convert(Tensor{dtype}, maxval)
        r = random_uniform(shape; seed = seed1, seed2 = seed2, dtype = dtype, name = name)
        out = r .* (maxval - minval) + minval
    end
    out
end


@op function Random.shuffle(t::AbstractTensor; kwargs...)
    Ops.random_shuffle(t; kwargs...)
end

@op function Base.range(start::AbstractTensor; stop, num = Union{Integer,Nothin}, kwargs...)
    Ops.lin_space(start, stop, num; kwargs...)
end

@op Base.range(start::AbstractTensor, length; kwargs...) = range(start, 1, length; kwargs...)

@op function Base.range(start::AbstractTensor, step, length; kwargs...)
    Ops.range(start, length + 1, step; kwargs...)
end

@op function Base.fill(n::AbstractTensor, dims; kwargs...) #TODO: I think this is uncallable in 0.5
    Ops.fill(convert(Tensor{Int32}, [dims...]), n; kwargs...)
end

@op function Base.fill(n::AbstractTensor, dims::AbstractTensor; kwargs...)
    Ops.fill(convert(Tensor{Int32}, dims), n; kwargs...)
end


@op function Base.reverse(x::AbstractTensor, indices; kwargs...)
    Ops.reverse_v2(x, indices; kwargs...)
end

@op function Base.reverse(x::AbstractTensor; dims=0, kwargs...)
    reverse(x, [dims]; kwargs...)
end

@op function Base.fill(n::AbstractTensor, dims::Tuple{Vararg{Int64,N}} where N; kwargs...)
    invoke(fill, Tuple{AbstractTensor,Any}, n, dims; kwargs...)
end
