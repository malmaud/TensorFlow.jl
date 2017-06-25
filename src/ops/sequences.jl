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
    @eval Base.$f(::Type{Tensor}, args...) = $f(Tensor{Float32}, args...)
    @eval Base.$f(::Type{Tensor}, args::Tuple) = $f(Tensor, args...)
    @eval Base.$f{T}(::Type{Tensor{T}}, args...) = constant($f(T, args...))
    @eval Base.$f{T}(::Type{Tensor{T}}, args::Tuple) = constant($f(T, args))
end

@op function random_normal(shape; mean=0.0, stddev=1.0, name=nothing, kwargs...)
    local out
    with_op_name(name, "random_normal") do
        standard = Ops.random_standard_normal(shape; name=name, kwargs...)
        out = standard.*stddev + mean
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
@op function random_uniform(shape, minval, maxval; name=nothing, seed=0, dtype=Float32)
    local out
    with_op_name(name, "RandomUniformScaled") do
        seed1 = 0
        # TODO use global seed
        seed2 = seed
        r = random_uniform(shape; seed=seed1, seed2=seed2, dtype=dtype, name=name)
        out = r .* (maxval-minval) + minval
    end
    out
end


@op function Base.shuffle(t::AbstractTensor; kwargs...)
    Ops.random_shuffle(t; kwargs...)
end

@op function Base.linspace(start::AbstractTensor, stop, num; kwargs...)
    Ops.lin_space(start, stop, num; kwargs...)
end

@op Base.range(start::AbstractTensor, length; kwargs...) = range(start, 1, length; kwargs...)

@op function Base.range(start::AbstractTensor, step, length; kwargs...)
    Ops.range(start, length+1, step; kwargs...)
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
