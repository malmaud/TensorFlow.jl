"""
Creates a constant `Tensor`.
"""
@op function constant(tensor; name=nothing)
    local desc
    with_op_name(name, "Const") do
        desc = NodeDescription("Const")
        tensor = RawTensor(tensor)
        desc["dtype"] = eltype(tensor)
        desc["value"] = tensor
    end
    Tensor(Operation(desc), 1)
end

function Base.zeros(::Type{Tensor}, T, shape::Tuple)
    constant(zeros(T, shape))
end

Base.zeros(::Type{Tensor}, shape::Tuple) = zeros(Tensor, Float32, shape)

function Base.ones(::Type{Tensor}, T, shape::Tuple)
    constant(ones(T, shape))
end

Base.ones(::Type{Tensor}, shape::Tuple) = ones(Tensor, Float32, shape)

"""
Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound `maxval` is excluded.

For floats, the default range is `[0, 1)`. For ints, at least `maxval` must be specified explicitly.

In the integer case, the random integers are slightly biased unless
`maxval - minval` is an exact power of two.
The bias is small for values of `maxval - minval` significantly smaller than the
range of the output (either 2**32 or 2**64).

Args:
* `shape`: A one dimensional `Tensor` or array containing the shape of the output `Tensor`.
* `minval`: A zero dimensional `Tensor` or value of type `dtype`. Lower bound on random values.
* `maxval`: A zero dimensional `Tensor` or value of type `dtype`. Upper bound on random values.
* `seed`: An integer to seed the RNG with. Defaults to `0`.
* `dtype`: Optional datatype of random values generated. Default is `Float32`.

Returns:
A `Tensor` of the specified `shape` and `dtype` containing random values.
"""
@op function random_uniform(shape, minval, maxval; name=nothing, seed=0, dtype=Float32)
    local out
    with_op_name(name, "RandomUniform") do
        desc = NodeDescription("RandomUniform")
        add_input(desc, Tensor(shape))
        desc["dtype"] = dtype
        desc["seed2"] = seed
        # TODO use global seed
        desc["seed"] = 0
        r = Tensor(Operation(desc), 1)
        minval = convert(Tensor{dtype}, minval)
        maxval = convert(Tensor{dtype}, maxval)
        out = r .* (maxval-minval) + minval
    end
    out
end

"""
Outputs random values from a normal distribution.

Args:
* `shape`: A one dimensional `Tensor` or array containing the shape of the output `Tensor`.
* `mean`: A zero dimensional `Tensor` or value of type `dtype`. The mean of the normal distribution.
* `stddev`: A zero dimensional `Tensor` or value of type `dtype`. The standard deviation of the normal distribution.
* `seed`: An integer to seed the RNG with. Defaults to `0`.
* `dtype`: Optional datatype of random values generated. Default is `Float32`.

Returns:
A `Tensor` of the specified `shape` and `dtype` containing random values.
"""
@op function random_normal(shape; mean=0.0, stddev=1.0, dtype=Float32, seed=0, name=nothing)
    local out
    with_op_name(name, "RandomNormal") do
        desc = NodeDescription("RandomStandardNormal")
        add_input(desc, Tensor(shape))
        desc["dtype"] = dtype
        var = Tensor(Operation(desc))
        out = stddev.*var + mean
    end
    out
end

@op function Base.shuffle(t::Tensor; seed=0, name=nothing)
    local desc
    with_op_name(name, "RandomShuffle") do
        desc = NodeDescription("RandomShuffle")
        add_input(desc, t)
        desc["seed2"] = Int64(seed)
    end
    Tensor(Operation(desc))
end

@op function Base.linspace(::Type{Tensor}, start, stop, num; name=nothing)
    local desc
    with_op_name(name, "LinSpace") do
        desc = NodeDescription("LinSpace")
        add_input(desc, convert(Tensor{Float32}, start))
        add_input(desc, convert(Tensor{Float32}, stop))
        add_input(desc, convert(Tensor{Int32}, num))
    end
    Tensor(Operation(desc), 1)
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

"""
`function fill(n::AbstractTensor, dims::AbstractTensor)`

Creates a tensor filled with a scalar value.

This operation creates a tensor of shape `dims` and fills it with `value`.

For example:

```prettyprint
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

Args:
  dims: A `Tensor` of type `int32`.
    1-D. Represents the shape of the output tensor.
  value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
  name: A name for the operation (optional).

Returns:
  A `Tensor`. Has the same type as `value`.
"""
@op function Base.fill(n::AbstractTensor, dims::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Fill") do
        desc = NodeDescription("Fill")
        add_input(desc, convert(Tensor{Int32}, dims))
        add_input(desc, n)
    end
    Tensor(Operation(desc), 1)
end

@op function Base.fill(::Type{Tensor}, n, dims; name=nothing)
    fill(Tensor(n), Tensor(dims); name=name)
end

"""
`reverse(x::AbstractTensor, indices)`

Reverses specific dimensions of a tensor.

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

`rank(tensor) = size(dims)`

For example:

```prettyprint
Reverses specific dimensions of a tensor.

NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.

`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.

For example:

```prettyprint
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [3] or 'dims' is -1
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

Args:
  tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `int64`, `bool`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    Up to 8-D.
  axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    1-D. The indices of the dimensions to reverse.
  name: A name for the operation (optional).

Returns:
  A `Tensor`. Has the same type as `tensor`. The same shape as `tensor`.
"""
@op function Base.reverse(x::AbstractTensor, indices; name=nothing)
    local desc
    with_op_name(name, "Reverse") do
        desc = NodeDescription("ReverseV2")
        add_input(desc, Tensor(x))
        add_input(desc, convert(Tensor{Int32}, indices) - 1)
    end
    Tensor(Operation(desc))
end
