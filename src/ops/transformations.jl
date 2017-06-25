import .Ops:
    strided_slice,
    expand_dims,
    tile,
    pad,
    gather,
    gather_nd,
    scatter_nd,
    dynamic_partition,
    dynamic_stitch

const concat = Ops.concat_v2
const stack = Ops.pack

function cast(value, dst_t; kwargs...)
    Ops.cast(value, DstT=dst_t, kwargs...)
end

function one_hot(indices, depth; on_value=1.0, off_value=0.0, kwargs...)
    Ops.one_hot(indices, depth, on_value, off_value; kwargs...)
end

"""
Base.reshape(n::AbstractTensor, dims; name="")

Reshapes a tensor.

Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.

If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.

If shape is 1-D or higher, then the operation returns a tensor with shape shape filled with the values of tensor. In this case, the number of elements implied by shape must be the same as the number of elements in tensor.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#reshape
"""
@op Base.reshape(n::AbstractTensor, dims; name=nothing) =
  reshape(n, Tensor(Int32[dims...]); name = name)

@op function Base.reshape(n::AbstractTensor, dims::AbstractTensor; kwargs...)
    Ops.reshape(n, dims; kwargs...)
end


"""
Base.split(split_dim, num_split, value::AbstractTensor; name="")

Splits a tensor into num_split tensors along one dimension.

Splits value along dimension split_dim into num_split smaller tensors. Requires that num_split evenly divide value.shape[split_dim].

For example:

# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(1, 3, value)
tf.shape(split0) ==> [5, 10]
Note: If you are splitting along an axis by the length of that axis, consider using unpack, e.g.

num_items = t.get_shape()[axis].value
[tf.squeeze(s, [axis]) for s in tf.split(axis, num_items, t)]
can be rewritten as

tf.unpack(t, axis=axis)
Args:

split_dim: A 0-D int32 Tensor. The dimension along which to split. Must be in the range [0, rank(value)).
num_split: A Python integer. The number of ways to split.
value: The Tensor to split.
name: A name for the operation (optional).
Returns:

num_split Tensor objects resulting from splitting value.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#split
"""
@op function Base.split(split_dim, num_split, value::AbstractTensor; kwargs...)
    Ops.split(split_dim, value; num_split=num_split, kwargs...)
end



"""
Applies ExpandDims to the input Tensors `xs`,
until they are all the same rank -- which must be at least `min_rank`
"""
function expand_to_same_ranks(min_rank, xs)
    compat_xs = collect(xs)
    @label fix_dims
    for (ii, x) in enumerate(compat_xs)
        x_dims = get_shape(x)
        x_dims.rank_unknown && continue
        rank = length(x_dims.dims)
        if rank > min_rank
            min_rank = rank
            @goto fix_dims # Have to restart
            # since previous things were only expanded to some insurficient rank
        elseif rank < min_rank
            for new_dim in (rank + 1) : min_rank
                x = expand_dims(x, new_dim) # have to add one at a time
                # as can't use reshape, as don't nesc. fully know the shape of input
            end
            compat_xs[ii] = x # save updated value
        end
    end
    compat_xs
end


"""
Concatenate the tensor `values` along the given axis (`dim`).
Unlike `concat` this automatically expands dimensions as required,
such that the resulting concatenated Tensor has rank equal to the
higher of the highest input rank, or the concatentation dimension (`dim`).
"""
function Base.cat(dim, xs::AbstractTensor...)
    with_op_name("Cat") do
        compat_xs = tf_promote(expand_to_same_ranks(dim, xs)...)
        if length(xs)>1
             concat(compat_xs, dim)
        else
             compat_xs[1] # If only one input then, no actual concatentation to be done
        end
    end
end

Base.cat(::Type{Tensor}, dim, values...) = cat(dim, Tensor.(values)...)


"""
Concatentate along dimension 1

`vcat(a, b)` can also be written `[a; b]` etc.
"""
Base.vcat(xs::AbstractTensor...) = cat(1, xs...)
# Catch common cases where not all args are Tensors, and convert them
Base.vcat(x1::AbstractTensor, xs...) = vcat(x1, Tensor.(xs)...)
Base.vcat(x1, x2::AbstractTensor, xs...) = vcat(Tensor(x1), x2, Tensor.(xs)...)
Base.vcat(x1::AbstractTensor, x2::AbstractTensor, xs...) = vcat(x1, x2, Tensor.(xs)...)


"""
Concatentate along dimension 2

`hcat(a, b)` can also be written `[a b]` etc.
"""
Base.hcat(xs::AbstractTensor...) = cat(2, xs...)
# Catch common cases where not all args are Tensors, and convert them
Base.hcat(x1::AbstractTensor, xs...) = hcat(x1, Tensor.(xs)...)
Base.hcat(x1, x2::AbstractTensor, xs...) = hcat(Tensor(x1), x2, Tensor.(xs)...)
Base.hcat(x1::AbstractTensor, x2::AbstractTensor, xs...) = hcat(x1, x2, Tensor.(xs)...)




"""
    unstack(value; num=nothing, axis=1, name="Unpack")

Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
If `num` is not specified (the default), it is inferred from `value`'s shape.
If `value.shape[axis]` is not known, `ValueError` is raised.

For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 1` then the i'th tensor in `output` is the slice
  `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
  (Note that the dimension unpacked along is gone, unlike `split`).

If `axis == 2` then the i'th tensor in `output` is the slice
  `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of pack.  The numpy equivalent is

    tf.unpack(x, n) = list(x)

Args:
  value: A rank `R > 0` `Tensor` to be unpacked.
  num: An `int`. The length of the dimension `axis`. Automatically inferred
    if `None` (the default).
  axis: An `int`. The axis to unpack along. Defaults to the first
    dimension. Supports negative indexes.
  name: A name for the operation (optional).

Returns:
  The list of `Tensor` objects unpacked from `value`.

Raises:
  ValueError: If `num` is unspecified and cannot be inferred.
  ValueError: If `axis` is out of the range [-R, R).
"""
@op function unstack(value; num=nothing, axis=1, name=nothing)
    num_split = num==nothing ? get_shape(value, axis) : num
    Ops.unpack(value, num=num_split, axis=axis)
end


"""
    squeeze(x::AbstractTensor, squeeze_dims; name="squeeze")

Removes dimensions of size 1 from the shape of a tensor.
Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`axis`.
For example:
```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```
Or, to remove specific size 1 dimensions:
```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [3, 5])) ==> [1, 2, 3, 1]
```
Args:
  input: A `Tensor`. The `input` to squeeze.
  axis: An optional list of `ints`. Defaults to `[]`.
    If specified, only squeezes the dimensions listed. The dimension
    index starts at 1. It is an error to squeeze a dimension that is not 1.
  name: A name for the operation (optional).
  squeeze_dims: Deprecated keyword argument that is now axis.
Returns:
  A `Tensor`. Has the same type as `input`.
  Contains the same data as `input`, but has one or more dimensions of
  size 1 removed.
Raises:
  ValueError: When both `squeeze_dims` and `axis` are specified.
"""
@op function Base.squeeze(x::AbstractTensor, squeeze_dims=nothing; kwargs...)
    if squeeze_dims !== nothing
        squeeze_dims = squeeze_dims - 1
    end
    Ops.squeeze(x; squeeze_dims=squeeze_dims, kwargs...)
end


"""
Base.rank(n::AbstractTensor; name="")

Returns the rank of a tensor.

This operation returns an integer representing the rank of input.

For example:

# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
Note: The rank of a tensor is not the same as the rank of a matrix. The rank of a tensor is the number of indices required to uniquely select each element of the tensor. Rank is also known as "order", "degree", or "ndims."

Args:

input: A Tensor or SparseTensor.
name: A name for the operation (optional).
Returns:

A Tensor of type int32.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#rank
"""
@define_unary Base.rank Ops.rank

@op function scatter_nd(indices, updates, shape::TensorFlow.TensorShape; name=nothing)
    if shape.rank_unknown || any(isnull.(shape.dims))
        error("TensorShape provided to scatter_nd not statically fully known ($shape). Consider using the dynamic `shape` operation instead of the static `get_shape` operation")
    end
    scatter_nd(indices, updates, get.(shape.dims); name=name)
end


"""
boolean_mask(tensor, mask)

Apply boolean mask to tensor.  Numpy equivalent is `tensor[mask]`.

```julia
# 1-D example
tensor = [0, 1, 2, 3]
mask = [True, False, True, False]
boolean_mask(tensor, mask) ==> [0, 2]
```

In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
the first K dimensions of `tensor`'s shape.  We then have:
  `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).

Args:
  tensor:  N-D tensor.
  mask:  K-D boolean tensor, K <= N and K must be known statically.
  name:  A name for this operation (optional).

Returns:
  Tensor populated by entries in `tensor` corresponding to `True` values in
    `mask`.

Raises:
  ValueError:  If shapes do not conform.

Examples:

```julia
# 2-D example
tensor = [[1, 2], [3, 4], [5, 6]]
mask = [True, False, True]
boolean_mask(tensor, mask) ==> [[1, 2], [5, 6]]
```
"""
@op function boolean_mask(tensor, mask::AbstractTensor; name=nothing)
    local result
    with_op_name(name, "BooleanMask") do
        indices = find(mask)  # TODO generalize to more dimensions
        squeezed = squeeze(indices, [2])
        result = tensor[squeezed]
    end
    result
end

@op function boolean_mask(tensor, mask::AbstractArray; name=nothing)
    local result
    with_op_name(name, "BooleanMask") do
        indices = find(mask)  # TODO generalize to more dimensions
        result = tensor[indices]
    end
    result
end



"""
`transpose(n::AbstractTensor, perm=nothing)`

Transposes `a`. Permutes the dimensions according to `perm`.

The returned tensor's dimension i will correspond to the input dimension
`perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
the rank of the input tensor. Hence by default, this operation performs a
regular matrix transpose on 2-D input Tensors.

For example:

```julia
# 'x' is [[1 2 3]
#         [4 5 6]]
tf.transpose(x) ==> [[1 4]
                     [2 5]
                     [3 6]]

# Equivalently
tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                  [2 5]
                                  [3 6]]

# 'perm' is more useful for n-dimensional tensors, for n > 2
# 'x' is   [[[1  2  3]
#            [4  5  6]]
#           [[7  8  9]
#            [10 11 12]]]
# Take the transpose of the matrices in dimension-0
tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
                                      [2  5]
                                      [3  6]]

                                     [[7 10]
                                      [8 11]
                                      [9 12]]]
```

Args:
  a: A `Tensor`.
  perm: A permutation of the dimensions of `a`.
  name: A name for the operation (optional).

Returns:
  A transposed `Tensor`.
"""
@op function Base.transpose(n::AbstractTensor, perm=nothing; name=nothing)
    local result
    with_op_name(name, "Transpose") do
        if perm === nothing
            r = range(constant(0), rank(n)-1)
            perm = reverse(r, [true])
        end
        result = Ops.transpose(n, perm)
    end
    result
end

@op function Base.permutedims(n::AbstractTensor, perm; name=nothing)
    transpose(n, perm - 1; name=name)
end

@define_unary Base.ctranspose transpose


"""
slice(n::AbstractTensor, begin_, size_; name="")
Extracts a slice from a tensor.
This operation extracts a slice of size size from a tensor input starting at the location specified by begin. The slice size is represented as a tensor shape, where size[i] is the number of elements of the 'i'th dimension of input that you want to slice. The starting location (begin) for the slice is represented as an offset in each dimension of input. In other words, begin[i] is the offset into the 'i'th dimension of input that you want to slice from.
begin is zero-based; size is one-based. If size[i] is -1, all remaining elements in dimension i are included in the slice. In other words, this is equivalent to setting:
size[i] = input.dim_size(i) - begin[i]
This operation requires that:
0 <= begin[i] <= begin[i] + size[i] <= Di for i in [0, n]
Args:
input_: A Tensor.
begin: An int32 or int64 Tensor.
size: An int32 or int64 Tensor.
name: A name for the operation (optional).
Returns:
A Tensor the same type as input.
https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#slice
"""
@op function slice(n::AbstractTensor, begin_, size_; name=nothing)
    with_op_name(name, "Slice") do
        begin_ = convert(Tensor{Int32}, begin_)
        size_ = convert(Tensor{Int32}, size_)
        Ops.slice(n, begin_, size_; name=name)
    end
end



include("indexing.jl")
