
"""
cast(x::Tensor, dtype; name="")

Casts a tensor to a new type.

The operation casts x (in case of Tensor) or x.values (in case of SparseTensor) to dtype.

For example:

# tensor `a` is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
Args:

x: A Tensor or SparseTensor.
dtype: The destination type.
name: A name for the operation (optional).
Returns:

A Tensor or SparseTensor with same shape as x.

Raises:

TypeError: If x cannot be cast to the dtype.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#cast
"""
@op function cast(x::Tensor, dtype; name=nothing)
    local desc
    with_op_name(name, "Cast") do
        desc = NodeDescription("Cast")
        add_input(desc, x)
        desc["DstT"] = dtype
        # desc["SrcT"] = eltype(x)
    end
    Tensor(Operation(desc), 1)
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

@op function Base.reshape(n::AbstractTensor, dims::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Reshape") do
        desc = NodeDescription("Reshape")
        add_input(desc, n)
        add_input(desc, dims)
    end
    Tensor(Operation(desc), 1)
end


# if isdefined(Base, :slice)  # Removed in .6
#     import Base: slice
# end
"""
    slice(n::AbstractTensor, begin_, size_; name="")

Extracts a slice from a tensor.

This operation extracts a slice of size size from a tensor input starting at the location specified by begin. The slice size is represented as a tensor shape, where size[i] is the number of elements of the 'i'th dimension of input that you want to slice. The starting location (begin) for the slice is represented as an offset in each dimension of input. In other words, begin[i] is the offset into the 'i'th dimension of input that you want to slice from.

`begin` and `size` are one-based. If size[i] is -1, all remaining elements in dimension `i` are included in the slice. In other words, this is equivalent to setting:

size[i] = input.dim_size(i) - begin[i]

This operation requires that:

1 <= begin[i] <= begin[i] + size[i] <= Di for i in [1, n]

Args:

input_: A Tensor.
begin: An int32 or int64 Tensor.
size: An int32 or int64 Tensor.
name: A name for the operation (optional).
Returns:

A Tensor the same type as input.
"""
@op function slice(n::AbstractTensor, begin_, size_; name=nothing)
    local desc
    with_op_name(name, "Slice") do
        desc = NodeDescription("Slice")
        add_input(desc, Tensor(n))
        add_input(desc, convert(Tensor{Int32}, begin_) - 1)  # Convert from 1-based to 0-based indexing
        add_input(desc, convert(Tensor{Int32}, size_))
    end
    Tensor(Operation(desc), 1)
end

@not_implemented function strided_slice()
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
@op function Base.split(split_dim, num_split, value::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Split") do
        desc = NodeDescription("Split")
        add_input(desc, convert(Tensor{Int32}, split_dim)-1)
        add_input(desc, Tensor(value))
        desc["num_split"] = num_split
    end
    op = Operation(desc)
    [Tensor(op, x) for x in 1:num_split]
end

"""
    concat(vaues, axis; name="concat")

Concatenates tensors along one dimension.

Concatenates the list of tensors `values` along dimension `axis` (1-based).  If
`values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated
result has shape

    [D0, D1, ... Raxis, ...Dn]

where

    Raxis = sum(Daxis(i))

That is, the data from the input tensors is joined along the `axis`
dimension.

The number of dimensions of the input tensors must match, and all dimensions
except `axis` must be equal.

For example:

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
tf.shape(tf.concat([t3, t4], 1)) ==> [2, 6]
```

Note: If you are concatenating along a new axis consider using stack.
E.g.

```python
tf.concat([tf.expand_dims(t, axis) for t in tensors], axis)
```

can be rewritten as

```python
tf.stack(tensors, axis=axis)
```

Args:
  values: A list of `Tensor` objects or a single `Tensor`.
  axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate.
  name: A name for the operation (optional).

Returns:
  A `Tensor` resulting from concatenation of the input tensors.
"""
@op function concat(values, axis; name=nothing)
    local desc
    with_op_name(name, "Concat") do
        desc = NodeDescription("ConcatV2")
        add_input(desc, [Tensor(x) for x in values])
        add_input(desc, convert(Tensor{Int32}, axis) - 1)
        desc["N"] = length(values)
    end
    Tensor(Operation(desc), 1)
end

Base.cat(::Type{Tensor}, dim, values...) = concat(values, dim)
Base.cat(dim, values::AbstractTensor...) = concat(values, dim)

"""
    stack(values; axis=1, name="")

Packs a list of rank-R tensors into one rank-(R+1) tensor.

Packs the list of tensors in values into a tensor with rank one higher than each tensor in values, by packing them along the axis dimension. Given a list of length N of tensors of shape (A, B, C);

If axis == 1 then the output tensor will have the shape (N, A, B, C).
If axis == 2 then the output tensor will have the shape (A, N, B, C). Etc.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#pack
"""
@op function stack(nodes; axis=1, name=nothing)
    local desc
    with_op_name(name, "Stack") do
        desc = NodeDescription("Pack")
        add_input(desc, [Tensor(x) for x in nodes])
        desc["N"] = length(nodes)
        desc["axis"] = axis -1
    end
    Tensor(Operation(desc), 1)
end

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
    local desc
    with_op_name(name, "Unstack") do
        desc = NodeDescription("Unpack")
        add_input(desc, value)
        desc["num"] = num_split
        desc["axis"] = axis - 1
    end
    op = Operation(desc)
    [Tensor(op, x) for x in 1:num_split]
end

"""
expand_dims(input, dim; name="")

Inserts a dimension of 1 into a tensor's shape.

Given a tensor input, this operation inserts a dimension of 1 at the dimension index dim of input's shape. The dimension index dim starts at one; if you specify a non-positive number for dim it is counted backward from the end. With `0` being the last dimension (`end-0`), `-1` being the second last (`end-1`) and so forth

This operation is useful if you want to add a batch dimension to a single element. For example, if you have a single image of shape `[height, width, channels]`, you can make it a batch of 1 image with `expand_dims(image, 1)`, which will make the shape `[1, height, width, channels]`.

Other examples:

```julia
# 't' is a tensor of shape [2]
shape(expand_dims(t, 1)) ==> [1, 2]
shape(expand_dims(t, 2)) ==> [2, 1]
shape(expand_dims(t, 0)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 1)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 4)) ==> [2, 3, 5, 1]
shape(expand_dims(t2, 0)) ==> [2, 3, 5, 1]
shape(expand_dims(t2, -1)) ==> [2, 3, 1, 5]

```

This operation requires that:
-input.dims() <= dim <= input.dims()+1

This operation is related to squeeze(), which removes dimensions of size 1.

Args:

input: A Tensor.
dim: A Tensor of type int32. 0-D (scalar). Specifies the dimension index at which to expand the shape of input.
name: A name for the operation (optional).
Returns:

A Tensor. Has the same type as input. Contains the same data as input, but its shape has an additional dimension of size 1 added.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#expand_dims
"""
@op function expand_dims(input, dim; name=nothing)
    local desc
    with_op_name(name, "ExpandDims") do
        desc = NodeDescription("ExpandDims")
        add_input(desc, Tensor(input))
        add_input(desc, convert(Tensor{Int32}, dim)-1)
    end
    Tensor(Operation(desc), 1)
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
@op function Base.squeeze(x::AbstractTensor, squeeze_dims=nothing; name=nothing)
    local desc
    with_op_name(name, "Squeeze") do
        desc = NodeDescription("Squeeze")
        add_input(desc, x)
        if !(squeeze_dims === nothing)
            set_attr_list(desc, "squeeze_dims", squeeze_dims-1)
        end
    end
    Tensor(Operation(desc), 1)
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
@op function Base.rank(n::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Rank") do
        desc = NodeDescription("Rank")
        add_input(desc, Tensor(n))
    end
    Tensor(Operation(desc), 1)
end

"""
Base.size(n::AbstractTensor; name="")
Returns the shape of the Tensor.
WARNING: this does not match the python TensorFlow `size` -- for that functionality, use `Base.length`
Returns the total number of elements in a Tensor.W
(Like julia `Base.length` does for an `Array`)
https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#size
"""
@op Base.size(n::AbstractTensor; name=nothing) = shape(n; name=name)
@op Base.size(n::AbstractTensor, i; name=nothing) = shape(n; name=name)[i]
# size(X, dim) must be implemented for indexing with X[..,end,...] to work

"""
Base.length(n::AbstractTensor; name="")
Returns the total number of elements in a Tensor.
This matchs python TensorFlow `size` operation
https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#size
"""
@op function Base.length(n::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Size") do
        desc = NodeDescription("Size")
        add_input(desc, Tensor(n))
    end
    Tensor(Operation(desc), 1)
end
@op Base.endof(n::AbstractTensor; name=nothing) = length(n; name=name)
# endof(X) must be implemented for indexing with X[end] to work

"""
tile(input, multiples; name="")

Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating input multiples times. The output tensor's i'th dimension has input.dims(i) * multiples[i] elements, and the values of input are replicated multiples[i] times along the 'i'th dimension. For example, tiling [a b c d] by [2] produces [a b c d a b c d].

Args:

input: A Tensor. 1-D or higher.
multiples: A Tensor of type int32. 1-D. Length must be the same as the number of dimensions in input
name: A name for the operation (optional).
Returns:

A Tensor. Has the same type as input.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#tile
"""
@op function tile(input, multiples; name=nothing)
    local desc
    with_op_name(name, "Tile") do
        desc = NodeDescription("Tile")
        add_input(desc, Tensor(input))
        add_input(desc, convert(Tensor{Int32}, multiples))
    end
    Tensor(Operation(desc), 1)
end


"""
pad(tensor, paddings; mode="CONSTANT", name="")

Pads a tensor.

This operation pads a tensor according to the paddings you specify. paddings is an integer tensor with shape [n, 2], where n is the rank of tensor. For each dimension D of input, paddings[D, 0] indicates how many values to add before the contents of tensor in that dimension, and paddings[D, 1] indicates how many values to add after the contents of tensor in that dimension. If mode is "REFLECT" then both paddings[D, 0] and paddings[D, 1] must be no greater than tensor.dim_size(D) - 1. If mode is "SYMMETRIC" then both paddings[D, 0] and paddings[D, 1] must be no greater than tensor.dim_size(D).

The padded size of each dimension D of the output is:

paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]

Args:

tensor: A Tensor.
paddings: A Tensor of type int32.
mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC".
name: A name for the operation (optional).
Returns:

A Tensor. Has the same type as tensor.

Raises:

ValueError: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#pad
"""
@op function pad(tensor, paddings; mode="CONSTANT", name=nothing)
    local desc
    with_op_name(name, "Pad") do
        desc = NodeDescription("Pad")
        add_input(desc, Tensor(tensor))
        add_input(desc, convert(Tensor{Int32}, paddings))
    end
    # TODO pay attention to mode
    mode != "CONSTANT" && warn("pad does not yet pay attention to mode")
    Tensor(Operation(desc))
end


# Colon promotion rules -- mostly this will make Ints into constants
immutable TensorRange
    start::Tensor{Int32}
    stop::Tensor{Int32}
end
Base.first(tr::TensorRange)=tr.start
Base.last(tr::TensorRange)=tr.stop

Base.colon(x,y::Tensor) = colon(Tensor(x), y)
Base.colon(x::Tensor, y) = colon(x, Tensor(y))
Base.colon(x::Tensor,y::Tensor) = TensorRange(x,y)

#For x[[1,2,3]] etc
function Base.getindex(params::AbstractTensor, indices)
    if eltype(indices) == Bool
        boolean_mask(params, indices)
    else
        gather(params, indices)
    end
end

#for slices eg X[1:end] etc
function Base.getindex(params::AbstractTensor, indices::Vararg{Union{TensorRange, UnitRange, Colon}})
    # This function is all about slices
    # NOTE: slice is still 0 based for begins

    # TODO: Assign a name prefix to all the tensors made as art of this section, including constants
    begins = Tensor{Int32}[]
    sizes = Tensor{Int32}[]

    function proc_ind!(ind::Colon)
        push!(begins, 0)
        push!(sizes, -1) # Slice mark for go to end
    end
    function proc_ind!(ind::Union{UnitRange, TensorRange})
        #NOTE: end has now been replace with `endof(X)` or `size(X,d)` giving the actual size
        begin_ =  first(ind) - 1
        push!(begins, begin_)
        end_ = last(ind)
        push!(sizes, end_ - begin_)
    end

    for ind in indices
        proc_ind!(ind)
    end

    begins_tensor = stack(begins)
    sizes_tensor = stack(sizes)
    slice(params, begins_tensor, sizes_tensor)
end


#For x[1,2,3] etc
function Base.getindex(params::AbstractTensor, indices...)
    inds::Vector = collect(indices) # Want Vector, not tuple. Could be a vector of Tensors though
    if eltype.(inds) ⊆ (Int32, Int64)
        gather_nd(params, inds)
    else
        error("julia style indexing is not currently supported for indicies $indices")
    end
end




"""
gather(params, indices; validate_indices=true, name="")

Gather slices from params according to indices.

#args:
params: A Tensor.
indices: A Tensor. Must be one of the following types: Int32, Int64.
validate_indices: An optional bool. Defaults to true.
name: A name for the operation (optional).

#returns:
A Tensor. Has the same type as params.

indices must be an integer tensor of any dimension (usually 0-D or 1-D). Produces an output tensor with shape [indices.shape; params.shape[2:end]] where:
```
# Scalar indices
output[:, ..., :] = params[indices, :, ... :]

# Vector indices
output[i, :, ..., :] = params[indices[i], :, ... :]

# Higher rank indices
output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

If indices is a permutation and `length(indices) == params.shape[1]` then this operation will permute params accordingly.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#gather
"""
@op function gather(params, indices; validate_indices=true, name=nothing)
    local desc
    with_op_name(name, "Gather") do
        desc = NodeDescription("Gather")
        add_input(desc, Tensor(params))
        add_input(desc, Tensor(indices)-1)
        desc["validate_indices"] = validate_indices
    end
    Tensor(Operation(desc), 1)
end

"""
### `gather_nd(params, indices, name="")` {#gather_nd}

Gather values or slices from `params` according to `indices`.

`params` is a Tensor of rank `P` and `indices` is a Tensor of rank `Q`.

`indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_1, ..., d_{Q-1}, K]` where `0 < K <= P`.
indicies are 1 based.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `params`.

Produces an output tensor with shape

```
[d_1, ..., d_{Q-1}, params.shape[K], ..., params.shape[P-1]].
```

Some examples below.

Simple indexing into a matrix:
```julia
    indices = [1 1; 2 2]""
    params = ['a' 'b';  'c' 'd']
    output = ['a', 'd']
```

Slice indexing into a matrix:
```julia
    indices = [2  1]'
    params = ['a' 'b'; 'c' 'd']
    output = ['c' 'd'; 'a' 'b']
```

##### Args:
*  <b>`params`</b>: A `Tensor`. `P-D`.  The tensor from which to gather values.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    `Q-D`.  Index tensor having shape `[d_1, ..., d_{Q-1}, K]`. 1 based
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:
  A `Tensor`. Has the same type as `params`.
  `(P+Q-K-1)-D`.  Values from `params` gathered from indices given by
  `indices`.
"""
@op function gather_nd(params, indicies; name=nothing)
    local desc
    with_op_name(name, "GatherNd") do
        desc = NodeDescription("GatherNd")
        add_input(desc, Tensor(params))
        add_input(desc, convert(Tensor{Int32}, indicies) - 1)
    end
    Tensor(Operation(desc), 1)
end


"""
### `scatter_nd`
Creates a new tensor by applying sparse `updates` to individual values
 or slices within a zero tensor of the given `shape` tensor according to indices.

This operator is the inverse of the `gather_nd`
operator which extracts values or slices from a given tensor.

`shape` is a `TensorShape` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `shape`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `shape`.

`updates` is Tensor of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, shape[K], ..., shape[P-1]].
```

The simplest form of scatter is to insert individual elements in a tensor by
index. For example, say we want to insert 4 scattered elements in a rank-1
tensor with 8 elements.


In Julia, this scatter operation would look like this:

    indices = constant([5 4 2 8]')
    updates = constant([9, 10, 11, 12])
    shape = constant([8])
    scatter_nd(indices, updates, shape)

The resulting tensor would look like this:

    [0, 11, 0, 10, 9, 0, 0, 12]

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.
"""
@op function scatter_nd(indices, updates, shape; name=nothing)
    local desc
    with_op_name(name, "ScatterNd") do
        desc = NodeDescription("ScatterNd")
		add_input(desc, convert(Tensor{Int32}, indices-1))
		add_input(desc, Tensor(updates))
		add_input(desc, convert(Tensor{Int32}, shape))  # Must be same type as indicies
    end
    Tensor(Operation(desc), 1)
end

function scatter_nd(indices, updates, shape::TensorFlow.ShapeInference.TensorShape; name=nothing)
    if shape.rank_unknown || any(isnull.(shape.dims))
        error("TensorShape provided to scatter_nd not statically fully known ($shape). Consider using the dynamic `shape` operation instead of the static `get_shape` operation")
    end
    scatter_nd(indices, updates, get.(shape.dims); name=name)
end

"""
one_hot(indices, depth; on_value=Float32(1), off_value=Float32(0), axis=-1, dtype=Float32, name="")

Returns a one-hot tensor.

The locations represented by indices in indices take value on_value, while all other locations take value off_value.

on_value and off_value must have matching data types. If dtype is also provided, they must be the same data type as specified by dtype.

If on_value is not provided, it will default to the value 1 with type dtype

If off_value is not provided, it will default to the value 0 with type dtype

If the input indices is rank N, the output will have rank N+1. The new axis is created at dimension axis (default: the new axis is appended at the end).

If indices is a scalar the output shape will be a vector of length depth

If indices is a vector of length features, the output shape will be: features x depth if axis == -1 depth x features if axis == 0

If indices is a matrix (batch) with shape [batch, features], the output shape will be: batch x features x depth if axis == -1 batch x depth x features if axis == 1 depth x batch x features if axis == 0

If dtype is not provided, it will attempt to assume the data type of on_value or off_value, if one or both are passed in. If none of on_value, off_value, or dtype are provided, dtype will default to the value tf.float32

Note: If a non-numeric data type output is desired (tf.string, tf.bool, etc.), both on_value and off_value must be provided to one_hot

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#one_hot

"""
@op function one_hot(indices, depth; on_value=Float32(1), off_value=Float32(0), axis=-1, dtype=Float32, name=nothing)
    local desc
    with_op_name(name, "OneHot") do
        desc = NodeDescription("OneHot")
        add_input(desc, Tensor(indices)-1)
        add_input(desc, Tensor(Int32(depth)))
        add_input(desc, Tensor(dtype(on_value)))
        add_input(desc, Tensor(dtype(off_value)))
        desc["axis"] = axis
        desc["T"] = dtype
    end
    Tensor(Operation(desc), 1)
end


"""
dynamic_partition(data, partitions, num_partitions; name="")

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#dynamic_partition
"""
@op function dynamic_partition(data, partitions, num_partitions; name=nothing)
    local desc
    with_op_name(name, "DynamicPartition") do
        desc = NodeDescription("DynamicPartition")
        add_input(desc, data)
        add_input(desc, partitions)
        desc["num_partitions"] = Int64(num_partitions)
    end
    op = Operation(desc)
    [Tensor(op, x) for x in 1:num_partitions]
end

"""
dynamic_stitch(indices, data; name="")

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#dynamic_stitch
"""
@op function dynamic_stitch(indices, data; name=nothing)
    local desc
    with_op_name(name, "DynamicStitch") do
        desc = NodeDescription("DynamicStitch")
        add_input(desc, indices)
        add_input(desc, data)
    end
    Tensor(Operation(desc), 1)
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
@op function boolean_mask(tensor, mask; name=nothing)
    local result
    with_op_name(name, "BooleanMask") do
        indices = find(mask)  # TODO generalize to more dimensions
        squeezed = squeeze(indices, [2])
        result = tensor[squeezed]
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
    local desc
    with_op_name(name, "Transpose") do
        if perm === nothing
            r = range(Tensor, 0, limit=rank(n))
            perm = reverse(r, [true])
        end
        desc = NodeDescription("Transpose")
        add_input(desc, Tensor(n))
        add_input(desc, convert(Tensor{Int32}, perm))
    end
    Tensor(Operation(desc))
end

@op function Base.permutedims(n::AbstractTensor, perm; name=nothing)
    transpose(n, perm.-1; name=name)
end

Base.ctranspose(n::AbstractTensor) = transpose(n)
