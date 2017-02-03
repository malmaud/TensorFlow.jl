
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
function cast(x::Tensor, dtype; name="Cast")
    local desc
    with_op_name(name) do
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
Base.reshape(n::AbstractTensor, dims; name="Reshape") =
  reshape(n, Tensor(Int32[dims...]); name = name)

function Base.reshape(n::AbstractTensor, dims::AbstractTensor; name="Reshape")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Reshape")
        add_input(desc, n)
        add_input(desc, dims)
    end
    Tensor(Operation(desc), 1)
end

Base.length(::Type{Tensor}, n::AbstractTensor; name="") = size(n, name)

if isdefined(Base, :slice)  # Removed in .6
    import Base: slice
end
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
function slice(n::AbstractTensor, begin_, size_; name="Slice")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Slice")
        add_input(desc, Tensor(n))
        add_input(desc, cast(Tensor(begin_), Int32))
        add_input(desc, cast(Tensor(size_), Int32))
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
function Base.split(split_dim, num_split, value::AbstractTensor; name="Split")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Split")
        add_input(desc, Tensor(convert_number(Int32, split_dim))-1)
        add_input(desc, Tensor(value))
        desc["num_split"] = num_split
    end
    op = Operation(desc)
    [Tensor(op, _) for _ in 1:num_split]
end

"""
concat(dim, values; name="")

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#concat
"""
function concat(dim, values; name="Concat")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Concat")
        add_input(desc, Tensor(convert_number(Int32, dim)))
        add_input(desc, [Tensor(_) for _ in values])
        desc["N"] = length(values)
    end
    Tensor(Operation(desc), 1)
end

Base.cat(::Type{Tensor}, dim, values...) = concat(dim-1, values)

"""
pack(values; axis=1, name="")

Packs a list of rank-R tensors into one rank-(R+1) tensor.

Packs the list of tensors in values into a tensor with rank one higher than each tensor in values, by packing them along the axis dimension. Given a list of length N of tensors of shape (A, B, C);

if axis == 1 then the output tensor will have the shape (N, A, B, C). if axis == 2 then the output tensor will have the shape (A, N, B, C). Etc.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#pack
"""
function pack(nodes; axis=1, name="Pack")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Pack")
        add_input(desc, [Tensor(_) for _ in nodes])
        desc["N"] = length(nodes)
        desc["axis"] = axis -1
    end
    Tensor(Operation(desc), 1)
end

function unpack(value; num=nothing, axis=1, name="Unpack")
    num_split = num==nothing ? get_shape(value, axis) : num
    local desc
    with_op_name(name) do
        desc = NodeDescription("Unpack")
        add_input(desc, value)
        desc["num"] = num_split
        desc["axis"] = axis - 1
    end
    op = Operation(desc)
    [Tensor(op, _) for _ in 1:num_split]
end

"""
expand_dims(input, dim; name="")

Inserts a dimension of 1 into a tensor's shape.

Given a tensor input, this operation inserts a dimension of 1 at the dimension index dim of input's shape. The dimension index dim starts at zero; if you specify a negative number for dim it is counted backward from the end.

This operation is useful if you want to add a batch dimension to a single element. For example, if you have a single image of shape [height, width, channels], you can make it a batch of 1 image with expand_dims(image, 0), which will make the shape [1, height, width, channels].

Other examples:

# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
This operation requires that:

-1-input.dims() <= dim <= input.dims()

This operation is related to squeeze(), which removes dimensions of size 1.

Args:

input: A Tensor.
dim: A Tensor of type int32. 0-D (scalar). Specifies the dimension index at which to expand the shape of input.
name: A name for the operation (optional).
Returns:

A Tensor. Has the same type as input. Contains the same data as input, but its shape has an additional dimension of size 1 added.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#expand_dims
"""
function expand_dims(input, dim; name="ExpandDims")
    local desc
    with_op_name(name) do
        desc = NodeDescription("ExpandDims")
        add_input(desc, Tensor(input))
        add_input(desc, Tensor(convert_number(Int32,dim)))
    end
    Tensor(Operation(desc), 1)
end

function Base.squeeze(x::AbstractTensor, squeeze_dims; name="squeeze")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Squeeze")
        add_input(desc, x)
        set_attr_list(desc, "squeeze_dims", squeeze_dims-1)
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
function Base.rank(n::AbstractTensor; name="Rank")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Rank")
        add_input(desc, Tensor(n))
    end
    Tensor(Operation(desc), 1)
end

"""
Base.size(n::AbstractTensor; name="")

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#size
"""
function Base.size(n::AbstractTensor; name="Size")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Size")
        add_input(desc, Tensor(n))
    end
    Tensor(Operation(desc), 1)
end

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
function tile(input, multiples; name="Tile")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Tile")
        add_input(desc, Tensor(input))
        add_input(desc, cast(Tensor(multiples), Int32))
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
function pad(tensor, paddings; mode="CONSTANT", name="Pad")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Pad")
        add_input(desc, Tensor(tensor))
        add_input(desc, cast(Tensor(paddings), Int32))
    end
    # TODO pay attention to mode
    mode != "CONSTANT" && warn("pad does not yet pay attention to mode")
    Tensor(Operation(desc))
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

# Scalar indices
output[:, ..., :] = params[indices, :, ... :]

# Vector indices
output[i, :, ..., :] = params[indices[i], :, ... :]

# Higher rank indices
output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
If indices is a permutation and length(indices) == params.shape[1] then this operation will permute params accordingly.

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#gather
"""
function gather(params, indices; validate_indices=true, name="Gather")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Gather")
        add_input(desc, Tensor(params))
        add_input(desc, Tensor(indices)-1)
        desc["validate_indices"] = validate_indices
    end
    Tensor(Operation(desc), 1)
end

@not_implemented function gather_nd()
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
function one_hot(indices, depth; on_value=Float32(1), off_value=Float32(0), axis=-1, dtype=Float32, name="OneHot")
    local desc
    with_op_name(name) do
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

# function Base.reverse()
# end

"""
dynamic_partition(data, partitions, num_partitions; name="")

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#dynamic_partition
"""
function dynamic_partition(data, partitions, num_partitions; name="DynamicPartition")
    local desc
    with_op_name(name) do
        desc = NodeDescription("DynamicPartition")
        add_input(desc, data)
        add_input(desc, partitions)
        desc["num_partitions"] = Int64(num_partitions)
    end
    op = Operation(desc)
    [Tensor(op, _) for _ in 1:num_partitions]
end

"""
dynamic_stitch(indices, data; name="")

https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#dynamic_stitch
"""
function dynamic_stitch(indices, data; name="DynamicStitch")
    local desc
    with_op_name(name) do
        desc = NodeDescription("DynamicStitch")
        add_input(desc, indices)
        add_input(desc, data)
    end
    Tensor(Operation(desc), 1)
end

@not_implemented function boolean_mask(tensor, mask; name="boolean_mask")
end

"""
`transpose(n::AbstractTensor, perm=nothing)`

Transposes `a`. Permutes the dimensions according to `perm`.

The returned tensor's dimension i will correspond to the input dimension
`perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
the rank of the input tensor. Hence by default, this operation performs a
regular matrix transpose on 2-D input Tensors.

For example:

```python
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
function Base.transpose(n::AbstractTensor, perm=nothing; name="transpose")
    local desc
    with_op_name(name) do
        if perm === nothing
            r = range(Tensor, 0, limit=rank(n))
            perm = reverse(r, [true])
        end
        perm = convert_number(Int32, perm)
        desc = NodeDescription("Transpose")
        add_input(desc, Tensor(n))
        add_input(desc, Tensor(perm))
    end
    Tensor(Operation(desc))
end

function Base.permutedims(n::AbstractTensor, perm; name="transpose")
    transpose(n, perm.-1; name=name)
end

Base.ctranspose(n::AbstractTensor) = transpose(n)
