
"""
cast(x::Tensor, dtype; name="")
"""
function cast(x::Tensor, dtype; name="")
    desc = NodeDescription("Cast",get_name(name))
    add_input(desc, x)
    desc["DstT"] = dtype
    # desc["SrcT"] = eltype(x)
    Tensor(Operation(desc), 1)
end

"""
Base.reshape(n::AbstractTensor, dims; name="")
"""
function Base.reshape(n::AbstractTensor, dims; name="")
    dims = Int32[dims...]
    desc = NodeDescription(get_def_graph(), "Reshape",  get_name(name))
    add_input(desc, n)
    add_input(desc, Tensor(dims))
    Tensor(Operation(desc), 1)
end

Base.length(::Type{Tensor}, n::AbstractTensor; name="") = size(n, name)

"""
Base.slice(n::AbstractTensor, begin_, size_; name="")
"""
function Base.slice(n::AbstractTensor, begin_, size_; name="")
    desc = NodeDescription(get_def_graph(), "Slice", get_name(name))
    add_input(desc, Tensor(n))
    add_input(desc, cast(Tensor(begin_), Int32))
    add_input(desc, cast(Tensor(size_), Int32))
    Tensor(Operation(desc), 1)
end

@not_implemented function strided_slice()
end

"""
Base.split(split_dim, num_split, value::AbstractTensor; name="")
"""
function Base.split(split_dim, num_split, value::AbstractTensor; name="")
    desc = NodeDescription("Split", get_name(name))
    add_input(desc, Tensor(convert_number(Int32, split_dim))-1)
    add_input(desc, Tensor(value))
    desc["num_split"] = num_split
    op = Operation(desc)
    [Tensor(op, _) for _ in 1:num_split]
end

"""
concat(dim, values; name="")
"""
function concat(dim, values; name="")
    desc = NodeDescription(get_def_graph(), "Concat", get_name(name))
    add_input(desc, Tensor(convert_number(Int32, dim)))
    add_input(desc, [Tensor(_) for _ in values])
    desc["N"] = length(values)
    Tensor(Operation(desc), 1)
end

Base.cat(::Type{Tensor}, dim, values...) = concat(dim-1, values)

"""
pack(nodes; axis=0, name="")
"""
function pack(nodes; axis=0, name="")
    desc = NodeDescription("Pack", get_name(name))
    add_input(desc, [Tensor(Operation(_), 1) for _ in nodes])
    desc["N"] = length(nodes)
    desc["axis"] = axis
    Tensor(Operation(desc), 1)
end

@not_implemented function unpack()
end

"""
expand_dims(input, dim; name="")
"""
function expand_dims(input, dim; name="")
    desc = NodeDescription("ExpandDims", get_name(name))
    add_input(desc, Tensor(input))
    add_input(desc, Tensor(convert_number(Int32,dim)))
    Tensor(Operation(desc), 1)
end

"""
Base.rank(n::AbstractTensor; name="")
"""
function Base.rank(n::AbstractTensor; name="")
    desc = NodeDescription("Rank", get_name(name))
    add_input(desc, Tensor(n))
    Tensor(Operation(desc), 1)
end

"""
Base.size(n::AbstractTensor; name="")
"""
function Base.size(n::AbstractTensor; name="")
    desc = NodeDescription(get_def_graph(), "Size", get_name(name))
    add_input(desc, Tensor(n))
    Tensor(Operation(desc), 1)
end

"""
tile(input, multiples; name="")
"""
function tile(input, multiples; name="")
    desc = NodeDescription("Tile", get_name(name))
    add_input(desc, Tensor(input))
    add_input(desc, cast(Tensor(multiples), Int32))
    Tensor(Operation(desc))
end


"""
pad(tensor, paddings; mode="CONSTANT", name="")
"""
function pad(tensor, paddings; mode="CONSTANT", name="")
    desc = NodeDescription("Pad",get_name(name))
    add_input(desc, Tensor(tensor))
    add_input(desc, cast(Tensor(paddings), Int32))
    # TODO pay attention to mode
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
"""
function gather(params, indices; validate_indices=true, name="")
    desc = NodeDescription("Gather", get_name(name))
    add_input(desc, Tensor(params))
    add_input(desc, Tensor(indices)-1)
    desc["validate_indices"] = validate_indices
    Tensor(Operation(desc))
end

@not_implemented function gather_nd()
end

"""
one_hot(indices, depth; on_value=Float32(1), off_value=Float32(0), axis=-1, dtype=Float32, name="")
"""
function one_hot(indices, depth; on_value=Float32(1), off_value=Float32(0), axis=-1, dtype=Float32, name="")
    desc = NodeDescription("OneHot", get_name(name))
    add_input(desc, Tensor(indices))
    add_input(desc, Tensor(Int32(depth)))
    add_input(desc, Tensor(dtype(on_value)))
    add_input(desc, Tensor(dtype(off_value)))
    desc["axis"] = axis
    desc["T"] = dtype
    Tensor(Operation(desc), 1)
end

# function Base.reverse()
# end

"""
dynamic_partition(data, partitions, num_partitions; name="")
"""
function dynamic_partition(data, partitions, num_partitions; name="")
    desc = NodeDescription("DynamicPartition", get_name(name))
    add_input(desc, data)
    add_input(desc, partitions)
    desc["num_partitions"] = Int64(num_partitions)
    op = Operation(desc)
    [Tensor(op, _) for _ in 1:num_partitions]
end

"""
dynamic_stitch(indices, data; name="")
"""
function dynamic_stitch(indices, data; name="")
    desc = NodeDescription("DynamicStitch", get_name(name))
    add_input(desc, indices)
    add_input(desc, data)
    Tensor(Operation(desc))
end

@not_implemented function boolean_mask(tensor, mask; name="")

end
