module ShapeInference

export
get_shape,
TensorShape

using ..TensorFlow
import TensorFlow: get_input, get_attr
const tf = TensorFlow


type TensorShape <: tf.AbstractTensorShape
    dims::Vector{Nullable{Int}}
    rank_unknown::Bool
end

function TensorShape(dims::Vector{Nullable{Int}})
    TensorShape(dims, false)
end

function TensorShape(dims::Vector)
    TensorShape([Nullable{Int64}(_) for _ in dims])
end

function TensorShape(dim::Void)
    TensorShape(Nullable{Int}[], true)
end

function Base.show(io::IO, shape::TensorShape)
    get_dim_name = x->begin
        if isnull(x)
            return "?"
        else
            return string(get(x))
        end
    end
    if shape.rank_unknown
        print(io, "TensorShape[unknown]")
    else
        print(io, "TensorShape[")
        print(io, join([get_dim_name(_) for _ in shape.dims], ", "))
        print(io, "]")
    end
end

Base.copy(shape::TensorShape) = TensorShape(copy(shape.dims), shape.rank_unknown)

function Base.broadcast!(s1::TensorShape, s2::TensorShape)
    while length(s1.dims) < length(s2.dims)
        unshift!(s1.dims, Nullable(1))
    end
    while length(s2.dims) < length(s1.dims)
        unshift!(s2.dims, Nullable(1))
    end
end

const shape_cache = Dict{Tuple{String, Int}, TensorShape}()

"""
Runs shape inference to return the shape of the tensor produced by the given operation.

Note this runs *statically*. Use the `shape` operation to dynamically get the shape of an operation.
"""
function get_shape(n::TensorFlow.AbstractTensor)
    empty!(shape_cache)
    _get_shape(n)
end

function get_shape(n::TensorFlow.AbstractTensor, dim::Integer)
    shape = get_shape(n)
    if shape.rank_unknown
        error("Shape of $(n.op.name) is unknown")
    end
    if isnull(shape.dims[dim])
        error("Shape of $(n.op.name) in dim $(dim) is unknown")
    end
    get(shape.dims[dim])
end

function _get_shape(n::TensorFlow.AbstractTensor)
    t = Tensor(n)
    cache_key = (t.op.name, t.value_index)
    if haskey(shape_cache, cache_key)
        return shape_cache[cache_key]
    end
    op = t.op
    if op.op_name == "Variable"
        maybe_node = get_node_by_name("$(op.name)/Assign")
        if !isnull(maybe_node)
            node = get(maybe_node)
            return _get_shape(get_input(node, 2))
        end
    end
    if op.op_name ∈ keys(shape_inferer)
        shape = shape_inferer[op.op_name](op)[t.value_index]
    else
        shape = TensorShape(nothing)
    end
    shape_cache[cache_key] = shape
    return shape
end

function _get_shape(v::Variable)
    return _get_shape(get_input(v.assign_node, 2))
end

function to_shape(x::AbstractArray)
    TensorShape(map(to_shape, x))
end

function to_shape(x)
    if x==-1
        return Nullable{Int}()
    else
        return Nullable(x)
    end
end

const shape_inferer = Dict{String, Function}()

function register_shape(func, op_name)
    shape_inferer[op_name] = func
end

register_shape("Placeholder") do op
    graph = tf.get_def_graph()
    if haskey(graph.shapes, op.name)
        return [graph.shapes[op.name]]
    else
        [TensorShape(nothing)]
    end
end

register_shape("Const") do op
    value = get(load_const(op))
    [TensorShape([size(value)...])]
end

for func in ["Log", "Exp", "Neg", "Ceil", "Floor", "Sqrt", "Square",
    "Cos", "Sin", "Tan", "Atan", "Asin", "Acos", "Tanh",
    "Cast", "Relu", "Relu6", "Elu", "Softplus", "Softsign",
    "Softmax", "Sigmoid", "Tanh", "SparseSoftmaxCrossEntropyWithLogits",
    "LogSoftmax", "LRN", "LogicalAnd", "LogicalNot", "LogicalOr", "LogicalXor",
    "Sign"]
    register_shape(func) do op
        [_get_shape(get_input(op, 1))]
    end
end

register_shape("SparseSoftmaxCrossEntropyWithLogits") do op
    s1 = _get_shape(get_input(op, 1))
    s2 = _get_shape(get_input(op, 2))
    if s1.rank_unknown || s2.rank_unknown
        return [TensorShape(nothing)]
    end
    return [TensorShape([s1.dims[1]])]
end

for func in ["Add", "Sub", "Mul", "Div", "Pow"]
    register_shape(func) do op
        s1 = _get_shape(get_input(op, 1))
        s2 = _get_shape(get_input(op, 2))
        if s1.rank_unknown || s2.rank_unknown
            return [TensorShape(nothing)]
        end
        broadcast!(s1, s2)
        dims = Nullable{Int}[]
        for (d1, d2) in zip(s1.dims, s2.dims)
            if isnull(d1) || isnull(d2)
                push!(dims, Nullable())
            else
                push!(dims, Nullable(max(get(d1), get(d2))))
            end
        end
        return [TensorShape(dims)]
    end
end

register_shape("Transpose") do op
    [TensorShape(reverse(_get_shape(get_input(op, 1)).dims))]
end

const_cache = Dict{String, Any}()

"""
`load_const(op::Operation)`

Load an op which is literally a constant or evaluated to a constant after
a small amount of constant propogation.
"""
function load_const(op)
    if haskey(const_cache, op.name)
        return const_cache[op.name]
    end
    if op.op_name == "Const"
        value = Nullable(get_attr(op, "value", Array))
    elseif op.op_name == "Cast"
        value = load_const(get_input(op, 1))
    elseif op.op_name ∈ ("Sub", "Add")
        x1 = load_const(get_input(op, 1))
        x2 = load_const(get_input(op, 2))
        if isnull(x1) || isnull(x2)
            return Nullable()
        else
            if op.op_name == "Sub"
                value = Nullable(get(x1) - get(x2))
            elseif op.op_name == "Add"
                value = Nullable(get(x1) + get(x2))
            end
        end
    elseif op.op_name == "Shape"
        value = Nullable(_get_shape(get_input(op, 1)))
    elseif op.op_name == "Rank"
        x = _get_shape(get_input(op, 1))
        if x.rank_unknown
            value = Nullable()
        else
            value = Nullable(length(x.dims))
        end
    elseif op.op_name == "Range"
        start = load_const(get_input(op, 1))
        limit = load_const(get_input(op, 2))
        delta = load_const(get_input(op, 3))
        if any(map(isnull, [start, limit, delta]))
            value = Nullable()
        else
            value = Nullable(collect(get(start)[]:get(delta)[]:(get(limit)[]-1)))
        end
    else
        value = Nullable()
    end
    const_cache[op.name] = value
    return value
end

load_const(x::Tensor) = load_const(x.op)

register_shape("Reshape") do op
    n = get_input(op, 2)
    op = tf.get_op(n)
    maybe = load_const(op)
    if isnull(maybe)
        return [TensorShape(nothing)]
    else
        return [TensorShape(get(maybe))]
    end
end

register_shape("MatMul") do op
    shape1 = _get_shape(get_input(op, 1))
    shape2 = _get_shape(get_input(op, 2))
    if shape1.rank_unknown || shape2.rank_unknown
        return [TensorShape(nothing)]
    end

    if get_attr(op, "transpose_a", Bool)#op.attrs["transpose_a"].b
        reverse!(shape1.dims)
    end
    if get_attr(op, "transpose_b", Bool)#op.attrs["transpose_b"].b
        reverse!(shape2.dims)
    end
    return [TensorShape([shape1.dims[1], shape2.dims[2]])]
end

for func in ["Sum", "Prod", "Min", "Max", "All", "Any", "Mean"]
    register_shape(func) do op
        # TODO handle case of partial reduction
        keep_dims = get_attr(op, "keep_dims", Bool)#tf.load_proto(op.attrs["keep_dims"])
        value_shape = copy(_get_shape(get_input(op, 1)))
        reduction_dims = get_input(op, 2)
        if value_shape.rank_unknown
            return [TensorShape(nothing)]
        end
        reduction_dim_values = load_const(reduction_dims)
        if isnull(reduction_dim_values)
            if keep_dims
                return [TensorShape([Nullable{Int}() for _ in 1:length(value_shape.dims)])]
            else
                return [TensorShape(nothing)]
            end
        else
            dims = get(reduction_dim_values)+1
            if keep_dims
                for dim in dims
                    value_shape.dims[dim] = Nullable(1)
                end
            else
                to_keep = fill(true, length(value_shape.dims))
                for dim in dims
                    to_keep[dim] = false
                end
                value_shape.dims = value_shape.dims[to_keep]
            end
            return [value_shape]
        end
    end
end

register_shape("Shape") do op
    s = _get_shape(get_input(op, 1))
    return [TensorShape([s.rank_unknown ? nothing : length(s.dims)])]
end

register_shape("Concat") do op
    dim_op = load_const(get_input(op, 1))
    if isnull(dim_op)
        return [TensorShape(nothing)]
    end
    dim = get(dim_op)[] + 1

    n_tensors = tf.get_input_list_length(op, "values")
    tensors = [get_input(op, i) for i in 2:(n_tensors+1)]
    base_shape = copy(_get_shape(tensors[1]))

    if base_shape.rank_unknown
        return [TensorShape(nothing)]
    end
    base_shape.dims[dim] = Nullable(0)
    for tensor in tensors
        shape = _get_shape(tensor)
        base_shape.dims[dim] = Nullable(get(base_shape.dims[dim]) + get(shape.dims[dim]))
    end
    [base_shape]
end

register_shape("Rank") do op
    return [TensorShape([])]
end

register_shape("OneHot") do op
    indices = get_input(op, 1)
    depth = get_input(op, 2)
    maybe_depth_value = load_const(depth)
    if isnull(maybe_depth_value)
        return [TensorShape(nothing)]
    end
    depth_value = get(maybe_depth_value)[]
    indices_shape = _get_shape(indices)
    if indices_shape.rank_unknown
        return [TensorShape(nothing)]
    end
    return [TensorShape([indices_shape.dims[1], Nullable(depth_value)])]
end

register_shape("ExpandDims") do op
    return [TensorShape(nothing)]
    x = get_input(op, 1)
    x_shape = _get_shape(x)
    dim = get_input(op, 2)
    if dim.op.op_name != "Const"
        if x_shape.rank_unknown
            return [TensorShape(nothing)]
        else
            return [TensorShape([Nullable{Int}() for dim in 1:(length(x_shape.dims)+1)])]
        end
    end
    # dim_value = tf.load_proto(dim.op.attrs["value"])[1]
    dim_value = get(load_const(dim.op))[]
    insert!(x_shape.dims, dim_value, Nullable(1))
    return [x_shape]
end

function conv_sizer(widths, strides, filter_shape)
    pos = ones(length(widths))
    while true
        while true
            if pos[1] + filter_shape[1] > widths[1]
                pos[1] -= strides[1]
                break
            end
            pos[1] += strides[1]
        end
        if pos[2] + filter_shape[2] > widths[2]
            pos[2] -= strides[2]
            break
        end
        pos[2] += strides[2]
    end
    return div(pos-1, strides)+1
end

register_shape("Conv2D") do op
    #  TODO: this is sometimes off by one when padding is VALID
    input_shape = _get_shape(get_input(op, 1))
    filter_shape = _get_shape(get_input(op, 2))
    if input_shape.rank_unknown || filter_shape.rank_unknown
        return [TensorShape(nothing)]
    end
    padding = get_attr(op, "padding", String)#tf.load_proto(op.attrs["padding"])
    strides = get_attr(op, "strides", Vector{Int})#tf.load_proto(op.attrs["strides"])
    for (shape, name) in [(input_shape, "input"), (filter_shape, "filter")]
        @assert length(shape.dims) == 4 "Convolution $name must be 4D"
    end
    @assert length(strides) == 4
    dims = Nullable{Int}[]
    push!(dims, input_shape.dims[1])
    if padding == "SAME"
        for i in 1:2
            if isnull(input_shape.dims[i+1]) || isnull(filter_shape.dims[i])
                push!(dims,  Nullable{Int}())
            else
                push!(dims, ceil(get(input_shape.dims[i+1])/strides[i+1]))
            end
        end
    elseif padding == "VALID"
        if isnull(input_shape.dims[2]) || isnull(input_shape.dims[3]) || isnull(filter_shape.dims[1]) || isnull(filter_shape.dims[2])
            for i in 1:2
                push!(dims, Nullable{Int}())
            end
        else
            new_dims = conv_sizer([get(input_shape.dims[2]), get(input_shape.dims[3])], [strides[2], strides[3]], [get(filter_shape.dims[1]), get(filter_shape.dims[2])])
            for i in 1:2
                push!(dims, Nullable(new_dims[i]))
            end
        end
    end
    push!(dims, filter_shape.dims[4])
    [TensorShape(dims)]
end

register_shape("MaxPool") do op
    # TODO: also can be off by one when padding is VALID
    input_shape = _get_shape(get_input(op, 1))
    padding = get_attr(op, "padding", String)#tf.load_proto(op.attrs["padding"])
    ksize = get_attr(op, "ksize", Vector{Int})#tf.load_proto(op.attrs["ksize"])
    strides = get_attr(op, "strides", Vector{Int})#tf.load_proto(op.attrs["strides"])
    if input_shape.rank_unknown
        return [TensorShape(nothing)]
    end
    dims = Nullable{Int}[]
    push!(dims, input_shape.dims[1])
    if padding == "SAME"
        for i in 1:2
            if isnull(input_shape.dims[i+1])
                push!(dims, Nullable{Int}())
            else
                push!(dims, ceil(get(input_shape.dims[i+1])/strides[i+1]))
            end
        end
    elseif padding == "VALID"
        if isnull(input_shape.dims[2]) || isnull(input_shape.dims[3])
            for i in 1:2
                push!(dims, Nullable{Int}())
            end
        else
            new_dims = 1+conv_sizer([get(input_shape.dims[2]), get(input_shape.dims[3])], [strides[2], strides[3]], [ksize[2], ksize[3]])
            for i in 1:2
                push!(dims, Nullable(new_dims[i]))
            end
        end
    end
    push!(dims, input_shape.dims[4])
    [TensorShape(dims)]

end

register_shape("Split") do op
    num_split = get_attr(op, "num_split", Int)#op.attrs["num_split"].i
    split_dim_value = load_const(get_input(op, 1))
    if isnull(split_dim_value)
        return [TensorShape(nothing)]
    end
    split_dim_value = get(split_dim_value)[] + 1

    value_shape = copy(_get_shape(get_input(op, 2)))
    if isnull(value_shape.dims[split_dim_value])
        split_value = Nullable{Int}()
    else
        split_value = Nullable(get(value_shape.dims[split_dim_value])/num_split)
    end
    value_shape.dims[split_dim_value] = split_value
    [value_shape for i in 1:num_split]
end



register_shape("Slice") do op
    slices = get_input(op, 3)
    slice_value = load_const(slices)
    if isnull(slice_value)
        slice_shape = _get_shape(slices)
        if slice_shape.unknown_rank
            return [TensorShape(nothing)]
        else
            return [TensorShape([Nullable{Int}() for i in 1:length(slice_shape.dims)])]
        end
    else
        return [TensorShape(get(slice_value))]
    end
end



register_shape("Pad") do op
    tensor_shape = copy(_get_shape(get_input(op, 1)))
    paddings = get_input(op, 2)
    if paddings.op.op_name != "Const"
        return [TensorShape([Nullable{Int}() for dim in 1:length(tensor_shape.dims)])]
    end
    padding_value = tf.load_proto(padding.attrs["value"])  # TODO: this might be transposed
    for dim in 1:length(tensor_shape.dims)
        if isnull(tensor_shape.dims[dim])
            continue
        end
        tensor_shape.dims[dim] = Nullable(get(tensorshape.dims[dim]) + padding_value[dim, 1] + padding_value[dim, 2])
    end
    [tensor_shape]
end

register_shape("Gather") do op
    value_dims = _get_shape(get_input(op, 1))
    index_dims = _get_shape(get_input(op, 2))
    if index_dims.rank_unknown || value_dims.rank_unknown
        return [TensorShape(nothing)]
    end
    [TensorShape(vcat(index_dims.dims, value_dims.dims[2:end]))]
end

function todo_register_shape(name)
end

todo_register_shape("DynamicPartition")
todo_register_shape("DynamicStitch")
todo_register_shape("Tile")
todo_register_shape("Pack")


for func in ["RandomStandardNormal", "RandomUniform"]
    register_shape(func) do op
        shape = get_input(op, 1)
        shape_value = load_const(shape)
        if isnull(shape_value)
            [TensorShape(nothing)]
        else
            [get(shape_value)]
        end
    end
end

register_shape("AddN") do op
    inputs = [get_input(op, i) for i in 1:tf.get_input_list_length(op, "inputs")]
    if length(inputs) == 0
        [TensorShape(nothing)]
    elseif length(inputs) == 1
        [_get_shape(inputs[1])]
    else
        # TODO handle broadcasting
        [_get_shape(inputs[1])]
    end
end

end
