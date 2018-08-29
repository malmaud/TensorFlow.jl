module ShapeInference

using Nullables
using Compat
using ..TensorFlow
import TensorFlow: get_input, get_attr, TensorShape, get_shape
const tf = TensorFlow

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
        print(io, join([get_dim_name(x) for x in shape.dims], ", "))
        print(io, "]")
    end
end

Base.copy(shape::TensorShape) = TensorShape(copy(shape.dims), shape.rank_unknown)

function Base.broadcast!(s1::TensorShape, s2::TensorShape)
    while length(s1.dims) < length(s2.dims)
        pushfirst!(s1.dims, Nullable(1))
    end
    while length(s2.dims) < length(s1.dims)
        pushfirst!(s2.dims, Nullable(1))
    end
    s1, s2
end

Base.broadcast(s1::TensorShape, s2::TensorShape) = broadcast!(copy(s1), copy(s2))

"""
Perform a unification over the input tensor shapes.
Replacing unknows with knowns.
Throws a ``DimensionMismatch`` if the shapes are not compatible
"""
function unify(value_shapes::TensorShape...) ::TensorShape
    union_dims = TensorShape(nothing)
    for value_dims in value_shapes
        if union_dims.rank_unknown # Unify per unknown shape
            union_dims = copy(value_dims)
        else
            value_dims.rank_unknown && continue
            if length(value_dims.dims) != length(union_dims.dims)
                throw(DimensionMismatch("Tensors have different ranks ($(union_dims) vs $(value_dims))"))
            end
            for ii in eachindex(value_dims.dims)
                isnull(value_dims.dims[ii]) && continue
                if isnull(union_dims.dims[ii]) # Unify per unknown dimention
                    union_dims.dims[ii] = value_dims.dims[ii]
                elseif !isequal(value_dims.dims[ii], union_dims.dims[ii])
                    throw(DimensionMismatch("Tensors have incompatiable shapes ($(union_dims) vs $(value_dims))"))
                end
            end
        end
    end
    union_dims
end


const shape_cache = Dict{Tuple{String, Int}, TensorShape}()

"""
Runs shape inference to return the shape of the tensor produced by the given operation.

Note this runs *statically*. Use the `shape` operation to dynamically get the shape of an operation.
"""
function get_shape(n::tf.AbstractTensor)
    empty!(shape_cache)
    _get_shape(n)
end

function get_shape(n::tf.AbstractTensor, dim::Integer)
    shape = get_shape(n)
    if shape.rank_unknown
        error("Shape of $(n.op.name) is unknown")
    end
    if isnull(shape.dims[dim])
        error("Shape of $(n.op.name) in dim $(dim) is unknown")
    end
    get(shape.dims[dim])
end

function _get_shape(n::tf.AbstractTensor)
    t = Tensor(n)
    cache_key = (t.op.name, t.value_index)
    if haskey(shape_cache, cache_key)
        return shape_cache[cache_key]
    end
    op = t.op
    if tf.Variables.is_variable(op)
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
    return _get_shape(get_input(tf.get_op(v.assign_node), 2))
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
    [TensorShape([x.size for x in get_def(op).attr["value"].tensor.tensor_shape.dim])]
end

# Simple 1-input 1-output functions that preserve the input shape
for func in ["Log", "Exp", "Neg", "Ceil", "Floor", "Sqrt", "Square",
    "Cos", "Sin", "Tan", "Atan", "Asin", "Acos", "Tanh",
    "Round", "Cast", "Identity",
    "Relu", "Relu6", "Elu", "Softplus", "Softsign","Softmax", "Sigmoid",
    "LogSoftmax", "LRN", "LogicalAnd", "LogicalNot", "LogicalOr", "LogicalXor",
    "Sign", "Exit", "Enter", "NextIteration", "LoopCond",
    "IsFinite", "IsInf", "IsNan"]
    register_shape(func) do op
        [_get_shape(get_input(op, 1))]
    end
end

register_shape("Select") do op
    shape1 = _get_shape(get_input(op, 2))
    shape2 = _get_shape(get_input(op, 3))
    if shape1.rank_unknown || shape2.rank_unknown
        [TensorShape(nothing)]
    elseif length(shape1.dims) != length(shape2.dims)
        [TensorShape(nothing)]
    else
        if any(.!isequal.(shape1.dims, shape2.dims) .& .!isnull.(shape1.dims) .& .!isnull.(shape2.dims))
            # some non-null dimension was different
            [TensorShape(nothing)]
        else
            dims = ifelse.(isnull.(shape1.dims), shape1.dims, shape2.dims)
            [TensorShape(dims)]
        end
    end
end

register_shape("Switch") do op
    input_shape = _get_shape(get_input(op, 1))
    [input_shape, input_shape]
end

register_shape("Merge") do op
    # TODO what if inputs have different lengths?
    input_shape = _get_shape(get_input(op, 1))
    [input_shape, TensorShape(Nullable{Int}[])]
end

register_shape("SparseSoftmaxCrossEntropyWithLogits") do op
    s1 = _get_shape(get_input(op, 1))
    s2 = _get_shape(get_input(op, 2))
    if s1.rank_unknown || s2.rank_unknown
        return [TensorShape(nothing), TensorShape(nothing)]
    end
    return [TensorShape([s1.dims[1]]), copy(s1)]
end

register_shape("SoftmaxCrossEntropyWithLogits") do op
    s1 = _get_shape(get_input(op, 1))
    s2 = _get_shape(get_input(op, 2))
    dim = Nullable{Int}()
    if !s1.rank_unknown
        dim = s1.dims[1]
    end
    if !s2.rank_unknown && !isnull(s2.dims[1])
        if isnull(dim)
            dim = s2.dims[1]
        else
            get(dim) == get(s2.dims[1]) || throw(DimensionMismatch("Tensors have incompatiable shapes ($(s1) vs $(s2))"))
        end
    end

    return [TensorShape([dim])]
end

# Binary functions that broadcast
for func in ["Add", "Sub", "Mul", "Div", "Pow", "SquaredDifference", "Less",
             "LessEqual", "Greater", "GreaterEqual", "Equal", "NotEqual",
             "Maximum", "Minimum"]
    register_shape(func) do op
        s1 = _get_shape(get_input(op, 1))
        s2 = _get_shape(get_input(op, 2))
        if s1.rank_unknown || s2.rank_unknown
            return [TensorShape(nothing)]
        end
        s1, s2 = broadcast(s1, s2)
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
    input_shape = _get_shape(get_input(op, 1))

    maybe_reorder = load_const(get_input(op, 2))
    if isnull(maybe_reorder)
        [TensorShape(nothing)]
    else
        order::Vector{Int32} = get(maybe_reorder) .+ 1
        if input_shape.rank_unknown
            # We know the rank,
            # it must be the same as the number of elements in the perm
            [TensorShape(fill(Nullable{Int32}(), length(order)))]
        else
            # Ideal case
            [TensorShape(input_shape.dims[order])]
        end
    end
end

"""
    `load_const(op::Operation)`

Load an op which is literally a constant or evaluated to a constant after
a small amount of constant propogation.
"""
function load_const(op)
    op = tf.get_op(op)
    if op.op_name == "Const"
        local value
        try
            value = convert(Array, tf.load_proto(get_def(op).attr["value"]))
        catch err
            if isa(err, tf.EmptyTensorError)
                T = eltype(Tensor(op, 1))
                value = Array{T}(undef, 0)
            else
                rethrow(err)
            end
        end
        if ndims(value) == 0 && length(value) ≥ 1
            value = value[1]
        end
        value = Nullable(value)
    elseif op.op_name == "Cast"
        value = load_const(get_input(op, 1))
    elseif op.op_name ∈ ("Sub", "Add")
        x1 = load_const(get_input(op, 1))
        x2 = load_const(get_input(op, 2))
        if isnull(x1) || isnull(x2)
            return Nullable()
        else
            tf_get(x) = get(x)
            # We need this since element-wise operations between Array{..., N}
            # and Array{..., 0} raises a deprecation error.
            tf_get(x::Nullable{Array{T, 0}}) where {T} = get(x)[1]
            if op.op_name == "Sub"
                value = Nullable(tf_get(x1) .- tf_get(x2))
            elseif op.op_name == "Add"
                value = Nullable(tf_get(x1) .+ tf_get(x2))
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
    elseif op.op_name == "Stack"
        n_inputs = tf.get_input_list_length(op, "values")
        inputs = [get_input(op, i) for i in 1:n_inputs]
        maybe_vals = load_const.(inputs)
        if any(isnull.(maybe_vals))
            value = Nullable()
        else
            vals = get.(maybe_vals)
            # We only handle packing scalars for now, so bail is anything else
            # is encountered.
            if any(size.(vals) .!= ())
                value = Nullable()
            else
                value = Nullable([x[1] for x in vals])
            end
        end
    else
        value = Nullable()
    end
    return value
end

load_const(x::Tensor) = load_const(x.op)

function get_shape_from_explict_shape_input(op, input_num)
    n = get_input(op, input_num)
    op = tf.get_op(n)
    maybe = load_const(op)
    if isnull(maybe)
        return [TensorShape(nothing)]
    else
        return [TensorShape(get(maybe))]
    end
end

register_shape("Reshape") do op
    get_shape_from_explict_shape_input(op, 2)
end

register_shape("ScatterNd") do op
    get_shape_from_explict_shape_input(op, 3)
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
                return [TensorShape([Nullable{Int}() for x in 1:length(value_shape.dims)])]
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
                return [TensorShape(value_shape.dims[to_keep])]
            end
            return [value_shape]
        end
    end
end

for func in ["ArgMax", "ArgMin"]
    register_shape(func) do op
        value_shape = copy(_get_shape(get_input(op, 1)))
        reduction_dims = get_input(op, 2)
        reduction_dim_values = load_const(reduction_dims)
        if value_shape.rank_unknown || isnull(reduction_dim_values)
            return [TensorShape(nothing)]
        else
            dims = get(reduction_dim_values)+1
            to_keep = fill(true, length(value_shape.dims))
            for dim in dims
                to_keep[dim] = false
            end
            return [TensorShape(value_shape.dims[to_keep])]
        end
    end
end

register_shape("UnsortedSegmentSum") do op
    value_shape = copy(_get_shape(get_input(op, 1)))
    if value_shape.rank_unknown
        return [TensorShape(nothing)]
    end
    b = load_const(get_input(op,3))
    if isnull(b)
        value_shape.dims[1] = Nullable{Int}()
    else
        value_shape.dims[1] = b.value[1]
    end
    return [value_shape]
end

register_shape("Shape") do op
    s = _get_shape(get_input(op, 1))
    if s.rank_unknown
        return [TensorShape(nothing)]
    end
    return [TensorShape([length(s.dims)])]
end

register_shape("Concat") do op
    dim_op = load_const(get_input(op, 1))
    if isnull(dim_op)
        return [TensorShape(nothing)]
    end
    dim = get(dim_op)[] + 1

    n_tensors = tf.get_input_list_length(op, "values")
    tensors = [get_input(op, i) for i in 2:(n_tensors+1)]

    axis_length = 0
    axis_length_known = true
    shapes=TensorShape[]
    for tensor in tensors
        shape = copy(_get_shape(tensor))
        if shape.rank_unknown
            return [TensorShape(nothing)]
        end
        if isnull(shape.dims[dim])
            axis_length_known = false
        else
            axis_length += get(shape.dims[dim])
        end
        shape.dims[dim] = Nullable{Int64}() #Null it for purposes of passing unification
        push!(shapes, shape)
    end

    base_shape = unify(shapes...)
    if axis_length_known
        base_shape.dims[dim] = Nullable(axis_length)
    else
        @assert(isnull(base_shape.dims[dim])) # Should be null from unification
    end
    [base_shape]
end

register_shape("ConcatV2") do op
    n_tensors = tf.get_input_list_length(op, "values")
    tensors = [get_input(op, i) for i in 1:n_tensors]
    dim_op = load_const(get_input(op, n_tensors+1))
    if isnull(dim_op)
        return [TensorShape(nothing)]
    end
    dim = get(dim_op)[] + 1
    axis_length = 0
    axis_length_known = true
    shapes=TensorShape[]
    for tensor in tensors
        shape = copy(_get_shape(tensor))
        if shape.rank_unknown
            return [TensorShape(nothing)]
        end
        if isnull(shape.dims[dim])
            axis_length_known = false
        else
            axis_length += get(shape.dims[dim])
        end
        shape.dims[dim] = Nullable{Int64}() #Null it for purposes of passing unification
        push!(shapes, shape)
    end

    base_shape = unify(shapes...)
    if axis_length_known
        base_shape.dims[dim] = Nullable(axis_length)
    else
        @assert(isnull(base_shape.dims[dim])) # Should be null from unification
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
    x = get_input(op, 1)
    x_shape = copy(_get_shape(x))
    if x_shape.rank_unknown
        return [TensorShape(nothing)]
    end
    dim = get_input(op, 2)
    maybe_dim_value = load_const(dim)
    if !isnull(maybe_dim_value)
        dim_value = get(maybe_dim_value)[]  # [] is to dereference the Array{Int32,0} returned
        dim_value = mod(dim_value, length(x_shape.dims) + 1) + 1 #allow inserting at `end-dim`
        insert!(x_shape.dims, dim_value, Nullable(1))
        [x_shape]
    else #Non-Const dim
        # Rank is known, because it is one greater than before
        # (Assuming the `dim` was a valid operation)
        # But we do not know where it was added in
        # so all dimensions are Null since we can't know which from which
        [TensorShape(fill(Nullable{Int}(), length(x_shape.dims)+1))]
    end
end

function conv_sizer(widths, strides, filter_shape)
    pos = ones(Int64, length(widths))
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
    return div.(pos.-1, strides).+1
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
            new_dims = 1 .+ conv_sizer([get(input_shape.dims[2]), get(input_shape.dims[3])], [strides[2], strides[3]], [ksize[2], ksize[3]])
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
    input = get_input(op, 1)
    begin_ = get_input(op, 2)
    size_ = get_input(op, 3)
    input_shape = _get_shape(input)
    maybe_begin_value = load_const(begin_)
    maybe_size_value = load_const(size_)

    if isnull(maybe_begin_value) || isnull(maybe_size_value)
        if input_shape.rank_unknown
            [TensorShape(nothing)]
        else
            [TensorShape([-1 for i in 1:length(input_shape.dims)])]
        end
    else
        begin_value = get(maybe_begin_value)
        size_value = get(maybe_size_value)
        for i in 1:length(size_value)
            # -1 is equivalent to Julia's `end`
            if size_value[i] == -1
                if !isnull(input_shape.dims[i])
                    size_value[i] = get(input_shape.dims[i]) - begin_value[i]
                end
            end
        end
        out_shape = Vector{Int}(undef, length(input_shape.dims))
        for i in 1:length(out_shape)
            if size_value[i] == -1
                out_shape[i] = -1
            else
                out_shape[i] = size_value[i]
            end
        end
        [TensorShape(out_shape)]
    end
end

register_shape("Pad") do op
    tensor_shape = copy(_get_shape(get_input(op, 1)))
    paddings = get_input(op, 2)
    if paddings.op.op_name != "Const"
        return [TensorShape([Nullable{Int}() for dim in 1:length(tensor_shape.dims)])]
    end
    padding_value = convert(Array, tf.load_proto(padding.attrs["value"]))  # TODO: this might be transposed
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

register_shape("GatherNd") do op
    value_dims = _get_shape(get_input(op, 1))
    index_dims = _get_shape(get_input(op, 2))
    if index_dims.rank_unknown || value_dims.rank_unknown || isnull(index_dims.dims[end])
        return [TensorShape(nothing)]
    end
    rr=get(index_dims.dims[end])

    [TensorShape(vcat(index_dims.dims[1:end-1], value_dims.dims[rr+1:end]))]
end

function todo_register_shape(name)
end

todo_register_shape("DynamicPartition")
todo_register_shape("DynamicStitch")
todo_register_shape("Tile")

register_shape("Pack") do op
    axis = get_attr(op, "axis", Int) + 1
    packed_len = get_attr(op, "N", Int)
    packed_values = [get_input(op, i) for i in 1:tf.get_input_list_length(op, "values")]

    @assert(length(packed_values)==packed_len)
    union_dims = unify(_get_shape.(packed_values)...)
    if !union_dims.rank_unknown
       insert!(union_dims.dims, axis, Nullable(packed_len))
    end
    [union_dims]
end

register_shape("Unpack") do op
    whole_value = get_input(op, 1)
    whole_shape = _get_shape(whole_value)
    num_split = get_attr(op,"num", Int)

    if whole_shape.rank_unknown
        # We don't know the size of the input,
        # so we don't know the size of the output
        # we do know how many their will be.
        fill(TensorShape(nothing), num_split)
    else
        axis = get_attr(op, "axis", Int) + 1
        slice_shape = copy(whole_shape)
        deleteat!(slice_shape.dims, axis)
        fill(slice_shape, num_split)
    end
end

register_shape("AddN") do op
    inputs = [get_input(op, i) for i in 1:tf.get_input_list_length(op, "inputs")]
    # TODO handle broadcasting
    union_dims = unify(_get_shape.(inputs)...)
    [union_dims]
end

for func in ["RandomStandardNormal", "RandomUniform"]
    register_shape(func) do op
        shape = get_input(op, 1)
        shape_value = load_const(shape)
        if isnull(shape_value)
            [TensorShape(nothing)]
        else
            [TensorShape(get(shape_value))]
        end
    end
end

register_shape("Where") do op
    input = get_input(op, 1)
    shape = _get_shape(input)
    if shape.rank_unknown
        [TensorShape(nothing)]
    else
        [TensorShape([Nullable{Int64}(), Nullable(length(shape.dims))])]
    end
end

register_shape("Squeeze") do op
    input_shape = _get_shape(get_input(op, 1))
    squeeze_dims = get_attr(op, "squeeze_dims", Vector{Int}) .+ 1
    if input_shape.rank_unknown
        [TensorShape(nothing)]
    elseif any(squeeze_dims .> length(input_shape.dims))
        # Workaround https://github.com/JuliaLang/julia/issues/22055
        throw(BoundsError(input_shape.dims, squeeze_dims))
    else
        new_shape = copy(input_shape)
        deleteat!(new_shape.dims, squeeze_dims)
        [new_shape]
    end
end

end
