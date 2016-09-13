using NullableArrays

type TensorShape
    dims::NullableArray{Int}
    rank_unknown::Bool
end

function TensorShape(dims::Vector)
    TensorShape(NullableArray(dims), false)
end

function TensorShape(dims::Vector{Nullable{Int}})
    TensorShape(dims, false)
end

function TensorShape(dims::NullableArray)
    TensorShape(dims, false)
end

function TensorShape(dim::Void)
    x = NullableArray{Int}(1)
    TensorShape(x, true)
end

function Base.show(io::IO, shape::TensorShape)
    get_dim_name = x->begin
        if isnull(x)
            return "Unknown"
        else
            return string(get(x))
        end
    end
    if shape.rank_unknown
        print(io, "Tensorshape[Unknown rank]")
    else
        print(io, "TensorShape[")
        print(io, join([get_dim_name(_) for _ in shape.dims], ", "))
        print(io, "]")
    end
end

function Base.broadcast!(s1::TensorShape, s2::TensorShape)
    while length(s1.dims) < length(s2.dims)
        unshift!(s1.dims, Nullable(1))
    end
    while length(s2.dims) < length(s1.dims)
        unshift!(s2.dims, Nullable(1))
    end
end


"""
Runs shape inference to return the shape of the tensor produced by the given operation.

Returns -1 if shape inference cannot infer a shape.

Note this runs *statically*. Use the `shape` operation to dynamically get the shape of an operation.
"""
function get_shape(n::AbstractTensor)
    t = Tensor(n)
    op = t.op
    fillin_operation(op)
    if op.op_name ∈ keys(shape_inferer)
        return shape_inferer[op.op_name](op)[t.value_index]
    else
        return TensorShape(nothing)
    end
end

function to_shape(x::AbstractArray)
    TensorShape(NullableArray(map(to_shape, x)))
end

function to_shape(x)
    if x==-1
        return Nullable{Int}()
    else
        return Nullable(x)
    end
end

const shape_inferer = Dict{String, Function}()

function register_shape(op_name, func)
    shape_inferer[op_name] = func
end

register_shape("Placeholder", op->begin
    if "shape" ∈ keys(op.attrs)
        [to_shape([_.size for _ in op.attrs["shape"].shape.dim])]
    else
        nothing
    end
end)

register_shape("Const", op->begin
    [to_shape([_.size for _ in op.attrs["value"].tensor.tensor_shape.dim])]
end)

for func in ["Log", "Exp", "Neg", "Ceil", "Floor", "Sqrt", "Square",
    "Cos", "Sin", "Tan", "Atan", "Asin", "Acos", "Tanh",
    "Cast", "Relu", "Relu6", "Elu", "Softplus", "Softsign",
    "Softmax", "Sigmoid", "Tanh", "SparseSoftmaxCrossEntropyWithLogits",
    "LogSoftmax", "LRN"]
    register_shape(func, op->begin
        [get_shape(op.inputs[1])]
    end)
end

for func in ["Add", "Sub", "Mul", "Div", "Pow"]
    register_shape(func, op->begin
        s1 = get_shape(op.inputs[1])
        s2 = get_shape(op.inputs[2])
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
    end)
end

register_shape("Transpose", op->begin
    [TensorShape(reverse(get_shape(op.inputs[1]).dims))]
end)

register_shape("Reshape", op->begin
    n = op.inputs[2]
    op = get_op(n)
    if op.op_name == "Const"
        tensor = op.attrs["tensor"].tensor
        return [TensorShape(tensor.int_val)]
    else
        return -1
    end
end)

register_shape("MatMul", op->begin
    shape1 = get_shape(op.inputs[1])
    shape2 = get_shape(op.inputs[2])
    if "transpose_a" in keys(op.attrs)
        if op.attrs["transpose_a"].b
            reverse!(shape1.dims)
        end
    end
    if "transpose_b" in keys(op.attrs)
        if op.attrs["transpose_b"].b
            reverse!(shape2.dims)
        end
    end
    return [TensorShape([shape1.dims[1], shape2.dims[2]])]
end)

for func in ["Sum", "Prod", "Min", "Max", "All", "Any", "Mean"]
    register_shape(func, op->begin
        # TODO handle case of partial reduction
        return [TensorShape(Int[])]
    end)
end

register_shape("Shape", op->begin
    s = get_shape(op.inputs[1])
    return [TensorShape([length(s)])]
end)

register_shape("Concat", op->begin
    dim_op = op.inputs[1]
    if dim_op.op.name != "Const"
        return TensorShape(nothing)
    end
    dim = dim_op.attrs["tensor"].tensor.int_val[1]
    tensors = op.inputs[2:end]
    #  TODO finish
end)

register_shape("Rank", op->begin
    return [TensorShape([])]
end)

register_shape("OneHot", op->begin
    indices = op.inputs[1]
    depth = op.indices[2]
    if depth.op.name != "Const"
        return TensorShape(nothing)
    end
    depth_value = depth.attrs["tensor"].tensor.int_val[1]
    indices_shape = get_shape(indices)
    return [TensorShape([indices_shape.dims[1], Nullable(depth_value)])]
end)

register_shape("ExpandDims", op->begin
    x = op.inputs[1]
    dim = op.inputs[2]
    if dim.op.name != "Const"
        return TensorShape(nothing)
    end
    dim_value = dim.op.attrs["tensor"].tensor.int_val[1]
    x_shape = get_shape(x)
    insert!(x_shape.dims, dim_value, Nullable(1))
    return [x_shape]
end)

register_shape("Conv2D", op->begin
    input = op.inputs[1]
    filter = op.inputs[2]
    padding = String(op["padding"].s)
    input_shape = get_shape(input)
    filter_shape = get_shape(filter)
    #  TODO finish
end)
