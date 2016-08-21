const shape_inferer = Dict{String, Function}()

function register_shape(op_name, func)
    shape_inferer[op_name] = func
end

register_shape("Placeholder", op->begin
    if "shape" ∈ keys(op.attrs)
        [[_.size for _ in op.attrs["shape"].shape.dim]]
    else
        -1
    end
end)

register_shape("Const", op->begin
    [[_.size for _ in op.attrs["value"].tensor.tensor_shape.dim]]
end)

for func in ["Log", "Exp", "Neg", "Ceil", "Floor", "Sqrt", "Square", "Cos", "Sin", "Tan", "Atan", "Asin", "Acos", "Tanh"]
    register_shape(func, op->begin
        [get_shape(op.inputs[1])]
    end)
end

for func in ["Add", "Sub", "Mul", "Div", "Pow"]
    register_shape(func, op->begin
        s1 = get_shape(op.inputs[1])
        s2 = get_shape(op.inputs[2])
        dims = Int[]
        for (d1, d2) in zip(s1, s2)
            push!(dims, max(d1, d2))
        end
        return [dims]
    end)
end

register_shape("Transpose", op->begin
    [reverse(get_shape(op.inputs[1]))]
end)

register_shape("Reshape", op->begin
    n = op.inputs[2]
    op = Operation(n)
    if op.op_name == "Const"
        tensor = op.attrs["tensor"].tensor
        return [tensor.int_val]
    else
        return -1
    end
end)

register_shape("MatMul", op->begin
    shape1 = get_shape(op.inputs[1])
    shape2 = get_shape(op.inputs[2])
    if "transpose_a" in keys(op.attrs)
        if op.attrs["transpose_a"].b
            reverse!(shape1)
        end
    end
    if "transpose_b" in keys(op.attrs)
        if op.attrs["transpose_b"].b
            reverse!(shape2)
        end
    end
    return [[shape1[1], shape2[2]]]
end)

for func in ["Sum", "Prod", "Min", "Max", "All", "Any", "Mean"]
    register_shape(func, op->begin
        # TODO handle case of partial reduction
        return [Int[]]
    end)
end

register_shape("Shape", op->begin
    s = get_shape(op.inputs[1])
    return [[length(s)]]
end)
