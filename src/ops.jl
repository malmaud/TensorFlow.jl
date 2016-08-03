import Base: log, exp, +, -, *, /

const name_idx = Ref{Int}(1)

function capitalize(s)
    string(uppercase(s[1]), s[2:end])
end

capitalize(s::Symbol) = capitalize(string(s))

function get_name(name)
    if length(name) > 0
        return name
    else
        name = "node$(name_idx[])"
        name_idx[] += 1
        return name
    end
end

function placeholder(dtype, name="")
    name = get_name(name)
    desc = NodeDescription(get_def_graph(), "Placeholder", name)
    desc["dtype"] = dtype
    node = Node(desc)
end

function constant(tensor, name="")
    name = get_name(name)
    desc = NodeDescription(get_def_graph(), "Const", name)
    tensor = Tensor(tensor)
    desc["dtype"] = eltype(tensor)
    desc["value"] = tensor
    node = Node(desc)
end

for (bin_op, jl_func_name, tf_func_name) in [
    (:+, :add, "Add"),
    (:-, :sub, "Sub"),
    (:(.*), :mul, "Mul"),
    (:*, :matmul, "MatMul"),
    (:/, :div, "Div"),
    (:^, :pow, "Pow")]
    @eval function $jl_func_name(n1::Node, n2::Node, name="")
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $tf_func_name, name)
        add_input(desc, Port(n1, 1))
        add_input(desc, Port(n2, 1))
        Node(desc)
    end

    @eval $bin_op(n1::Node, n2::Node) = $jl_func_name(n1, n2)
    @eval $bin_op(n1::Node, n2) = $jl_func_name(n1, constant(n2))
    @eval $bin_op(n1, n2::Node) = $jl_func_name(constant(n1), n2)
end

for (jl_func_name, tf_func_name) in [
    (:log, "Log"),
    (:exp, "Exp"),
    (:neg, "Neg"),
    (:ceil, "Ceil"),
    (:floor, "Floor"),
    (:sqrt, "Sqrt"),
    (:square, "Square"),
    (:cos, "Cos"),
    (:sin, "Sin"),
    (:tan, "Tan"),
    (:transpose, "Transpose")]
    @eval function $jl_func_name(n::Node, name="")
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $tf_func_name, name)
        add_input(desc, Port(n, 1))
        Node(desc)
    end
end

-(n::Node) = neg(n)

# Reductions

function reduce_sum(n::Node, name="")
    name = get_name(name)
    range_start = constant(Int32(0))
    range_delta = constant(Int32(1))
    desc = NodeDescription(get_def_graph(), "Rank", "$name/rank")
    rank = Node(desc)
    desc = NodeDescription(get_def_graph(), "Range", "$name/range")
    add_input(desc, range_start)
    add_input(desc, rank)
    add_input(desc, range_delta)
    range = Node(desc)
    desc = NodeDescription(get_def_graph(), "Sum", name)
    add_input(desc, n)
    add_inpnut(desc, range)
    Node(desc)
end

include("nn.jl")
