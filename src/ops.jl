import Base: log, exp, +, -, *, /

const name_idx = Ref{Int}(1)

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
    (:neg, "Neg")]
    @eval function $jl_func_name(n::Node, name="")
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $tf_func_name, name)
        add_input(desc, Port(n, 1))
        Node(desc)
    end
end

-(n::Node) = neg(n)
