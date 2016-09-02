function identity(input; name="")
    desc = NodeDescription("Identity", get_name(name))
    add_input(desc, Tensor(input))
    Tensor(Operation(desc))
end

function make_tuple(tensors; name="", control_inputs=Operation[])
    group_deps = group(vcat(tensors, control_inputs)...)
    ops = Tensor[]
    name_base = get_name(name)
    for (idx, input) in enumerate(tensors)
        n = string(name_base, "_", idx)
        desc = NodeDescription("Identity", n)
        add_input(desc, input)
        push!(ops, Tensor(Operation(desc)))
    end
    ops
end

function group(tensors...; name="")
    desc = NodeDescription("NoOp", get_name(name))
    for tensor in tensors
        add_control_input(desc, tensor)
    end
    Tensor(Operation(desc))
end

function no_op(name="")
    desc = NodeDescription("NoOp", get_name(name))
    Tensor(Operation(desc))
end

@not_implemented function count_up_to()
end

function Base.cond(pred::AbstractTensor, fn1, fn2; name="")
    #  TODO add control dependencies to subgraphs
    base_name = get_name(name)
    switch1 = NodeDescription("Switch", "$base_name/switch1")
    add_input(switch1, fn1())
    add_input(switch1, Tensor(pred))
    switch2 = NodeDescription("Switch", "$base_name/switch2")
    add_input(switch2, fn2())
    add_input(switch2, pred)
    merge = NodeDescription("Merge", "$base_name/merge")
    add_input(merge, [Tensor(Operation(switch1), 2), Tensor(Operation(switch2), 1)])
    Tensor(Operation(merge), 1)
end

@not_implemented function case()
end

@not_implemented function while_loop()
end
