function logical_and(x, y; name="")
    desc = NodeDescription("LogicalAnd", get_name(name))
    add_input(desc, Tensor(x))
    add_input(desc, Tensor(y))
    Tensor(Operation(desc))
end

function logical_not(x; name="")
    desc = NodeDescription("LogicalNot", get_name(name))
    add_input(desc, Tensor(x))
    Tensor(Operation(desc))
end

function logical_or(x, y; name="")
    desc = NodeDescription("LogicalOr", get_name(name))
    add_input(desc, Tensor(x))
    add_input(desc, Tensor(y))
    Tensor(Operation(desc))
end

function logical_xor(x, y; name="")
    desc = NodeDescription("LogicalXor", get_name(name))
    add_input(desc, Tensor(x))
    add_input(desc, Tensor(y))
    Tensor(Operation(desc))
end

import Base: &, |, ~

for (sym, f) in [(:&, :logical_and), (:|, :logical_or)]
    @eval $sym(t1::AbstractTensor, t2::AbstractTensor) = $f(t1, t2)
end

~(t::AbstractTensor) = logical_not(t)
