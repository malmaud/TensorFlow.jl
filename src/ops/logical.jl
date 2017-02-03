"""
    logical_and(x, y; name="LogicalAnd")

Returns the truth value of `x` AND `y` element-wise.

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Args:
*  `x`: A `Tensor` of type `bool`.
*  `y`: A `Tensor` of type `bool`.
*  `name`: A name for the operation (optional).

Returns:
  A `Tensor` of type `bool`.
"""
function logical_and(x, y; name="LogicalAnd")
    local desc
    with_op_name(name) do
        desc = NodeDescription("LogicalAnd")
        add_input(desc, Tensor(x))
        add_input(desc, Tensor(y))
    end
    Tensor(Operation(desc))
end

"""
    logical_not(x; name="LogicalNot")

Returns the truth value of NOT `x` element-wise.

Args:
*  `x`: A `Tensor` of type `bool`.
*  `name`: A name for the operation (optional).

Returns:
*  A `Tensor` of type `bool`.
"""
function logical_not(x; name="LogicalNot")
    local desc
    with_op_name(name) do
        desc = NodeDescription("LogicalNot")
        add_input(desc, Tensor(x))
    end
    Tensor(Operation(desc))
end

"""
    logical_or(x, y; name="LogicalOr")

Returns the truth value of `x` OR `y` element-wise.

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Args:
*  `x`: A `Tensor` of type `bool`.
*  `y`: A `Tensor` of type `bool`.
*  `name`: A name for the operation (optional).

Returns:
  A `Tensor` of type `bool`.
"""
function logical_or(x, y; name="LogicalOr")
    local desc
    with_op_name(name) do
        desc = NodeDescription("LogicalOr")
        add_input(desc, Tensor(x))
        add_input(desc, Tensor(y))
    end
    Tensor(Operation(desc))
end

"""
    logical_xor(x, y; name="LogicalXor")

Returns the truth value of `x` XOR `y` element-wise.

*NOTE*: `LogicalXor` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Args:
*  `x`: A `Tensor` of type `bool`.
*  `y`: A `Tensor` of type `bool`.
*  `name`: A name for the operation (optional).

Returns:
  A `Tensor` of type `bool`.
"""
function logical_xor(x, y; name="LogicalXor")
    local n
    with_op_name(name) do
        n = (x|y) & (~(x&y))
    end
    n
end

import Base: &, |, ~
if isdefined(Base, :⊻)
    import Base: ⊻
end

for (sym, f) in [(:&, :logical_and), (:|, :logical_or), (:⊻, :logical_xor)]
    @eval $sym(t1::AbstractTensor, t2::AbstractTensor) = $f(t1, t2)
end

~(t::AbstractTensor) = logical_not(t)
