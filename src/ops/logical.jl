import .Ops: logical_and, logical_not, logical_or

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
@op function logical_xor(x, y; name=nothing)
    local n
    with_op_name(name, "LogicalXor") do
        n = (x|y) & (~(x&y))
    end
    n
end

import Base: &, |, ~, ⊻

for (sym, f) in [(:&, :logical_and), (:|, :logical_or), (:⊻, :logical_xor)]
    @eval @define_binary $sym $f
end

@define_unary (~) logical_not
