import Base: less

for (func, op) in [
    (:equal, "Equal"),
    (:not_equal, "NotEqual"),
    (:less, "Less"),
    (:less_equal, "LessEqual"),
    (:greater, "Greater"),
    (:greater_equal, "GreaterEqual")]

    @eval @op function $func(t1::AbstractTensor, t2::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, $op) do
            desc = NodeDescription($op)
            add_input(desc, Tensor(t1))
            add_input(desc, Tensor(t2))
        end
        Tensor(Operation(desc), 1)
    end

end

import Base: .==, .!=, .>, .<, .≥, .≤, >, <, ≥, ≤

for (func, sym) in [
    (:equal, :.==),
    (:not_equal, :.!=),
    (:less, :.<),
    (:less_equal, :.≤),
    (:greater, :.>),
    (:greater_equal, :.≥),
    (:less, :<),
    (:less_equal, :≤),
    (:greater, :>),
    (:greater_equal, :≥)
    ]

    @eval $sym(t1::AbstractTensor, t2::AbstractTensor) = $func(t1, t2)
    @eval $sym(t1::AbstractTensor, t2) = $func(t1, Tensor(t2))
    @eval $sym(t1, t2::AbstractTensor) = $func(Tensor(t1), t2)
end

@op function Base.select(condition::AbstractTensor, t, e; name=nothing)
    local desc
    with_op_name(name, "Select") do
        desc = NodeDescription("Select")
        add_input(desc, Tensor(condition))
        add_input(desc, Tensor(t))
        add_input(desc, Tensor(e))
    end
    Tensor(Operation(desc), 1)
end

"""
Returns locations of `true` values in a boolean `Tensor`.
"""
@op function where(input; name=nothing)
    local desc
    with_op_name(name, "Where") do
        desc = NodeDescription("Where")
        add_input(desc, Tensor(input))
    end
    Tensor(Operation(desc), 1)
end

Base.find(input::AbstractTensor) = where(input)+1  # Convert from 0-based indices
