import Base: less

for (func, op) in [
    (:equal, "Equal"),
    (:not_equal, "NotEqual"),
    (:less, "Less"),
    (:less_equal, "LessEqual"),
    (:greater, "Greater"),
    (:greater_equal, "GreaterEqual")]

    @eval function $func(t1::AbstractTensor, t2::AbstractTensor; name=$op)
        local desc
        with_op_name(name) do
            desc = NodeDescription($op)
            add_input(desc, Tensor(t1))
            add_input(desc, Tensor(t2))
        end
        Tensor(Operation(desc), 1)
    end

end

import Base: .==, .!=, .>, .<, .≥, .≤

for (func, sym) in [
    (:equal, :.==),
    (:not_equal, :.!=),
    (:less, :.<),
    (:less_equal, :.≤),
    (:greater, :.>),
    (:greater_equal, :.≥)]

    @eval $sym(t1::AbstractTensor, t2::AbstractTensor) = $func(t1, t2)

end

function Base.select(condition::AbstractTensor, t, e; name="Select")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Select")
        add_input(desc, Tensor(condition))
        add_input(desc, Tensor(t))
        add_input(desc, Tensor(e))
    end
    Tensor(Operation(desc), 1)
end

function where(input; name="Where")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Where")
        add_input(desc, Tensor(input))
    end
    Tensor(Operation(desc), 1)
end

Base.ifelse(input::AbstractTensor) = where(input)
