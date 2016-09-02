import Base: less

for (func, op) in [
    (:equal, "Equal"),
    (:not_equal, "NotEqual"),
    (:less, "Less"),
    (:less_equal, "LessEqual"),
    (:greater, "Greater"),
    (:greater_equal, "GreaterEqual")]

    @eval function $func(t1::AbstractTensor, t2::AbstractTensor; name="")
        desc = NodeDescription($op, get_name(name))
        add_input(desc, Tensor(t1))
        add_input(desc, Tensor(t2))
        Tensor(Operation(desc))
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
