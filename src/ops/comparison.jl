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

const undotted_func_list = [
    (:less, :<),
    (:less_equal, :≤),
    (:greater, :>),
    (:greater_equal, :≥)
]

const func_list = copy(undotted_func_list)

import Base: >, <, ≥, ≤

if VERSION < v"0.6-"
    import Base: .==, .!=, .>, .<, .≥, .≤
    for (func_name, sym) in undotted_func_list
        push!(func_list, (func_name, Symbol(string(".", sym))))
    end
    push!(func_list, (:equal, :(.==)))
end

for (func, sym) in func_list
    @eval $sym(t1::AbstractTensor, t2::AbstractTensor) = $func(t1, t2)
    @eval $sym(t1::AbstractTensor, t2) = $func(t1, Tensor(t2))
    @eval $sym(t1, t2::AbstractTensor) = $func(Tensor(t1), t2)
end

@static if VERSION > v"0.6-"
    for (func, sym) in undotted_func_list
        @eval Base.broadcast(::typeof($sym), t1::AbstractTensor, t2::AbstractTensor) = $func(t1, t2)
        @eval Base.broadcast(::typeof($sym), t1::AbstractTensor, t2) = $func(t1, Tensor(t2))
        @eval Base.broadcast(::typeof($sym), t1, t2::AbstractTensor) = $func(Tensor(t1), t2)
    end
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
