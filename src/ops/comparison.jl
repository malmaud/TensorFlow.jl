import Base: less

import .Ops: equal, not_equal, less_equal, greater, greater_equal, where

Base.less(x::AbstractTensor, y::AbstractTensor; kwargs...) = Ops.less(x, y; kwargs...)

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
    push!(func_list, (:not_equal, :(.!=)))
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

Base.select(condition::AbstractTensor, args...; kwargs...) = Ops.select(condition, args...; kwargs...)


Base.find(input::AbstractTensor) = Ops.where(input)+1  # Convert from 0-based indices
