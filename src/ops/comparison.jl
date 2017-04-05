import Base: less

import .Ops: equal, not_equal, less_equal, greater, greater_equal, where

Base.less(x::AbstractTensor, y::AbstractTensor; kwargs...) = Ops.less(x, y; kwargs...)

const func_list = [
    (:less, :<),
    (:less_equal, :≤),
    (:greater, :>),
    (:greater_equal, :≥),
    (:equal, :(==)),
    (:not_equal, :(!=))
]

import Base: >, <, ≥, ≤

@static if VERSION < v"0.6-"
    import Base: .==, .!=, .>, .<, .≥, .≤
end

for (func, sym) in func_list
    @eval @define_binary($sym, $func)
end

@static if VERSION > v"0.6-"
    for (func, sym) in func_list
        @eval @define_broadcast($sym, $func)
    end
else
    for (func, sym) in func_list
        dotted_sym = Symbol(string(".", sym))
        @eval @define_binary($(dotted_sym), $func)
    end
end

Base.select(condition::AbstractTensor, args...; kwargs...) = Ops.select(condition, args...; kwargs...)

Base.find(input::AbstractTensor) = Ops.where(input)+1  # Convert from 0-based indices
