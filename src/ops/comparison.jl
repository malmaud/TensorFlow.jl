#import InteractiveUtils #less

import .Ops: equal, not_equal, less_equal, greater, greater_equal, where

less(x::AbstractTensor, y::AbstractTensor; kwargs...) = Ops.less(x, y; kwargs...)

const func_list = [
    (:less, :<),
    (:less_equal, :≤),
    (:greater, :>),
    (:greater_equal, :≥),
    #(:equal, :(==)),  # Don't want to have equal return a non-bool
    (:not_equal, :(!=))
]

import Base: >, <, ≥, ≤, >=, <=, !=, ==

const OP_VERSION_CHANGE  = v"0.6.0-dev.1632"

for (func, sym) in func_list
    @eval @define_binary($sym, $func)
    @eval @define_broadcast($sym, $func)
end

@define_broadcast(==, equal)

function select(condition::AbstractTensor, args...; kwargs...)
    Ops.select(condition, args...; kwargs...)
end

Base.findall(input::AbstractTensor) = Ops.where(input)+1  # Convert from 0-based indices
