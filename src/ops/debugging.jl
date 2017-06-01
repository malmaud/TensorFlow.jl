import .Ops: is_finite, is_nan, is_inf

function Base.print(x::AbstractTensor, args...; kwargs...)
    Ops.print(x, args...; kwargs...)
end

for (func, name) in [
    (:is_finite, :isfinite),
    (:is_nan, :isnan),
    (:is_inf, :isinf)]
    @eval @op function Base.$name(t::AbstractTensor; name=nothing)
        $func(t; name=name)
    end
end
