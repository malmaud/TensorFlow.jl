import LinearAlgebra
import SpecialFunctions
import Statistics

import .Ops:
    add_n,
    arg_min,
    arg_max,
    add,
    sub,
    mat_mul,
    mul,
    pow,
    matrix_solve,
    matrix_triangular_solve,
    matrix_solve_ls,
    cholesky,
    neg,
    square,
    shape,
    unsorted_segment_sum,
    unsorted_segment_max,
    segment_sum,
    segment_max,
    segment_mean,
    segment_min,
    segment_prod


@op Base.argmin(n::AbstractTensor, dim; name = nothing) = Ops.arg_min(n, dim; name = name) + 1

@op Base.argmax(n::AbstractTensor, dim; name = nothing) = Ops.arg_max(n, dim; name = name) + 1

@op Base.max(x::AbstractTensor, y; kwargs...) = Ops.maximum(x, y; kwargs...)
@op Base.min(x::AbstractTensor, y; kwargs...) = Ops.minimum(x, y; kwargs...)


@op function LinearAlgebra.svd(a::AbstractTensor; full = false, kwargs...)
    # Match Base names and ordering of results
    s, u, v = Ops.svd(a; compute_uv = true, full_matrices = full, kwargs...)
    u, s, v
end



@define_unary negative Ops.neg
@define_binary multiply Ops.mul
const self_adjoint_eig = Ops.self_adjoint_eig_v2

for (bin_op, jl_func_name) in [
    (:+, :add),
    (:-, :sub),
    (:*, :mat_mul),
    (:/, :(Ops.div)),
    (:^, :pow)]

    @eval @define_binary($bin_op, $jl_func_name)
end


const matmul = mat_mul

@define_broadcast(*, mul)
@define_broadcast(+, add)
@define_broadcast(-, sub)
@define_broadcast(/, Ops.div)
@define_broadcast(^, pow)

Broadcast.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::AbstractTensor, y::Val{T}) where T = x^Tensor(T)

@op function batch_matmul(x::AbstractTensor, y::AbstractTensor; adj_x = false, adj_y = false, name = nothing)
    if tf_version() >= v"1.0.0-"
        Base.depwarn("""
        batch_matmul is deprecated. Its functionality is now subsumed by matmul.
        """, :batch_matmul)
    end
    local desc
    with_op_name(name, "BatchMatMul") do
        x = Tensor(x)
        y = Tensor(y)
        desc = NodeDescription("BatchMatMul")
        add_input(desc, x)
        add_input(desc, y)
        desc["adj_x"] = adj_x
        desc["adj_y"] = adj_y
    end
    Tensor(Operation(desc), 1)
end

"""
    squared_difference(x, y)

Returns (x - y)(x - y) element-wise.

*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Args:
  x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
  y: A `Tensor`. Must have the same type as `x`.
  name: A name for the operation (optional).

Returns:
  A `Tensor`. Has the same type as `x`.
 
"""
@op function squared_difference(x, y; name = nothing)
    local desc
    with_op_name(name, "SquaredDifference") do
        x = Tensor(x)
        y = Tensor(y)
        desc = NodeDescription("SquaredDifference")
        add_input(desc, x)
        add_input(desc, y)
    end
    Tensor(Operation(desc), 1)
end

@op function LinearAlgebra.cross(n1::AbstractTensor, n2::AbstractTensor; kwargs...)
    Ops.cross(n1, n2; kwargs...)
end

*(x::Number, n::AbstractTensor) = x .* n    # For supporting notation like `2x`

^(n::AbstractTensor, x::Int) = invoke(^, Tuple{AbstractTensor,Any}, n, x)

for jl_func_name in [
    :log,
    :exp,
    :ceil,
    :floor,
    :sqrt,
    :abs,
    :cos,
    :sin,
    :tan,
    :atan,
    :asin,
    :acos,
    :tanh,
    :real,
    :imag,
    :sign,
    :conj,
    :round]
    @eval @define_unary Base.$jl_func_name Ops.$jl_func_name
    @eval @define_broadcast_unary Base.$jl_func_name
end


for jl_func_name in [
    :lgamma,
    :erf,
    :erfc]
    @eval @define_unary SpecialFunctions.$jl_func_name Ops.$jl_func_name
    @eval @define_broadcast_unary SpecialFunctions.$jl_func_name
end

for jl_func_name in [
    :polygamma,
    :zeta]
    @eval @define_binary SpecialFunctions.$jl_func_name Ops.$jl_func_name
    @eval @define_broadcast_binary SpecialFunctions.$jl_func_name
end

function Base.round(::Type{T}, value::AbstractTensor) where T
    convert(Tensor{T}, round(value))
end


@op function -(n::AbstractTensor; kwargs...)
    negative(n; kwargs...)
end

@op function Base.complex(x_r::AbstractTensor, x_i::AbstractTensor; kwargs...)
    Ops.complex(x_r, x_i; kwargs...)
end

# Matrix math

@define_unary Base.inv Ops.matrix_inverse

for (jl_func_name, tf_func_name) in [
    (:det, :matrix_determinant),
    (:diag, :matrix_diag_part)]

    @eval @define_unary LinearAlgebra.$jl_func_name Ops.$tf_func_name
end

function LinearAlgebra.diagm(kv::Pair{T,S}) where {T <: Integer,S <: AbstractTensor}
    if kv.first == 0
        return Ops.diag(kv.second)
    end
    error("diagm only supports the calling form diagm(0=>x) where 'x' is a tensor.")
end

# Reductions

# TODO Clean this up
for reduction in [:sum, :prod, :min, :max, :all, :any, :mean]
    @eval @op function $(Symbol("reduce_", reduction))(n::AbstractTensor; axis = nothing, keep_dims = false, name = nothing)
        if name === nothing
            name = get_name("reduce")
        end
        if in_eager_mode()
            if axis === nothing
                n_value = convert(Array, n)  # TODO use shape functions instead
                num_axis = length(size(n_value))
                axis = Ops.range(constant(0), constant(num_axis), constant(1))
                fn = Ops.$reduction
                fn(n, axis, keep_dims = keep_dims)
            end  # TODO else case
        else
            if axis == nothing
                n = Tensor(n)  # TODO: rewrite this
                range_start = constant(Int32(0))
                range_delta = constant(Int32(1))
                desc = NodeDescription("Rank", "$name/rank")
                add_input(desc, n)
                rank = Tensor(Operation(desc), 1)
                desc = NodeDescription("Range", "$name/range")
                add_input(desc, range_start)
                add_input(desc, rank)
                add_input(desc, range_delta)
                range = Tensor(Operation(desc), 1)
                desc = NodeDescription($(capitalize(reduction)), name)
                add_input(desc, n)
                add_input(desc, range)
                Tensor(Operation(desc), 1)
            else
                if isa(axis, Number)
                    axis = [axis]
                end
                axis = [Int32(idx - 1) for idx in axis]
                desc = NodeDescription($(capitalize(reduction)), name)
                add_input(desc, Tensor(n))
                add_input(desc, Tensor(axis))
                desc["keep_dims"] = keep_dims
                Tensor(Operation(desc), 1)
            end
        end
    end
end

# TODO Match Julia reduction behavior when `axis` is passed
for (jl_func, tf_func) in [
    (:(Base.sum), :reduce_sum),
    (:(Base.prod), :reduce_prod),
    (:(Base.minimum), :reduce_min),
    (:(Base.maximum), :reduce_max),
    (:(Base.all), :reduce_all),
    (:(Base.any), :reduce_any),
    (:(Statistics.mean), :reduce_mean),
    ]
    @eval function $jl_func(n::AbstractTensor, axis = nothing; kwargs...)
        $tf_func(n; axis = axis, kwargs...)
    end
end
