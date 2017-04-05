import .Ops:
    add_n,
    arg_min,
    arg_max,
    maximum,
    minimum,
    add,
    sub,
    mat_mul,
    mul,
    pow,
    matrix_solve,
    matrix_triangular_solve,
    matrix_solve_ls,
    cholesky,
    cross,
    neg,
    square,
    shape,
    lbeta,
    zeta,
    polygamma,
    matrix_inverse,
    matrix_determinant,
    diag,
    matrix_diag_part,
    unsorted_segment_sum,
    unsorted_segment_max,
    segment_sum,
    segment_max,
    segment_mean,
    segment_min,
    segment_prod


@op Base.indmin(n::AbstractTensor, dim; name=nothing) = Ops.arg_min(n, dim; name=name)+1

@op Base.indmax(n::AbstractTensor, dim; name=nothing) = Ops.arg_max(n, dim; name=name)+1

@op Base.max(x::AbstractTensor, y; kwargs...) = Ops.maximum(x, y; kwargs...)
@op Base.min(x::AbstractTensor, y; kwargs...) = Ops.minimum(x, y; kwargs...)

const multiply = Ops.mul
const negative = Ops.neg
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

@static if VERSION > v"0.6-"  # Cope with changes in broadcasting in Julia 0.6
    @define_broadcast(*, multiply)
else
    @define_binary(.*, multiply)
end

@op function batch_matmul(x::AbstractTensor,y::AbstractTensor; adj_x=false, adj_y=false, name=nothing)
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
@op function squared_difference(x, y; name=nothing)
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

@op function Base.cross(n1::AbstractTensor, n2::AbstractTensor; kwargs...)
    Ops.cross(n1, n2; kwargs...)
end

*(x::Number, n::AbstractTensor) = x.*n    # For supporting notation like `2x`

^(n::AbstractTensor, x::Int) = invoke(^, Tuple{AbstractTensor, Any}, n, x)

@static if VERSION < v"0.6-"
    .^(n::AbstractTensor, x) = n^x
else
    Base.broadcast(::typeof(^), n::AbstractTensor, x) = n^x
end

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
    :lgamma,
    :erf,
    :erfc,
    :real,
    :imag,
    :sign,
    :conj,
    :round]
    @eval @op function Base.$jl_func_name(n::AbstractTensor; kwargs...)
        Ops.$jl_func_name(n; kwargs...)
    end
end

for jl_func_name in [
    :lbeta,
    :polygamma,
    :zeta]
    @eval @op function Base.$jl_func_name(x::AbstractTensor, y; kwargs...)
        Ops.$jl_func_name(x, y; kwargs...)
    end
end

-(n::AbstractTensor) = negative(n)

@op function Base.complex(x_r::AbstractTensor, x_i::AbstractTensor; kwargs...)
    Ops.complex(x_r, x_i; kwargs...)
end

# Matrix math

for (jl_func_name, tf_func_name) in [
    (:inv, :matrix_inverse),
    (:det, :matrix_determinant),
    (:diagm, :diag),
    (:diag, :matrix_diag_part)]
    @eval @op function Base.$jl_func_name(n::AbstractTensor; kwargs...)
        Ops.$tf_func_name(n; kwargs...)
    end
end

# Reductions

for reduction in [:sum, :prod, :min, :max, :all, :any, :mean]
    @eval @op function $(Symbol("reduce_", reduction))(n::AbstractTensor; axis=nothing, keep_dims=false, name=nothing)
        if name === nothing
            name = get_name("reduce")
        end
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
            axis = [Int32(idx-1) for idx in axis]
            desc = NodeDescription($(capitalize(reduction)), name)
            add_input(desc, Tensor(n))
            add_input(desc, Tensor(axis))
            desc["keep_dims"] = keep_dims
            Tensor(Operation(desc), 1)
        end
    end
end
