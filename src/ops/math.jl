"""
`function add_n(inputs; name="AddN")`

Adds all input tensors element-wise.

Args:
  inputs: A list of `Tensor` objects, each with same shape and type.
  name: A name for the operation (optional).

Returns:
  A `Tensor` of same shape and type as the elements of `inputs`.

Raises:
  ValueError: If `inputs` don't all have same shape and dtype or the shape
  cannot be inferred.
"""
@op function add_n(inputs; name=nothing)
    local desc
    with_op_name(name, "AddN") do
        desc = NodeDescription("AddN")
        add_input(desc, Tensor.(inputs))
    end
    Tensor(Operation(desc))
end

"""
Returns the index with the smallest value across dimensions of a tensor.

Args:
* `n`: A `Tensor`.
* `dim`: A `Tensor`. Describes which dimension of `n` to reduce across.

Returns:
A `Tensor` of type `Int64`.
"""
@op function argmin(n::AbstractTensor, axis; name=nothing)
    local desc
    with_op_name(name, "ArgMin") do
        desc = NodeDescription("ArgMin")
        add_input(desc, Tensor(n))
        add_input(desc, convert(Tensor{Int32}, axis))
    end
    Tensor(Operation(desc), 1)
end

Base.indmin(n::AbstractTensor, dim) = argmin(n, dim-1)

"""
Returns the index with the largest value across dimensions of a tensor.

Args:
* `n`: A `Tensor`.
* `dim`: A `Tensor`. Describes which dimension of `n` to reduce across.

Returns:
A `Tensor` of type `Int64`.
"""
@op function argmax(n::AbstractTensor, axis; name=nothing)
    local desc
    with_op_name(name, "ArgMax") do
        desc = NodeDescription("ArgMax")
        add_input(desc, Tensor(n))
        add_input(desc, convert(Tensor{Int32}, axis))
    end
    Tensor(Operation(desc), 1)
end

@op Base.indmax(n::AbstractTensor, dim; name=nothing) = argmax(n, dim-1; name=name)

@op function Base.max(x::AbstractTensor, y, name=nothing)
    local desc
    with_op_name(name, "Maximum") do
        desc = NodeDescription("Maximum")
        add_input(desc, Tensor(x))
        add_input(desc, Tensor(y))
    end
    Tensor(Operation(desc))
end

@op function Base.min(x::AbstractTensor, y, name=nothing)
    local desc
    with_op_name(name, "Minimum") do
        desc = NodeDescription("Minimum")
        add_input(desc, Tensor(x))
        add_input(desc, Tensor(y))
    end
    Tensor(Operation(desc))
end


for (bin_op, jl_func_name, tf_func_name) in [
    (:+, :add, "Add"),
    (:-, :subtract, "Sub"),
    (:*, :matmul, "MatMul"),
    (:.*, :multiply, "Mul"),
    (:/, :divide, "Div"),
    (:^, :pow, "Pow")]
    @eval function $jl_func_name(n1::AbstractTensor, n2::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, lowercase($tf_func_name)) do
            n1 = Tensor(n1)
            n2 = Tensor(n2)
            desc = NodeDescription($tf_func_name)
            add_input(desc, n1)
            add_input(desc, n2)
        end
        Tensor(Operation(desc), 1)
    end
    jl_func_name == :multiply && continue  # Defined below
    @eval $bin_op(n1::AbstractTensor, n2::AbstractTensor) = $jl_func_name(n1, n2)
    @eval $bin_op(n1::AbstractTensor, n2) = $jl_func_name(n1, tf_promote(n1, n2))
    @eval $bin_op(n1, n2::AbstractTensor) = $jl_func_name(tf_promote(n2, n1), n2)
end

@static if VERSION > v"0.6-"  # Cope with changes in broadcasting in Julia 0.6
    Base.broadcast(::typeof(*), n1::AbstractTensor, n2::AbstractTensor) = multiply(n1, n2)
    Base.broadcast(::typeof(*), n1::AbstractTensor, n2) = multiply(n1, tf_promote(n1, n2))
    Base.broadcast(::typeof(*), n1, n2::AbstractTensor) = multiply(tf_promote(n2, n1), n2)
else
    .*(n1::AbstractTensor, n2::AbstractTensor) = multiply(n1, n2)
    .*(n1::AbstractTensor, n2) = multiply(n1, tf_promote(n1, n2))
    .*(n1, n2::AbstractTensor) = multiply(tf_promote(n2, n1), n2)
end

function batch_matmul(x::AbstractTensor,y::AbstractTensor; adj_x=false, adj_y=false, name=nothing)
    if tf_version() >= v"1.0.0-"
        Base.depwarn("""
        batch_matmul is deprecated. It's functionality is now subsumed by matmul.
        """)
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
    squared_difference(x, y; name=nothing)

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
function squared_difference(x, y; name=nothing)
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

# TO DO provide the aliases for Base functions
@op function matrix_solve(matrix, rhs; adjoint=false, name=nothing)
    local desc
    with_op_name(name, "MatrixSolve") do
        desc = NodeDescription("MatrixSolve")
        matrix = Tensor(matrix)
        rhs = Tensor(rhs)
        add_input(desc, matrix)
        add_input(desc, rhs)
        desc["adjoint"] = adjoint
    end
    Tensor(Operation(desc), 1)
end

@op function matrix_triangular_solve(matrix, rhs; lower=true, adjoint=false, name=nothing)
    local desc
    with_op_name(name, "MatrixTriangularSolve") do
        desc = NodeDescription("MatrixTriangularSolve")
        matrix = Tensor(matrix)
        rhs = Tensor(rhs)
        add_input(desc, matrix)
        add_input(desc, rhs)
        desc["lower"] = lower
        desc["adjoint"] = adjoint
    end
    Tensor(Operation(desc), 1)
end

@op function matrix_solve_ls(matrix, rhs; l2regularizer=0., fast=true, name=nothing)
    local desc
    with_op_name(name, "MatrixSolveLS") do
        desc = NodeDescription("MatrixSolveLS")
        matrix = Tensor(matrix)
        rhs = Tensor(rhs)
        add_input(desc, matrix)
        add_input(desc, rhs)
        desc["l2regularizer"] = l2regularizer
        desc["fast"] = fast
    end
    Tensor(Operation(desc), 1)
end

@op function self_adjoint_eig(tensor; name=nothing)
    local desc
    with_op_name(name, "SelfAdjointEig") do
        desc = NodeDescription("SelfAdjointEigV2")
        add_input(desc, Tensor(tensor))
    end
    op = Operation(desc)
    [Tensor(op, 1), Tensor(op, 2)]
end

@op function cholesky(input; name=nothing)
    local desc
    with_op_name(name, "Cholesky") do
        desc = NodeDescription("Cholesky")
        add_input(desc, Tensor(input))
    end
    Tensor(Operation(desc), 1)
end

@op function Base.cross(n1::AbstractTensor, n2::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Cross") do
        n1 = Tensor(n1)
        n2 = Tensor(n2)
        desc = NodeDescription("Cross")
        add_input(desc, n1)
        add_input(desc, n2)
    end
    Tensor(Operation(desc), 1)
end

*(x::Number, n::AbstractTensor) = x.*n    # For supporting notation like `2x`

^(n::AbstractTensor, x::Int) = invoke(^, (AbstractTensor, Any), n, x)

@static if VERSION < v"0.6-"
    .^(n::AbstractTensor, x) = n^x
else
    Base.broadcast(::typeof(^), n::AbstractTensor, x) = n^x
end

for (jl_func_name, tf_func_name) in [
    (:sign, "Sign"),
    (:negative, "Neg"),
    (:square, "Square"),
    (:shape, "Shape")]
    @eval @op function $jl_func_name(n::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, $tf_func_name) do
            n = Tensor(n)
            desc = NodeDescription($tf_func_name)
            add_input(desc, n)
        end
        Tensor(Operation(desc), 1)
    end
end

for (jl_func_name, tf_func_name) in [
    (:log, "Log"),
    (:exp, "Exp"),
    (:ceil, "Ceil"),
    (:floor, "Floor"),
    (:sqrt, "Sqrt"),
    (:abs, "Abs"),
    (:cos, "Cos"),
    (:sin, "Sin"),
    (:tan, "Tan"),
    (:atan, "Atan"),
    (:asin, "Asin"),
    (:acos, "Acos"),
    (:tanh, "Tanh"),
    (:lgamma, "Lgamma"),
    (:erf, "Erf"),
    (:erfc, "Erfc"),
    (:real, "Real"),
    (:imag, "Imag"),
    (:conj, "Conj")]
    @eval @op function Base.$jl_func_name(n::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, $tf_func_name) do
            n = Tensor(n)
            desc = NodeDescription($tf_func_name)
            add_input(desc, n)
        end
        Tensor(Operation(desc), 1)
    end
end

@op function Base.lbeta(x1::AbstractTensor, x2; name=nothing)
    local out
    with_op_name(name, "lbeta") do
        x1 = Tensor(x1)
        x2 = Tensor(x2)
        out = lgamma(x1) + lgamma(x2) - lgamma(x1 + x2)
    end
    out
end

#two arg special functions
for (jl_func_name, tf_func_name) in [
    (:zeta, "Zeta"),
    (:polygamma, "Polygamma")]
    @eval @op function Base.$jl_func_name(x::AbstractTensor, q::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, $tf_func_name) do
            x = Tensor(x)
            q = Tensor(q)
            desc = NodeDescription($tf_func_name)
            add_input(desc, x)
            add_input(desc, q)
        end
        Tensor(Operation(desc), 1)
    end
    @eval $jl_func_name(x::AbstractTensor, q) = $jl_func_name(x, tf_promote(x, q))
end

-(n::AbstractTensor) = negative(n)

@op function Base.complex(x_r::AbstractTensor, x_i::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Complex") do
        x_r = Tensor(x_r)
        x_i = Tensor(x_i)
        desc = NodeDescription("Complex")
        add_input(desc, x_r)
        add_input(desc, x_i)
    end
    Tensor(Operation(desc), 1)
end

# Matrix math

for (jl_func_name, tf_func_name) in [
    (:inv, "MatrixInverse"),
    (:det, "MatrixDeterminant"),
    (:diagm, "Diag"),
    (:diag, "MatrixDiagPart")]
    @eval @op function Base.$jl_func_name(n::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, $tf_func_name) do
            n = Tensor(n)
            desc = NodeDescription($tf_func_name)
            add_input(desc, n)
        end
        Tensor(Operation(desc), 1)
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

for reduction in [:sum, :prod, :min, :max, :mean]
    func_name = ucfirst(string(reduction))
    @eval @op function $(Symbol("segment_", reduction))(n::AbstractTensor, segment_indices; name=nothing)
        segment_indices = convert(Tensor{Int32}, segment_indices) - 1
        local desc
        with_op_name(name, string("Segment", $func_name)) do
            desc = NodeDescription("Segment"*$func_name)
            add_input(desc, Tensor(n))
            add_input(desc, Tensor(segment_indices))
        end
        Tensor(Operation(desc), 1)
    end
end
