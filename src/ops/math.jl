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
function add_n(inputs; name="AddN")
    local desc
    with_op_name(name) do
        desc = NodeDescription("AddN")
        add_input(desc, Tensor.(inputs))
    end
    Tensor(Operation(desc))
end

function argmin(n::AbstractTensor, dim; name="ArgMin")
    local desc
    with_op_name(name) do
        desc = NodeDescription("ArgMin", get_name(name))
        add_input(desc, Tensor(n))
        add_input(desc, Tensor(convert_number(Int32,dim)))
    end
    Tensor(Operation(desc), 1)
end

Base.indmin(n::AbstractTensor, dim) = argmin(n, dim-1)

function argmax(n::AbstractTensor, dim; name="ArgMax")
    local desc
    with_op_name(name) do
        desc = NodeDescription("ArgMax", get_name(name))
        add_input(desc, Tensor(n))
        add_input(desc, Tensor(convert_number(Int32, dim)))
    end
    Tensor(Operation(desc), 1)
end

Base.indmax(n::AbstractTensor, dim) = argmax(n, dim-1)

function Base.max(x::AbstractTensor, y, name="Maximum")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Maximum")
        add_input(desc, Tensor(x))
        add_input(desc, Tensor(y))
    end
    Tensor(Operation(desc))
end

function Base.min(x::AbstractTensor, y, name="Minimum")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Minimum")
        add_input(desc, Tensor(x))
        add_input(desc, Tensor(y))
    end
    Tensor(Operation(desc))
end


for (bin_op, jl_func_name, tf_func_name) in [
    (:+, :add, "Add"),
    (:-, :sub, "Sub"),
    (:(.*), :mul, "Mul"),
    (:*, :matmul, "MatMul"),
    (:/, :div, "Div"),
    (:^, :pow, "Pow")]
    @eval function $jl_func_name(n1::AbstractTensor, n2::AbstractTensor; name=$tf_func_name)
        local desc
        with_op_name(name) do
            n1 = Tensor(n1)
            n2 = Tensor(n2)
            name = get_name(name)
            desc = NodeDescription($tf_func_name)
            add_input(desc, n1)
            add_input(desc, n2)
        end
        Tensor(Operation(desc), 1)
    end

    @eval $bin_op(n1::AbstractTensor, n2::AbstractTensor) = $jl_func_name(n1, n2)
    @eval $bin_op(n1::AbstractTensor, n2) = $jl_func_name(n1, tf_promote(n1, n2))
    @eval $bin_op(n1, n2::AbstractTensor) = $jl_func_name(tf_promote(n2, n1), n2)
end

function Base.cross(n1::AbstractTensor, n2::AbstractTensor; name="Cross")
    local desc
    with_op_name(name) do
        n1 = Tensor(n1)
        n2 = Tensor(n2)
        desc = NodeDescription("Cross")
        add_input(desc, n1)
        add_input(desc, n2)
    end
    Tensor(Operation(desc), 1)
end

*(x::Number, n::AbstractTensor) = x.*n

  # For supporting notation like `2x`
^(n::AbstractTensor, x::Int) = invoke(^, (AbstractTensor, Any), n, x)
.^(n::AbstractTensor, x::Number) = n^x

for (jl_func_name, tf_func_name) in [
    (:neg, "Neg"),
    (:square, "Square"),
    (:shape, "Shape")]
    @eval function $jl_func_name(n::AbstractTensor; name=$tf_func_name)
        local desc
        with_op_name(name) do
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
    #(:lbeta, "Lbeta"), #not working for now
    (:lgamma, "Lgamma"),
    (:erf, "Erf"),
    (:erfc, "Erfc"),
    (:real, "Real"),
    (:imag, "Imag"),
    (:conj, "Conj")]
    @eval function Base.$jl_func_name(n::AbstractTensor; name=$tf_func_name)
        local desc
        with_op_name(name) do
            n = Tensor(n)
            desc = NodeDescription($tf_func_name)
            add_input(desc, n)
        end
        Tensor(Operation(desc), 1)
    end
end

function Base.lbeta(x1::AbstractTensor, x2; name="lbeta")
    local out
    with_op_name(name) do
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
    @eval function Base.$jl_func_name(x::AbstractTensor, q::AbstractTensor; name=$tf_func_name)
        local desc
        with_op_name(name) do
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

-(n::AbstractTensor) = neg(n)

function Base.complex(x_r::AbstractTensor, x_i::AbstractTensor; name="Complex")
    local desc
    with_op_name(name) do
        x_r = Tensor(x_r)
        x_i = Tensor(x_i)
        desc = NodeDescription("Complex")
        add_input(desc, x_r)
        #desc = NodeDescription(get_def_graph(), "Imag", "$name/imag")
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
    @eval function Base.$jl_func_name(n::AbstractTensor; name=$tf_func_name)
        local desc
        with_op_name(name) do
            n = Tensor(n)
            desc = NodeDescription($tf_func_name)
            add_input(desc, n)
        end
        Tensor(Operation(desc), 1)
    end
end

# Reductions

for reduction in [:sum, :prod, :min, :max, :all, :any, :mean]
    @eval function $(Symbol("reduce_", reduction))(n::AbstractTensor; reduction_indices=nothing, keep_dims=false, name="")
        if reduction_indices == nothing
            n = Tensor(n)  # TODO: rewrite this
            name = get_name(name)
            range_start = constant(Int32(0))
            range_delta = constant(Int32(1))
            desc = NodeDescription(get_def_graph(), "Rank", "$name/rank")
            add_input(desc, n)
            rank = Tensor(Operation(desc), 1)
            desc = NodeDescription(get_def_graph(), "Range", "$name/range")
            add_input(desc, range_start)
            add_input(desc, rank)
            add_input(desc, range_delta)
            range = Tensor(Operation(desc), 1)
            desc = NodeDescription($(capitalize(reduction)), name)
            add_input(desc, n)
            add_input(desc, range)
            Tensor(Operation(desc), 1)
        else
            if isa(reduction_indices, Number)
                reduction_indices = [reduction_indices]
            end
            reduction_indices = [Int32(idx-1) for idx in reduction_indices]
            desc = NodeDescription($(capitalize(reduction)), get_name(name))
            add_input(desc, Tensor(n))
            add_input(desc, Tensor(reduction_indices))
            desc["keep_dims"] = keep_dims
            Tensor(Operation(desc), 1)
        end
    end
end
