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

*(x::Number, n::AbstractTensor) = x.*n

  # For supporting notation like `2x`
^(n::AbstractTensor, x::Int) = invoke(^, (AbstractTensor, Any), n, x)
.^(n::AbstractTensor, x::Number) = n^x


for (jl_func_name, tf_func_name) in [
    (:log, "Log"),
    (:exp, "Exp"),
    (:neg, "Neg"),
    (:ceil, "Ceil"),
    (:floor, "Floor"),
    (:sqrt, "Sqrt"),
    (:square, "Square"),
    (:abs, "Abs"),
    (:cos, "Cos"),
    (:sin, "Sin"),
    (:tan, "Tan"),
    (:atan, "Atan"),
    (:asin, "Asin"),
    (:acos, "Acos"),
    (:tanh, "Tanh"),
    (:shape, "Shape")]
    @eval function $jl_func_name(n::AbstractTensor; name=$tf_func_name)
        local desc
        with_op_name(name) do
            n = Tensor(n)
            name = get_name(name)
            desc = NodeDescription($tf_func_name, name)
            add_input(desc, n)
        end
        Tensor(Operation(desc), 1)
    end
end

-(n::AbstractTensor) = neg(n)




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
