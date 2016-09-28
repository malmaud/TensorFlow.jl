import Base: log, exp, abs, +, -, *, /, .*, .+, ./, .-, ^, .^, sin, cos, tan, asin, acos, atan, div, tanh, sqrt, floor, .==, ceil, floor

function tf_promote(t, x::Number)
    return Tensor(eltype(t)(x))
end

tf_promote(t, x) = Tensor(x)


convert_number(t, n) = n
convert_number(t, x::Number) =  t(x)

to_tensor(x::Union{Number, String, AbstractTensor}) = Tensor(x)
to_tensor(x::AbstractArray) = Tensor(x)

to_list(x::AbstractArray) = x
to_list(x) = [x]

macro not_implemented(f)
    if f.head != :function
        error("Invalid use of not_implemented")
    end
    func_name = f.args[1].args[1]
    quote
        function $(esc(func_name))(args...; kwargs...)
            error("Not implemented yet")
        end
    end
end

const name_idx = Dict{String,Int}()

function capitalize(s)
    string(uppercase(s[1]), s[2:end])
end

capitalize(s::Symbol) = capitalize(string(s))

function get_name(name="node")
    if name == ""
        name = "node"
    end
    cur_idx = get(name_idx, name, 1)
    name_idx[name] = cur_idx + 1
    if cur_idx == 1
        name
    else
        string(name, "_", cur_idx)
    end
end

function placeholder(dtype; name="placeholder", shape=nothing)
    local node
    with_op_name(name) do
        graph = get_def_graph()
        desc = NodeDescription("Placeholder")
        desc["dtype"] = dtype
        node = Operation(desc)
        if shape===nothing
            graph.shapes[name] = ShapeInference.TensorShape(nothing)
        else
            dims = Nullable{Int}[]
            for dim in shape
                if dim==-1 || dim==nothing
                    push!(dims, Nullable{Int}())
                else
                    push!(dims, Nullable(dim))
                end
            end
            graph.shapes[get_cur_node_name()] = ShapeInference.TensorShape(dims)
        end
    end
    Tensor(node, 1)
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
    (:abs, "Abs"),
    (:neg, "Neg"),
    (:ceil, "Ceil"),
    (:floor, "Floor"),
    (:sqrt, "Sqrt"),
    (:square, "Square"),
    (:cos, "Cos"),
    (:sin, "Sin"),
    (:tan, "Tan"),
    (:atan, "Atan"),
    (:asin, "Asin"),
    (:acos, "Acos"),
    (:tanh, "Tanh"),
    (:shape, "Shape"),
    (:transpose, "Transpose")]
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

function read_file(filename; name="ReadFile")
    local desc
    with_op_name(name) do
        desc = NodeDescription("ReadFile")
        add_input(desc, Tensor(filename))
    end
    Tensor(Operation(desc), 1)
end

Base.read(::Type{Tensor}, filename) = read_file(filename)

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

include("ops/sequences.jl")
include("ops/control_flow.jl")
include("ops/logical.jl")
include("ops/comparison.jl")
include("ops/transformations.jl")
include("ops/nn.jl")
include("ops/image.jl")
include("ops/summaries.jl")
include("ops/queues.jl")
