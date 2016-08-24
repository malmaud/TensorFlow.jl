import Base: log, exp, +, -, *, /, .*, .+, ./, .-, ^, .^, sin, cos, tan, asin, acos, atan, div, tanh, sqrt, floor, .==

const name_idx = Ref{Int}(1)

function capitalize(s)
    string(uppercase(s[1]), s[2:end])
end

capitalize(s::Symbol) = capitalize(string(s))

function get_name(name="")
    if length(name) > 0
        return name
    else
        name = "node$(name_idx[])"
        name_idx[] += 1
        return name
    end
end

function placeholder(dtype; name="", shape=nothing)
    name = get_name(name)
    desc = NodeDescription(get_def_graph(), "Placeholder", name)
    desc["dtype"] = dtype
    if shape !== nothing
        desc["shape"] = (shape...)
    end
    node = Operation(desc)
end

function constant(tensor; name="")
    name = get_name(name)
    desc = NodeDescription(get_def_graph(), "Const", name)
    tensor = Tensor(tensor)
    desc["dtype"] = eltype(tensor)
    desc["value"] = tensor
    node = Operation(desc)
end

Base.convert(::Type{Operation}, x::Union{Number, String}) = constant(x)
Base.convert{T<:Union{Number, String}}(::Type{Operation}, x::Array{T}) = constant(x)

function tf_promote(t, x::Number)
    return Operation(eltype(t)(x))
end

tf_promote(t, x) = Operation(x)

for (bin_op, jl_func_name, tf_func_name) in [
    (:+, :add, "Add"),
    (:-, :sub, "Sub"),
    (:(.*), :mul, "Mul"),
    (:*, :matmul, "MatMul"),
    (:/, :div, "Div"),
    (:^, :pow, "Pow"),
    (:(.==), :equal, "Equal")]
    @eval function $jl_func_name(n1::AbstractOperation, n2::AbstractOperation; name="")
        n1 = Operation(n1)
        n2 = Operation(n2)
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $tf_func_name, name)
        add_input(desc, Port(Operation(n1), 1))
        add_input(desc, Port(Operation(n2), 1))
        Operation(desc)
    end

    @eval $bin_op(n1::AbstractOperation, n2::AbstractOperation) = $jl_func_name(n1, n2)
    @eval $bin_op(n1::AbstractOperation, n2) = $jl_func_name(n1, tf_promote(n1, n2))
    @eval $bin_op(n1, n2::AbstractOperation) = $jl_func_name(tf_promote(n2, n1), n2)
end

*(x::Number, n::AbstractOperation) = x.*n



  # For supporting notation like `2x`
^(n::AbstractOperation, x::Int) = invoke(^, (AbstractOperation, Any), n, x)
.^(n::AbstractOperation, x::Number) = n^x

for (jl_func_name, tf_func_name) in [
    (:log, "Log"),
    (:exp, "Exp"),
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
    @eval function $jl_func_name(n::AbstractOperation; name="")
        n = Operation(n)
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $tf_func_name, name)
        add_input(desc, Port(n, 1))
        Operation(desc)
    end
end

-(n::AbstractOperation) = neg(n)

# Reductions

for reduction in [:sum, :prod, :min, :max, :all, :any, :mean]
    @eval function $(Symbol("reduce_", reduction))(n::AbstractOperation; reduction_indices=nothing, keep_dims=false, name="")
        if reduction_indices == nothing
            n = Operation(n)  # TODO: rewrite this
            name = get_name(name)
            range_start = constant(Int32(0))
            range_delta = constant(Int32(1))
            desc = NodeDescription(get_def_graph(), "Rank", "$name/rank")
            add_input(desc, n)
            rank = Operation(desc)
            desc = NodeDescription(get_def_graph(), "Range", "$name/range")
            add_input(desc, range_start)
            add_input(desc, rank)
            add_input(desc, range_delta)
            range = Operation(desc)
            desc = NodeDescription(get_def_graph(), $(capitalize(reduction)), name)
            add_input(desc, n)
            add_input(desc, range)
            Operation(desc)
        else
            if isa(reduction_indices, Number)
                reduction_indices = [reduction_indices]
            end
            reduction_indices = [Int32(idx-1) for idx in reduction_indices]
            desc = NodeDescription(get_def_graph(), $(capitalize(reduction)), get_name(name))
            add_input(desc, Operation(n))
            add_input(desc, Operation(reduction_indices))
            desc["keep_dims"] = keep_dims
            Operation(desc)
        end
    end
end

function Base.reshape(n::Operation, dims; name="")
    dims = Int32[dims...]
    desc = NodeDescription(get_def_graph(), "Reshape",  get_name(name))
    add_input(desc, n)
    add_input(desc, Operation(dims))
    Operation(desc)
end

function Base.fill(n::Operation, dims::Operation; name="")
    desc = NodeDescription(get_def_graph(), "Fill", get_name(name))
    add_input(desc, dims)
    add_input(desc, n)
    Operation(desc)
end

function Base.fill(::Type{Operation}, n, dims; name="")
    fill(Operation(n), Operation(dims); name=name)
end

convert_number(t, n) = n
convert_number(t, x::Number) =  t(x)

function Base.linspace(::Type{Operation}, start, stop, num; name="")
    desc = NodeDescription(get_def_graph(), "LinSpace", get_name(name))
    add_input(desc, Operation(convert_number(Float32, start)))
    add_input(desc, Operation(convert_number(Float32, stop)))
    add_input(desc, Operation(convert_number(Int32, num)))
    Operation(desc)
end

function Base.range(::Type{Operation}, start; limit=nothing, delta=1, name="")
    if limit == nothing
        limit = start
        start = 0
    end
    desc = NodeDescription(get_def_graph(), "Range", get_name(name))
    add_input(desc, start)
    add_input(desc, limit)
    add_input(desc, delta)
    Operation(desc)
end

function Base.rank(n::AbstractOperation; name="")
    desc = NodeDescription(get_def_graph(), "Rank", get_name(name))
    add_input(desc, Operation(n))
    Operation(desc)
end

function Base.size(n::AbstractOperation; name="")
    desc = NodeDescription(get_def_graph(), "Size", get_name(name))
    add_input(desc, Operation(n))
    Operation(desc)
end

Base.length(::Type{Operation}, n::AbstractOperation; name="") = size(n, name)

function Base.slice(n::AbstractOperation, begin_, size_; name="")
    desc = NodeDescription(get_def_graph(), "Slice", get_name(name))
    add_input(desc, Operation(n))
    add_input(desc, Operation(begin_))
    add_input(desc, Operation(size_))
    Operation(desc)
end

function Base.split(split_dim, num_split, value::AbstractOperation; name="")
    desc = NodeDescription(get_def_graph(), "Split", get_name(name))
    add_input(desc, Operation(convert_number(Int32, split_dim)))
    add_input(desc, Operation(value))
    desc["num_split"] = num_split
    Operation(desc)
end

function concat(dim, values; name="")
    desc = NodeDescription(get_def_graph(), "Concat", get_name(name))
    add_input(desc, Operation(convert_number(Int32, dim)))
    add_input(desc, [Port(_, 1) for _ in values])
    desc["N"] = length(values)
    Operation(desc)
end

Base.cat(::Type{Operation}, dim, values...) = concat(dim-1, values)

function read_file(filename; name="")
    desc = NodeDescription(get_def_graph(), "ReadFile", get_name(name))
    add_input(desc, Operation(filename))
    Operation(desc)
end

Base.read(::Type{Operation}, filename) = read_file(filename)

function pack(nodes; axis=0, name="")
    desc = NodeDescription(get_def_graph(), "Pack", get_name(name))
    add_input(desc, [Port(Operation(_), 1) for _ in nodes])
    desc["N"] = length(nodes)
    desc["axis"] = axis
    Operation(desc)
end

function expand_dims(input, dim; name="")
    desc = NodeDescription(get_def_graph(), "ExpandDims", get_name(name))
    add_input(desc, Operation(input))
    add_input(desc, Operation(convert_number(Int32,dim)))
    Operation(desc)
end


function argmin(n::AbstractOperation, dim; name="")
    desc = NodeDescription(get_def_graph(), "ArgMin", get_name(name))
    add_input(desc, Operation(n))
    add_input(desc, Operation(convert_number(Int32,dim)))
    Operation(desc)
end

Base.indmin(n::AbstractOperation, dim) = argmin(n, dim-1)

function argmax(n::AbstractOperation, dim; name="")
    desc = NodeDescription(get_def_graph(), "ArgMax", get_name(name))
    add_input(desc, Operation(n))
    add_input(desc, Operation(convert_number(Int32, dim)))
    Operation(desc)
end

Base.indmax(n::AbstractOperation, dim) = argmax(n, dim-1)

function Base.zeros(::Type{Operation}, T, shape)
    constant(zeros(T, shape))
end

Base.zeros(::Type{Operation}, shape) = zeros(Operation, Float32, shape)

function Base.ones(::Type{Operation}, T, shape)
    constant(zeros(T, shape))
end

Base.ones(::Type{Operation}, shape) = ones(Operation, Float32, shape)

function one_hot(indices, depth; on_value=Float32(1), off_value=Float32(0), axis=-1, dtype=Float32, name="")
    desc = NodeDescription("OneHot", get_name(name))
    add_input(desc, Operation(indices))
    add_input(desc, Operation(Int32(depth)))
    add_input(desc, Operation(dtype(on_value)))
    add_input(desc, Operation(dtype(off_value)))
    desc["axis"] = axis
    desc["T"] = dtype
    Operation(desc)
end

function random_uniform(shape; name="", seed=0, dtype=Float32)
    desc = NodeDescription("RandomUniform", get_name(name))
    add_input(desc, Operation(shape))
    desc["dtype"] = dtype
    desc["seed2"] = seed
    # TODO use global seed
    desc["seed"] = 0
    Operation(desc)
end

function Base.split(split_dim, num_split, value::AbstractOperation; name="")
    # TODO get this to return multiple tensors
    desc = NodeDescription("Split", get_name(name))
    add_input(desc, Operation(convert_number(Int32, split_dim))-1)
    add_input(desc, value)
    desc["num_split"] = Int64(num_split)
    Operation(desc)
end


include("nn.jl")
include("image.jl")
