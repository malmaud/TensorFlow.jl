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
    Tensor(node, 1)
end

function constant(tensor; name="")
    name = get_name(name)
    desc = NodeDescription(get_def_graph(), "Const", name)
    tensor = RawTensor(tensor)
    desc["dtype"] = eltype(tensor)
    desc["value"] = tensor
    node = Operation(desc)
    Tensor(node, 1)
end

Base.convert(::Type{Tensor}, x::Union{Number, String}) = constant(x)
Base.convert{T<:Union{Number, String}}(::Type{Tensor}, x::Array{T}) = constant(x)

function tf_promote(t, x::Number)
    return Tensor(eltype(t)(x))
end

tf_promote(t, x) = Tensor(x)

for (bin_op, jl_func_name, tf_func_name) in [
    (:+, :add, "Add"),
    (:-, :sub, "Sub"),
    (:(.*), :mul, "Mul"),
    (:*, :matmul, "MatMul"),
    (:/, :div, "Div"),
    (:^, :pow, "Pow"),
    (:(.==), :equal, "Equal")]
    @eval function $jl_func_name(n1::AbstractTensor, n2::AbstractTensor; name="")
        n1 = Tensor(n1)
        n2 = Tensor(n2)
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $tf_func_name, name)
        add_input(desc, n1)
        add_input(desc, n2)
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
    (:cos, "Cos"),
    (:sin, "Sin"),
    (:tan, "Tan"),
    (:atan, "Atan"),
    (:asin, "Asin"),
    (:acos, "Acos"),
    (:tanh, "Tanh"),
    (:shape, "Shape"),
    (:transpose, "Transpose")]
    @eval function $jl_func_name(n::AbstractTensor; name="")
        n = Tensor(n)
        name = get_name(name)
        desc = NodeDescription($tf_func_name, name)
        add_input(desc, n)
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
            add_input(desc, Tensor(Operation(n), 1))
            add_input(desc, Tensor(Operation(reduction_indices), 1))
            desc["keep_dims"] = keep_dims
            Tensor(Operation(desc), 1)
        end
    end
end

function Base.reshape(n::AbstractTensor, dims; name="")
    dims = Int32[dims...]
    desc = NodeDescription(get_def_graph(), "Reshape",  get_name(name))
    add_input(desc, n)
    add_input(desc, Tensor(dims))
    Operation(desc)
end

function Base.fill(n::AbstractTensor, dims::AbstractTensor; name="")
    desc = NodeDescription("Fill", get_name(name))
    add_input(desc, dims)
    add_input(desc, n)
    Tensor(Operation(desc), 1)
end

function Base.fill(::Type{Tensor}, n, dims; name="")
    fill(Tensor(n), Tensor(dims); name=name)
end

convert_number(t, n) = n
convert_number(t, x::Number) =  t(x)

function Base.linspace(::Type{Tensor}, start, stop, num; name="")
    desc = NodeDescription(get_def_graph(), "LinSpace", get_name(name))
    add_input(desc, Tensor(convert_number(Float32, start)))
    add_input(desc, Tensor(convert_number(Float32, stop)))
    add_input(desc, Tensor(convert_number(Int32, num)))
    Tensor(Operation(desc), 1)
end

function Base.range(::Type{Tensor}, start; limit=nothing, delta=1, name="")
    if limit == nothing
        limit = start
        start = 0
    end
    desc = NodeDescription("Range", get_name(name))
    add_input(desc, start)
    add_input(desc, limit)
    add_input(desc, delta)
    Tensor(Operation(desc), 1)
end

function Base.rank(n::AbstractTensor; name="")
    desc = NodeDescription("Rank", get_name(name))
    add_input(desc, Tensor(n))
    Tensor(Operation(desc), 1)
end

function Base.size(n::AbstractTensor; name="")
    desc = NodeDescription(get_def_graph(), "Size", get_name(name))
    add_input(desc, Tensor(n))
    Tensor(Operation(desc), 1)
end

Base.length(::Type{Tensor}, n::AbstractTensor; name="") = size(n, name)

function Base.slice(n::AbstractTensor, begin_, size_; name="")
    desc = NodeDescription(get_def_graph(), "Slice", get_name(name))
    add_input(desc, Tensor(n))
    add_input(desc, Tensor(begin_))
    add_input(desc, Tensor(size_))
    Tensor(Operation(desc), 1)
end

function Base.split(split_dim, num_split, value::AbstractTensor; name="")
    desc = NodeDescription("Split", get_name(name))
    add_input(desc, Tensor(convert_number(Int32, split_dim-1)))
    add_input(desc, Tensor(value))
    desc["num_split"] = num_split
    op = Operation(desc)
    [Tensor(op, _) for _ in 1:num_split]
end

function concat(dim, values; name="")
    desc = NodeDescription(get_def_graph(), "Concat", get_name(name))
    add_input(desc, Tensor(convert_number(Int32, dim)))
    add_input(desc, [Tensor(_, 1) for _ in values])
    desc["N"] = length(values)
    Tensor(Operation(desc), 1)
end

Base.cat(::Type{Tensor}, dim, values...) = concat(dim-1, values)

function read_file(filename; name="")
    desc = NodeDescription("ReadFile", get_name(name))
    add_input(desc, Tensor(filename))
    Tensor(Operation(desc), 1)
end

Base.read(::Type{Tensor}, filename) = read_file(filename)

function pack(nodes; axis=0, name="")
    desc = NodeDescription("Pack", get_name(name))
    add_input(desc, [Tensor(Operation(_), 1) for _ in nodes])
    desc["N"] = length(nodes)
    desc["axis"] = axis
    Tensor(Operation(desc), 1)
end

function expand_dims(input, dim; name="")
    desc = NodeDescription("ExpandDims", get_name(name))
    add_input(desc, Tensor(input))
    add_input(desc, Tensor(convert_number(Int32,dim)))
    Tensor(Operation(desc), 1)
end


function argmin(n::AbstractTensor, dim; name="")
    desc = NodeDescription("ArgMin", get_name(name))
    add_input(desc, Tensor(n))
    add_input(desc, Tensor(convert_number(Int32,dim)))
    Tensor(Operation(desc), 1)
end

Base.indmin(n::AbstractTensor, dim) = argmin(n, dim-1)

function argmax(n::AbstractTensor, dim; name="")
    desc = NodeDescription("ArgMax", get_name(name))
    add_input(desc, Tensor(n))
    add_input(desc, Tensor(convert_number(Int32, dim)))
    Tensor(Operation(desc), 1)
end

Base.indmax(n::AbstractTensor, dim) = argmax(n, dim-1)

function Base.zeros(::Type{Tensor}, T, shape)
    constant(zeros(T, shape))
end

Base.zeros(::Type{Tensor}, shape) = zeros(Tensor, Float32, shape)

function Base.ones(::Type{Tensor}, T, shape)
    constant(zeros(T, shape))
end

Base.ones(::Type{Tensor}, shape) = ones(Tensor, Float32, shape)

function one_hot(indices, depth; on_value=Float32(1), off_value=Float32(0), axis=-1, dtype=Float32, name="")
    desc = NodeDescription("OneHot", get_name(name))
    add_input(desc, Tensor(indices))
    add_input(desc, Tensor(Int32(depth)))
    add_input(desc, Tensor(dtype(on_value)))
    add_input(desc, Tensor(dtype(off_value)))
    desc["axis"] = axis
    desc["T"] = dtype
    Tensor(Operation(desc), 1)
end

function random_uniform(shape; name="", seed=0, dtype=Float32)
    desc = NodeDescription("RandomUniform", get_name(name))
    add_input(desc, Tensor(shape))
    desc["dtype"] = dtype
    desc["seed2"] = seed
    # TODO use global seed
    desc["seed"] = 0
    Tensor(Operation(desc), 1)
end


include("nn.jl")
include("image.jl")
