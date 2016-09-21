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


function Base.zeros(::Type{Tensor}, T, shape)
    constant(zeros(T, shape))
end

Base.zeros(::Type{Tensor}, shape) = zeros(Tensor, Float32, shape)

function Base.ones(::Type{Tensor}, T, shape)
    constant(zeros(T, shape))
end

Base.ones(::Type{Tensor}, shape) = ones(Tensor, Float32, shape)

function random_uniform(shape; name="RandomUniform", seed=0, dtype=Float32)
    local desc
    with_op_name(name) do
        desc = NodeDescription("RandomUniform")
        add_input(desc, Tensor(shape))
        desc["dtype"] = dtype
        desc["seed2"] = seed
        # TODO use global seed
        desc["seed"] = 0
    end
    Tensor(Operation(desc), 1)
end

function random_normal(shape; mean=0.0, stddev=1.0, dtype=Float32, seed=0, name="RandomNormal")
    local out
    with_op_name(name) do
        desc = NodeDescription("RandomStandardNormal")
        add_input(desc, Tensor(shape))
        desc["dtype"] = dtype
        var = Tensor(Operation(desc))
        out = stddev.*var + mean
    end
    out
end

function Base.shuffle(t::Tensor; seed=0, name="RandomShuffle")
    local desc
    with_op_name(name) do
        desc = NodeDescription("RandomShuffle")
        add_input(desc, t)
        desc["seed2"] = Int64(seed)
    end
    Tensor(Operation(desc))
end

function Base.linspace(::Type{Tensor}, start, stop, num; name="LinSpace")
    local desc
    with_op_name(name) do
        desc = NodeDescription("LinSpace")
        add_input(desc, Tensor(convert_number(Float32, start)))
        add_input(desc, Tensor(convert_number(Float32, stop)))
        add_input(desc, Tensor(convert_number(Int32, num)))
    end
    Tensor(Operation(desc), 1)
end

function Base.range(::Type{Tensor}, start; limit=nothing, delta=1, name="Range")
    if limit == nothing
        limit = start
        start = 0
    end
    local desc
    with_op_name(name) do
        desc = NodeDescription("Range")
        add_input(desc, cast(Tensor(start), Int32))
        add_input(desc, cast(Tensor(limit), Int32))
        add_input(desc, cast(Tensor(delta), Int32))
    end
    Tensor(Operation(desc), 1)
end


function Base.fill(n::AbstractTensor, dims::AbstractTensor; name="Fill")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Fill", get_name(name))
        add_input(desc, dims)
        add_input(desc, n)
    end
    Tensor(Operation(desc), 1)
end

function Base.fill(::Type{Tensor}, n, dims; name="")
    fill(Tensor(n), Tensor(dims); name=name)
end
