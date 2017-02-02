using ProtoBuf
using PyCall

import Base: setindex!, getindex, run

const LIB_BASE = joinpath(dirname(@__FILE__), "..", "deps")
const LIBTF = joinpath(LIB_BASE, "usr", "bin", "libtensorflow_c")

include("py.jl")

type Status
    ptr::Ptr{Void}
    function Status()
        ptr = ccall((:TF_NewStatus, LIBTF), Ptr{Void}, ())
        this = new(ptr)
        this
    end
end

function get_code(s::Status)
    code = ccall((:TF_GetCode, LIBTF), Cint, (Ptr{Void},), s.ptr)
    return TF_Code(code)
end

"""
A TensorFlow computation graph
"""
type Graph
    ptr::Ptr{Void}
    collections::Dict{Symbol, Any}
    shapes::Dict{String, AbstractTensorShape}

    function Graph()
        ptr = ccall((:TF_NewGraph, LIBTF), Ptr{Void}, ())
        collections = Dict{Symbol, Any}()
        collections[:Variables] = []
        collections[:TrainableVariables] = []
        collections[:Summaries] = []
        collections[:QueueRunners] = []
        self = new(ptr, collections, Dict{String, AbstractTensorShape}())
        finalizer(self, self->begin
            ccall((:TF_DeleteGraph, LIBTF), Void, (Ptr{Void},), self.ptr)
        end)
        self
    end
end

function add_to_collection(g::Graph, name, node)
    push!(g.collections[name], node)
end

"""
Returns a collection attached to the graph `g` named `name`
"""
function get_collection(g::Graph, name)
    g.collections[name]
end

"""
Returns a collection from the default graph
"""
get_collection(name) = get_collection(get_def_graph(), name)

const DEBUG_EXTEND_GRAPH = false

function extend_graph(graph::Graph, node_defs)
    n_nodes = length(node_defs)
    nodes = []
    for node_idx in 1:n_nodes
        proto = node_defs[node_idx]
        b = IOBuffer()
        write(b, proto)
        seekstart(b)
        node_def = tensorflow.NodeDef()
        readproto(b, node_def)
        if isnull(get_node_by_name(graph, node_def.name))
            push!(nodes, node_def)
        end
    end
    for (node_idx, node) in enumerate(nodes)
        if DEBUG_EXTEND_GRAPH
            info(node)
        end
        Operation(node)
    end
end

add_to_collection(name, node) = add_to_collection(get_def_graph(), name, node)

type SessionOptions
    ptr::Ptr{Void}

    function SessionOptions()
        ptr = ccall((:TF_NewSessionOptions, LIBTF), Ptr{Void}, ())
        return new(ptr)
    end
end

immutable TFException <: Exception
    status::Status
end


function check_status(status)
    if get_code(status) ≠ TF_OK
        throw(TFException(status))
    end
    nothing
end

"""
Returns the default computation graph, an object of type `Graph`.
"""
get_def_graph() = def_graph

function set_def_graph(g)
    global def_graph
    def_graph = g
end

function as_default(f, g::Graph)
    old_def = get_def_graph()
    set_def_graph(g)
    f()
    set_def_graph(old_def)
end

"""
A TensorFlow session.
"""
type Session
    ptr::Ptr{Void}
    graph::Graph

    function Session(graph, config=nothing)
        set_def_graph(graph)
        options = SessionOptions()
        if config !== nothing
            b = IOBuffer()
            writeproto(b, config)
            seekstart(b)
            proto = read(b)
            config_status = Status()
            ccall((:TF_SetConfig, LIBTF), Void, (Ptr{Void}, Ptr{Void}, Csize_t, Ptr{Void}), options.ptr, proto, sizeof(proto), config_status.ptr)
            check_status(config_status)
        end
        status = Status()
        ptr = ccall((:TF_NewSession, LIBTF), Ptr{Void}, (Ptr{Void}, Ptr{Void}, Ptr{Void}), graph.ptr, options.ptr, status.ptr)
        this = new(ptr, graph)
        check_status(status)
        finalizer(this, self->begin
            status = Status()
            ccall((:TF_DeleteSession, LIBTF), Void, (Ptr{Void}, Ptr{Void}), self.ptr, status.ptr)
        end)
        return this
    end

    function Session(;config=nothing, allow_growth=false)
        if config === nothing
            config = tensorflow.ConfigProto()
            gpu_config = tensorflow.GPUOptions()
            gpu_config.allow_growth = allow_growth
            config.gpu_options = gpu_config
        end
        Session(get_def_graph(), config)
    end
end


type Buffer
    ptr::Ptr{Void}

    function Buffer(s::Vector{UInt8})
        ptr = ccall((:TF_NewBufferFromString, LIBTF), Ptr{Void}, (Ptr{Void}, Csize_t), pointer(s), sizeof(s))
        return new(ptr)
    end

    function Buffer()
        self = new()
        self.ptr = ccall((:TF_NewBuffer, LIBTF), Ptr{Void}, ())
        finalizer(self, self->begin
            ccall((:TF_DeleteBuffer, LIBTF), Void, (Ptr{Void},), self.ptr)
        end)
        return self
    end
end

type BufferStruct
    data::Ptr{UInt8}
    len::Csize_t
    deallocator::Ptr{Void}
end

function getindex(b::Buffer)
    ccall((:TF_GetBuffer, LIBTF), BufferStruct, (Ptr{Void},), b.ptr)
end

function Base.convert(::Type{Array}, buf::Buffer)
    struct = buf[]
    array = unsafe_wrap(Array, struct.data, (struct.len,))
    copy(array)
end

function deallocator(data, len, arg)

end

const c_deallocator = Ref{Ptr}()

"""
Convert from row-major to column-major or vice-versa
"""
function convert_major_order(array)
    permutedims(array, length(size(array)):-1:1)
end


type RawTensor
    ptr::Ptr{Void}
    data::Array  # To avoid underlying data being GCed

    RawTensor() = new()

    function RawTensor(data::Array)
        dims = [size(data)...]
        dt = jl_to_df_type(eltype(data))
        data = convert_major_order(data)
        ptr = ccall((:TF_NewTensor, LIBTF), Ptr{Void}, (Cint, Ptr{Cint}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
            Int(dt),
            pointer(dims),
            length(dims),
            pointer(data),
            sizeof(data),
            c_deallocator[],
            C_NULL)
        return new(ptr, data)
    end

    function RawTensor(data::Number)
        dims = Cint[]
        dt = jl_to_df_type(eltype(data))
        data_boxed = [data]
        ptr = ccall((:TF_NewTensor, LIBTF), Ptr{Void}, (Cint, Ptr{Void}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
            Int(dt),
            pointer(dims),
            length(dims),
            pointer(data_boxed),
            sizeof(data_boxed),
            c_deallocator[],
            C_NULL)
        return new(ptr, data_boxed)
    end

    function RawTensor(ptr::Ptr)
        this = new(ptr)
        finalizer(this, this->begin
            ccall((:TF_DeleteTensor, LIBTF), Void, (Ptr{Void},), this.ptr)
        end)
        return this
    end
end

RawTensor(t::RawTensor) = t

function varint_encode(b::IO, n::Integer)
    while n ≥ 2^7
        write(b, UInt8(0b10000000 | (n & 0b1111111)))
        n >>= 7
    end
    write(b, UInt8(n))
end

function varint_decode(b::IO)
    n = 0
    idx = 0
    while true
        x = read(b, UInt8)
        if (x & 0b10000000) > 0
            x = x & 0b01111111
            n = n | (Int64(x) << 7idx)
        else
            n = n | (Int64(x) << 7idx)
            break
        end
        idx += 1
    end
    return n
end

function RawTensor(data::Array{String}, is_scalar=false)
    # TODO make work for multidimensional arrays
    # Currently only works for vectors and scalars
    t = RawTensor()
    t.data = data
    if is_scalar
        dims = Cint[]
    else
        dims = [size(data)...]
    end
    data = convert_major_order(data)
    b_data = IOBuffer()
    b = IOBuffer()
    for str in data
        write(b, UInt64(length(b_data.data)))
        varint_encode(b_data, sizeof(str))
        write(b_data, Vector{UInt8}(str))
    end
    seekstart(b_data)
    write(b, read(b_data))
    seekstart(b)
    data_encoded = read(b)
    dt = jl_to_df_type(String)
    ptr = ccall((:TF_NewTensor, LIBTF), Ptr{Void}, (Cint, Ptr{Int64}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
        Int(dt),
        dims,
        length(dims),
        data_encoded,
        length(data_encoded),
        c_deallocator[],
        C_NULL)
    if ptr == C_NULL
        error("Error creating tensor")
    end

    t.ptr = ptr
    return t
end

function RawTensor(data::String)
    RawTensor([data], true)
end


function Base.ndims(t::RawTensor)
    ccall((:TF_NumDims, LIBTF), Cint, (Ptr{Void},), t.ptr) |> Int
end

function Base.size(t::RawTensor, dim::Integer)
    n = ndims(t)
    dim -= 1
    @assert dim < n
    ccall((:TF_Dim, LIBTF), Clonglong, (Ptr{Void}, Cint), t.ptr, dim)
end

function Base.size(t::RawTensor)
    d = (size(t,_) for _ in 1:ndims(t))
    (d...)
end

function Base.sizeof(t::RawTensor)
    ccall((:TF_TensorByteSize, LIBTF), Csize_t, (Ptr{Void},), t.ptr) |> Int
end


type NodeDescription
    ptr::Ptr{Void}
    graph::Graph

    function NodeDescription(graph, op_type, full_name)
        desc = ccall((:TF_NewOperation, LIBTF), Ptr{Void}, (Ptr{Void}, Cstring, Cstring), graph.ptr, op_type, full_name)
        self = new(desc, graph)
        for control_op in vcat(op_context.control_ops)
            add_control_input(self, control_op)
        end
        self
    end

end

NodeDescription(op_type, node_name) = NodeDescription(get_def_graph(), op_type, node_name)

function get_cur_node_name()
    join(op_context.names, "/")
end

function NodeDescription(op_type)
    name = get_cur_node_name()
    NodeDescription(op_type, name)
end

get_graph(desc::NodeDescription) = Nullable(desc.graph)

abstract AbstractOperation

"""
An operation in the computation graph.
"""
type Operation <: AbstractOperation
    ptr::Ptr{Void}
    graph::Nullable{Graph}
    op_name::String
    name::String
    Operation() = new()
end

immutable Port
    node_ptr::Ptr{Void}
    index::Int
end

function get_input(op::Operation, idx)
    port = Port(op.ptr, idx-1)
    in_port = ccall((:TF_OperationInput, LIBTF), Port, (Port,), port)
    Tensor(in_port)
end

function get_input_list_length(op::Operation, arg_name)
    status = Status()
    out = ccall((:TF_OperationInputListLength, LIBTF), Cint, (Ptr{Void}, Cstring, Ptr{Void}), op.ptr, arg_name, status.ptr)
    check_status(status)
    Int(out)
end

immutable AttrMetadata
    is_list::Bool
    list_size::Int64
    _type::Int32
    total_size::Int64
end

function get_attr_metadata(op::Operation, attr)
    status = Status()
    out = ccall((:TF_OperationGetAttrMetadata, LIBTF), AttrMetadata, (Ptr{Void}, Cstring, Ptr{Void}), op.ptr, attr, status.ptr)
    check_status(status)
    out
end

function get_attr(op::Operation, attr, ::Type{Int})
    out = Ref{Int}()
    status = Status()
    ccall((:TF_OperationGetAttrInt, LIBTF), Void, (Ptr{Void}, Cstring, Ref{Int}, Ptr{Void}), op.ptr, attr, out, status.ptr)
    check_status(status)
    out[]
end

function get_attr(op::Operation, attr, ::Type{Array})
    out = Ref{Ptr{Void}}()
    status = Status()
    ccall((:TF_OperationGetAttrTensor, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Ptr{Void}}, Ptr{Void}), op.ptr, attr, out, status.ptr)
    check_status(status)
    Array(RawTensor(out[]))
end

function get_attr(op::Operation, attr, ::Type{Bool})
    out = Ref{Bool}()
    status = Status()
    ccall((:TF_OperationGetAttrBool, LIBTF), Void, (Ptr{Void}, Cstring, Ref{Bool}, Ptr{Void}), op.ptr, attr, out, status.ptr)
    check_status(status)
    out[]
end

function get_attr(op::Operation, attr, ::Type{Vector{Int}})
    meta = get_attr_metadata(op, attr)
    out = Vector{Int}(meta.list_size)
    status = Status()
    ccall((:TF_OperationGetAttrIntList, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Int}, Cint, Ptr{Void}), op.ptr, attr, out, length(out), status.ptr)
    check_status(status)
    out
end

function get_attr(op::Operation, attr, ::Type{String})
    meta = get_attr_metadata(op, attr)
    out = Vector{UInt8}(meta.total_size)
    status = Status()
    ccall((:TF_OperationGetAttrString, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{UInt8}, Cint, Ptr{Void}), op.ptr, attr, out, length(out), status.ptr)
    check_status(status)
    String(out)
end

function fillin(op::Operation)
    op.name = ccall((:TF_OperationName, LIBTF), Cstring, (Ptr{Void},), op.ptr) |> unsafe_string
    op.op_name = ccall((:TF_OperationOpType, LIBTF), Cstring, (Ptr{Void},), op.ptr) |> unsafe_string
end

type OperationContext
    control_ops::Vector{Vector{Operation}}
    names::Vector{String}
end

const op_context = OperationContext(Vector{Operation}[], String[])

function with_op_name(f, name)
    push!(op_context.names, get_name(name))
    f()
    pop!(op_context.names)
end

function with_op_control(f, control_ops)
    push!(op_context.control_ops, control_ops)
    f()
    pop!(op_context.control_ops)
end

function Operation(desc::NodeDescription)
    self = Operation()
    status = Status()
    ptr = ccall((:TF_FinishOperation, LIBTF), Ptr{Void}, (Ptr{Void}, Ptr{Void}), desc.ptr, status.ptr)
    check_status(status)
    self.ptr = ptr
    self.graph = Nullable(desc.graph)
    fillin(self)
    return self
end

function Operation(ptr::Ptr)
    self = Operation()
    self.ptr = ptr
    self.graph = Nullable{Graph}()
    fillin(self)
    return self
end

type NodeNameNotFound <: Exception
    name::String
end

get_graph(n::AbstractOperation) = Operation(n).graph

function load_proto(tensor::tensorflow.TensorProto)
    dtype = tensor.dtype
    dim = (Int[_.size for _ in tensor.tensor_shape.dim]...)
    if dtype == tensorflow._DataType.DT_FLOAT
        val = tensor.float_val
    elseif dtype == tensorflow._DataType.DT_INT32
        val = tensor.int_val
    elseif dtype == tensorflow._DataType.DT_INT64
        val = tensor.int64_val
    elseif dtype == tensorflow._DataType.DT_DOUBLE
        val = tensor.double_val
    elseif dtype == tensorflow._DataType.DT_STRING
        val = tensor.string_val
    else
        warn("Unrecognized datatype $dtype")
    end
    # Sometimes Tensorflow store the tensor content in the 'tensor_content' byte array,
    # and sometimes in a typed field. Haven't figured out the rational yet.
    if length(tensor.tensor_content) > 0
        # Vector-valued string tensors are stored as eg
        # ""\x02\x03hibye" for ["hi", "bye"]
        if dtype == tensorflow._DataType.DT_STRING
            bytes = tensor.tensor_content
            sizes = Int[]
            pos = 1
            for i in 1:prod(dim)
                push!(sizes, bytes[pos])
                pos += 1
            end
            val = Vector{UInt8}[]
            for i in 1:length(sizes)
                val_i = UInt8[]
                for j in 1:sizes[i]
                    push!(val_i, bytes[pos])
                    pos += 1
                end
                push!(val, val_i)
            end
        else
            val = reinterpret(eltype(val), tensor.tensor_content)
        end
    end
    if length(val) == 0 && length(dim) == 0
        zeros(eltype(val),0)
    elseif length(dim) == 0
        val[1]
    else
        # https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#constant
        if length(val) < prod(dim)
            last_val = val[end]
            original_length = length(val)
            resize!(val, prod(dim))
            for i in (original_length+1):length(val)
                val[i] = last_val
            end
        end
        reshape(val, dim) |> convert_major_order
    end
end

function load_proto(shape::tensorflow.TensorShapeProto)
    dims = Nullable{Int}[]
    if shape.unknown_rank
        ShapeInference.TensorShape(nothing)
    else
        for dim in shape.dim
            if dim == -1
                push!(dims, Nullable{Int}())
            else
                push!(dims, Nullable(dim))
            end
        end
        ShapeInference.TensorShape(dims)
    end
end

function load_proto(list::tensorflow.AttrValue_ListValue)
    if has_field(list, :i)
        list.i
    elseif has_field(list, :b)
        list.b
    elseif has_field(list, :f)
        list.f
    elseif has_field(list, :s)
        list.s
    end
end

function load_proto(value::tensorflow.AttrValue)
    if has_field(value, :tensor)
        load_proto(value.tensor)
    elseif has_field(value, :s)
        String(value.s)
    elseif has_field(value, :i)
        value.i
    elseif has_field(value, :b)
        value.b
    elseif has_field(value, :shape)
        load_proto(value.shape)
    elseif has_field(value, :list)
        load_proto(value.list)
    end
end

# Replace this entire function once we can import protobufs into a graph
function Operation(node_def::tensorflow.NodeDef)
    graph = get_def_graph()
    desc = NodeDescription(graph, node_def.op, node_def.name)

    if node_def.op == "DynamicStitch"
        inputs = []
        ports = []
        for input in node_def.input
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)|>get
            push!(inputs, input_node)
            push!(ports, port)
        end
        add_input(desc, [Tensor(inputs[1], ports[1]), Tensor(inputs[2], ports[2])])
        add_input(desc, [Tensor(inputs[3], ports[3]), Tensor(inputs[4], ports[4])])
        return Operation(desc)
    end
    if node_def.op ∈ ("ConcatOffset", "Concat")
        input, port = parse_port_name(node_def.input[1])
        input_node = get_node_by_name(input) |> get
        add_input(desc, Tensor(input_node, port))
        inputs = Tensor[]
        for idx in 2:length(node_def.input)
            input, port = parse_port_name(node_def.input[idx])
            input_node = get_node_by_name(input) |> get
            push!(inputs, Tensor(input_node, port))
        end
        add_input(desc, inputs)
    elseif node_def.op ∈ ("AddN", "ShapeN")
        inputs = Tensor[]
        for input in node_def.input
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)|>get
            push!(inputs, Tensor(input_node, port))
        end
        add_input(desc, inputs)
    elseif node_def.op == "Pack"
        inputs = Tensor[]
        for input in node_def.input
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input) |> get
            push!(inputs, Tensor(input_node, port))
        end
        add_input(desc, inputs)
    else
        for (input_idx, input) in enumerate(node_def.input)
            input_kind = :normal
            if input[1] == '^'
                input_kind = :control
                input = input[2:end]
            end
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)
            if isnull(input_node)
                warn("Could not find name $input")
            end
            if input_kind == :normal
                add_input(desc, Tensor(input_node |> get, port))
            elseif input_kind == :control
                add_control_input(desc, input_node |> get)
            end
        end
    end
    if isdefined(node_def, :attr)  # TODO: complete this
        for (attr_name, attr) in node_def.attr
            if attr_name ∈ ("dtype", "T", "DstT", "SrcT", "Index")
                ccall((:TF_SetAttrType, LIBTF), Void, (Ptr{Void}, Cstring, Cint), desc.ptr, attr_name, attr._type)
            elseif attr_name == "value"
                desc["value"] = RawTensor(load_proto(attr.tensor))
            elseif attr_name == "keep_dims"
                desc["keep_dims"] = attr.b
            elseif attr_name == "N"
                desc["N"] = attr.i
            elseif attr_name == "transpose_a"
                desc["transpose_a"] = attr.b
            elseif attr_name == "transpose_b"
                desc["transpose_b"] = attr.b
            elseif attr_name == "strides"
                set_attr_list(desc, "strides", attr.list.i)
            elseif attr_name == "padding"
                desc["padding"] = String(attr.s)
            elseif attr_name == "ksize"
                set_attr_list(desc, "ksize", attr.list.i)
            elseif attr_name == "data_format"
                desc["data_format"] = String(attr.s)
            elseif attr_name == "use_cudnn_on_gpu"
                desc["use_cudnn_on_gpu"] = attr.b
            elseif attr_name == "axis"
                desc["axis"] = attr.i
            elseif attr_name ∈ ("begin_mask", "ellipsis_mask", "shrink_axis_mask", "new_axis_mask", "end_mask")
                desc[attr_name] = attr.i
            else
                #warn("Unrecognized attribute $attr_name")
            end
        end
    end
    Operation(desc)
end

"""
`node_name(node::AbstractOperation)`

Returns the name of a node in the computation graph.
"""
node_name(node::AbstractOperation) = ccall((:TF_OperationName, LIBTF), Cstring, (Ptr{Void},), Operation(node).ptr) |> unsafe_string


function get_attr_value_proto(node::Operation, attr_name)
    buf = Buffer()
    status = Status()
    ccall((:TF_OperationGetAttrValueProto, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Void}, Ptr{Void}), node.ptr, attr_name, buf.ptr, status.ptr)
    check_status(status)
    proto = Array(buf)
    b = IOBuffer()
    write(b, proto)
    seekstart(b)
    val = tensorflow.AttrValue()
    readproto(b, val)
    return val
end

Base.getindex(node::Operation, attr_name) = get_attr_value_proto(node, attr_name)

const dt = tensorflow._DataType
const proto_type_map = Dict(dt.DT_FLOAT=>Float32, dt.DT_INT32=>Int32, dt.DT_DOUBLE=>Float64, dt.DT_INT64=>Int64, dt.DT_STRING=>String, dt.DT_BOOL=>Bool)

abstract AbstractTensor

"""
Represents the output of an operation in the computation graph
"""
type Tensor <: AbstractTensor
    op::Operation
    value_index::Int
end

function Base.isequal(t1::Tensor, t2::Tensor)
    t1.op.ptr == t2.op.ptr && t1.value_index==t2.value_index
end

function Base.hash(t::Tensor, h::UInt64)
    hash(t.op.ptr, hash(t.value_index, h))
end

node_name(t::AbstractTensor) = (node_name(Tensor(t).op), Tensor(t).value_index)

Tensor(op::Operation) = Tensor(op, 1)

function Base.eltype(t::AbstractTensor)
    tf_type = ccall((:TF_OperationOutputType, LIBTF), TF_DataType, (Port,), Port(Tensor(t)))
    if !haskey(type_map, tf_type)
        local dtype
        try
            dtype = get_op(t)["T"]._type
        catch
            dtype = get_op(t)["dtype"]._type
        end
        return proto_type_map[dtype]
    else
        return tf_to_jl_type(tf_type)
    end
end

Port(t::Tensor) = Port(t.op.ptr, t.value_index-1)
Port(op::Operation) = Port(Tensor(op))
Port(port::Port) = port

Tensor(p::Port) = Tensor(Operation(p.node_ptr), p.index+1)

function add_input(desc::NodeDescription, input::Union{Tensor, Operation})
    ccall((:TF_AddInput, LIBTF), Void, (Ptr{Void}, Port), desc.ptr, Port(input))
end

function add_input(desc::NodeDescription, inputs::Vector)
    inputs = map(Port, inputs)
    ccall((:TF_AddInputList, LIBTF), Void, (Ptr{Void}, Ptr{Void}, Cint), desc.ptr, inputs, length(inputs))
end

function add_control_input(desc::NodeDescription, op::Operation)
    ccall((:TF_AddControlInput, LIBTF), Void, (Ptr{Void}, Ptr{Void}), desc.ptr, op.ptr)
end

add_control_input(desc::NodeDescription, t::Tensor) = add_control_input(desc, t.op)

function setindex!(desc::NodeDescription, tensor::RawTensor, attr_name)
    status = Status()
    ccall((:TF_SetAttrTensor, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Void}, Ptr{Void}), desc.ptr, attr_name, tensor.ptr, status.ptr)
    check_status(status)
end

function setindex!(desc::NodeDescription, tensors::Vector{RawTensor}, attr_name)
    status = Status()
    ccall((:TF_SetAttrTensorList, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Ptr{Void}}, Cint, Ptr{Void}), desc.ptr, attr_name, [_.ptr for _ in tensors], length(tensors), status.ptr)
    check_status(status)
end

function setindex!(desc::NodeDescription, dtype::DataType, attr_name)
    ccall((:TF_SetAttrType, LIBTF), Void, (Ptr{Void}, Cstring, TF_DataType), desc.ptr, attr_name, dtype|>jl_to_df_type)
end

function setindex!(desc::NodeDescription, value::Int, attr_name)
    ccall((:TF_SetAttrInt, LIBTF), Void, (Ptr{Void}, Cstring, Int64), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::Tuple, attr_name)
    dims = Int[value...]
    ccall((:TF_SetAttrShape, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Int64}, Cint), desc.ptr, attr_name, dims, length(dims))
end

function setindex!(desc::NodeDescription, value::Bool, attr_name)
    ccall((:TF_SetAttrBool, LIBTF), Void, (Ptr{Void}, Cstring, Cuchar), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::Float32, attr_name)
    ccall((:TF_SetAttrFloat, LIBTF), Void, (Ptr{Void}, Cstring, Cfloat), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::AbstractString, attr_name)
    value = String(value)
    ccall((:TF_SetAttrString, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Void}, Cint), desc.ptr, attr_name, Vector{UInt8}(value), sizeof(value))
end

function set_attr_list(desc::NodeDescription, attr_name, list::Vector{Int})
    ccall((:TF_SetAttrIntList, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Int64}, Cint), desc.ptr, attr_name, list, length(list))
end

function set_attr_list{T<:DataType}(desc::NodeDescription, attr_name, list::Vector{T})
    list = map(jl_to_df_type, list)
    ccall((:TF_SetAttrTypeList, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Void}, Cint), desc.ptr, attr_name, list, length(list))
end

function set_attr_shape_list(desc::NodeDescription, attr_name, list::Vector)
    dims = Vector{Int}[]
    for shape in list
        push!(dims, [shape...])
    end
    ccall((:TF_SetAttrShapeList, LIBTF), Void, (Ptr{Void}, Cstring, Ptr{Ptr{Int64}}, Ptr{Cint}, Cint),
        desc.ptr,
        attr_name,
        dims,
        [length(_) for _ in dims],
        length(dims))
end

function Base.eltype(t::RawTensor)
    tf_type = ccall((:TF_TensorType, LIBTF), TF_DataType, (Ptr{Void},), t.ptr)
    tf_to_jl_type(tf_type)
end

const type_map = Dict(TF_UINT8=>UInt8, TF_FLOAT=>Float32, TF_INT32=>Int32,
                      TF_INT64=>Int64, TF_DOUBLE=>Float64, TF_STRING=>String,
                      TF_BOOL=>Bool, TF_COMPLEX64=>Complex64,
                      TF_COMPLEX128=>Complex128)
const inv_type_map = Dict(v=>k for (k, v) in type_map)

function tf_to_jl_type(dt::TF_DataType)
    return type_map[dt]
end

function jl_to_df_type(dt)
    return inv_type_map[dt]
end

function Base.convert(::Type{Array}, t::RawTensor)
    dims = ndims(t)
    data = ccall((:TF_TensorData, LIBTF), Ptr{Void}, (Ptr{Void},), t.ptr)
    data = convert(Ptr{eltype(t)}, data)
    if eltype(t) == String
        d = size(t)
        out = String[]
        array = unsafe_wrap(Array, convert(Ptr{UInt8}, data), sizeof(t))
        b = IOBuffer(array)
        seekstart(b)
        read(b, UInt64, prod(d))  # The offsets
        for i in 1:prod(d)
            len = varint_decode(b)
            raw_data = read(b, UInt8, len)
            push!(out, String(raw_data))
        end
        out
    else
        if dims > 0
            convert_major_order(unsafe_wrap(Array, data, size(t)|>reverse))
        else
            copy(unsafe_wrap(Array, data, size(t)))
        end
    end
end

function Base.convert{T<:Union{Number, String}}(::Type{T}, t::RawTensor)
    @assert ndims(t)==0
    return convert(Array, t)[]
end

function get_proto(graph::Graph)
    output = Buffer()
    status = Status()
    ccall((:TF_GraphToGraphDef, LIBTF), Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}), graph.ptr, output.ptr, status.ptr)
    check_status(status)
    convert(Array, output)
end

function get_proto(node::Operation)
    output = Buffer()
    status = Status()
    ccall((:TF_OperationToNodeDef, LIBTF), Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}), node.ptr, output.ptr, status.ptr)
    check_status(status)
    convert(Array, output)
end

get_def_type(::Type{Operation}) = tensorflow.NodeDef
get_def_type(::Type{Graph}) = tensorflow.GraphDef

"""
Returns the definition of the given operation or graph
"""
function get_def(n::Union{Operation, Graph})
    p = get_proto(n)
    b = IOBuffer()
    write(b, p)
    seekstart(b)
    desc = get_def_type(typeof(n))()
    readproto(b, desc)
    return desc
end

get_def(t::Tensor) = get_def(t.op)

function parse_port_name(name)
    m = match(r"(.*):(.*)", name)
    if m==nothing
        return (name, 1)
    else
        port = parse(Int, m[2]) + 1
        return (m[1], port)
    end

end

"""
Returns an operation by searching for its name in the given graph.
"""
function get_node_by_name(graph::Graph, name::AbstractString)
    name, port = parse_port_name(name)
    node_ptr = ccall((:TF_GraphOperationByName, LIBTF), Ptr{Void}, (Ptr{Void}, Cstring), graph.ptr, name)
    if node_ptr == C_NULL
        return Nullable{Operation}()
    else
        return Nullable(Operation(node_ptr))
    end
end

get_node_by_name(name) = get_node_by_name(get_def_graph(), name)

function gradients(y, x::AbstractArray)
    x_names = [node_name(_) for _ in x]
    y_name = node_name(y)
    graph_proto = get_def_graph() |> get_proto
    eval(Main, quote
        node_protos, grad_names = remotecall_fetch(($pyproc[])) do
            py_gradients($graph_proto, $x_names, $y_name)
        end
    end)
    node_protos = Main.node_protos
    grad_names = Main.grad_names
    extend_graph(get_def_graph(), node_protos)
    out = []
    for name in grad_names
        if isa(name, String)
            push!(out, Tensor(get_node_by_name(name)|>get, 1))
        else
            push!(out, IndexedSlices(Tensor(get_node_by_name(name[1])|>get,1), Tensor(get_node_by_name(name[2])|>get,1)+1))
        end
    end
    return out
end

gradients(y, x) = gradients(y, [x])[1]

function get_num_outputs(op::Operation)
    ccall((:TF_OperationNumOutputs, LIBTF), Cint, (Ptr{Void},), op.ptr) |> Int
end

function get_device(op::Operation)
    ccall((:TF_OperationDevice, LIBTF), Cstring, (Ptr{Void},), op.ptr) |> String
end

get_device(t::Tensor) = get_device(t.op)

function num_outputs(op::Operation)
    ccall((:TF_OperationNumOutputs, LIBTF), Cint, (Ptr{Void},), op.ptr) |> Int
end

get_op(op::Operation) = op
get_op(t::AbstractTensor) = Tensor(t).op

type IndexedSlices
    values::Tensor
    indices::Tensor
end

Base.eltype(i::IndexedSlices) = eltype(i.values)

type IndexedSlicesValue
    values
    indices
end
