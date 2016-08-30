using ProtoBuf
using PyCall

import Base: setindex!, getindex, run

const LIB_BASE = joinpath(dirname(@__FILE__), "..", "deps")

if myid() == 1
    if is_apple()
        Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local-fastbuild", "bin", "tensorflow", "libtensorflow"), Libdl.RTLD_GLOBAL)
        Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local-fastbuild", "bin", "tensorflow", "c", "libc_api"), Libdl.RTLD_GLOBAL)
    end
    if is_linux()
        if "TF_USE_CPU" ∈ keys(ENV)
            Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local_linux-fastbuild", "bin", "tensorflow", "libtensorflow"), Libdl.RTLD_GLOBAL)
            Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local_linux-fastbuild", "bin", "tensorflow", "c", "libc_api"), Libdl.RTLD_GLOBAL)
        else
            Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local_linux-opt", "bin", "tensorflow", "libtensorflow"), Libdl.RTLD_GLOBAL)
            Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local_linux-opt", "bin", "tensorflow", "c", "libc_api"), Libdl.RTLD_GLOBAL)
        end
    end
end

type Status
    ptr::Ptr{Void}
    function Status()
        ptr = ccall((:TF_NewStatus), Ptr{Void}, ())
        this = new(ptr)
        this
    end
end

function Base.show(io::IO, s::Status)
    msg = ccall((:TF_Message), Cstring, (Ptr{Void},), s.ptr) |> unsafe_string
    print(io, @sprintf("Status: %s", msg))
end

function get_code(s::Status)
    code = ccall((:TF_GetCode), Cint, (Ptr{Void},), s.ptr)
    return TF_Code(code)
end

"""
A TensorFlow computation graph
"""
type Graph
    ptr::Ptr{Void}
    collections::Dict{Symbol, Any}

    function Graph()
        ptr = ccall((:TF_NewGraph), Ptr{Void}, ())
        collections = Dict{Symbol, Any}()
        collections[:Variables] = []
        self = new(ptr, collections)
        finalizer(self, self->begin
            ccall((:TF_DeleteGraph), Void, (Ptr{Void},), self.ptr)
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
        Operation(node)
    end
end

add_to_collection(name, node) = add_to_collection(get_def_graph(), name, node)

type SessionOptions
    ptr::Ptr{Void}

    function SessionOptions()
        ptr = ccall((:TF_NewSessionOptions), Ptr{Void}, ())
        return new(ptr)
    end
end

immutable TFException <: Exception
    status::Status
end

function Base.show(io::IO, err::TFException)
    println(io, @sprintf("Tensorflow error: %s", string(err.status)))
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

    function Session(graph)
        set_def_graph(graph)
        options = SessionOptions()
        status = Status()
        ptr = ccall((:TF_NewSessionWithGraph), Ptr{Void}, (Ptr{Void}, Ptr{Void}, Ptr{Void}), graph.ptr, options.ptr, status.ptr)
        this = new(ptr, graph)
        check_status(status)
        finalizer(this, self->begin
            status = Status()
            ccall((:TF_DeleteSessionWithGraph), Void, (Ptr{Void}, Ptr{Void}), self.ptr, status.ptr)
        end)
        return this
    end

    Session() = Session(get_def_graph())
end

function Base.show(io::IO, s::Session)
    print(io, "Session($(pointer_from_objref(s)))")
end

type Buffer
    ptr::Ptr{Void}

    function Buffer(s::Vector{UInt8})
        ptr = ccall((:TF_NewBufferFromString), Ptr{Void}, (Ptr{Void}, Csize_t), pointer(s), sizeof(s))
        return new(ptr)
    end

    function Buffer()
        self = new()
        self.ptr = ccall(:TF_NewBuffer, Ptr{Void}, ())
        finalizer(self, self->begin
            ccall(:TF_DeleteBuffer, Void, (Ptr{Void},), self.ptr)
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
    ccall(:TF_GetBuffer, BufferStruct, (Ptr{Void},), b.ptr)
end

function Base.convert(::Type{Array}, buf::Buffer)
    struct = buf[]
    array = unsafe_wrap(Array, struct.data, (struct.len,))
    copy(array)
end

function deallocator(data, len, arg)

end

const c_deallocator = cfunction(deallocator, Void, (Ptr{Void}, Csize_t, Ptr{Void}))

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
        ptr = ccall((:TF_NewTensor), Ptr{Void}, (Cint, Ptr{Cint}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
            Int(dt),
            pointer(dims),
            length(dims),
            pointer(data),
            sizeof(data),
            c_deallocator,
            C_NULL)
        return new(ptr, data)
    end

    function RawTensor(data::Number)
        dims = Cint[]
        dt = jl_to_df_type(eltype(data))
        data_boxed = [data]
        ptr = ccall((:TF_NewTensor), Ptr{Void}, (Cint, Ptr{Void}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
            Int(dt),
            pointer(dims),
            length(dims),
            pointer(data_boxed),
            sizeof(data_boxed),
            c_deallocator,
            C_NULL)
        return new(ptr, data_boxed)
    end

    function RawTensor(ptr::Ptr)
        this = new(ptr)
        finalizer(this, this->begin
            ccall((:TF_DeleteTensor), Void, (Ptr{Void},), this.ptr)
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

function RawTensor(data::String)
    # TODO: Support arrays of strings
    b = IOBuffer()
    write(b, UInt64(0))
    varint_encode(b, sizeof(data))
    write(b, data.data)
    seekstart(b)
    data_encoded = read(b)
    dims = Cint[]
    dt = jl_to_df_type(String)
    ptr = ccall(:TF_NewTensor, Ptr{Void}, (Cint, Ptr{Int64}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
        Int(dt),
        dims,
        0,
        data_encoded,
        length(data_encoded),
        c_deallocator,
        C_NULL)
    if ptr == C_NULL
        error("Error creating tensor")
    end
    t = RawTensor()
    t.data = [data]
    t.ptr = ptr
    return t
end


function Base.show(io::IO, t::RawTensor)
    print(io, "RawTensor: ")
    if ndims(t) == 0
        if eltype(t) == String
            show(io, String(t))
        else
            show(io, Number(t))
        end
    else
        show(io, Array(t))
    end
end

function Base.ndims(t::RawTensor)
    ccall((:TF_NumDims), Cint, (Ptr{Void},), t.ptr) |> Int
end

function Base.size(t::RawTensor, dim::Integer)
    n = ndims(t)
    dim -= 1
    @assert dim < n
    ccall((:TF_Dim), Clonglong, (Ptr{Void}, Cint), t.ptr, dim)
end

function Base.size(t::RawTensor)
    d = (size(t,_) for _ in 1:ndims(t))
    (d...)
end

function Base.sizeof(t::RawTensor)
    ccall((:TF_TensorByteSize), Csize_t, (Ptr{Void},), t.ptr) |> Int
end


type NodeDescription
    ptr::Ptr{Void}
    graph::Graph

    function NodeDescription(graph, op_type, node_name)
        desc = ccall((:TF_NewNode), Ptr{Void}, (Ptr{Void}, Cstring, Cstring), graph.ptr, op_type, node_name)
        new(desc, graph)
    end

end

NodeDescription(op_type, node_name) = NodeDescription(get_def_graph(), op_type, node_name)

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
    inputs::Vector  # Vector{Tensor}
    attrs::Dict{String, Any}
    filled_in::Bool

    Operation() = new()


end

function Operation(desc::NodeDescription)
    self = Operation()
    self.filled_in = false
    status = Status()
    ptr = ccall((:TF_FinishNode), Ptr{Void}, (Ptr{Void}, Ptr{Void}), desc.ptr, status.ptr)
    check_status(status)
    self.ptr = ptr
    self.graph = Nullable(desc.graph)
    #fillin_operation(self)
    return self
end

function Operation(ptr::Ptr)
    self = Operation()
    self.filled_in = false
    self.ptr = ptr
    self.graph = Nullable{Graph}()
    # fillin_operation(self)
    return self
end

function fillin_operation(op::Operation)
    op.filled_in && return
    my_desc = get_def(op)
    if has_field(my_desc, :op)
        op.op_name = my_desc.op
    end
    if has_field(my_desc, :name)
        op.name = my_desc.name
    end
    if has_field(my_desc, :attr)
        op.attrs = my_desc.attr
    end
    if isnull(op.graph)
        graph = get_def_graph()
    else
        graph = get(op.graph)
    end
    if has_field(my_desc, :input)
        op.inputs = [Tensor((get_node_by_name(graph, name) |> get)) for name in my_desc.input]
        for input in op.inputs
            fillin_operation(input.op)
        end
    end
    op.filled_in = true
    return op
end

get_graph(n::AbstractOperation) = Operation(n).graph

function Base.show(io::IO, n::Operation)
    print(io, "<Operation '$(node_name(n))' dtype=$(eltype(n))>")
end

# Replace this entire function once we can import protobufs into a graph
function Operation(node_def::tensorflow.NodeDef)
    graph = get_def_graph()
    desc = NodeDescription(graph, node_def.op, node_def.name)
    if node_def.op == "DynamicStitch"
        inputs = []
        for input in node_def.input
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)|>get
            push!(inputs, input_node)
        end
        add_input(desc, [Tensor(inputs[1], 1), Tensor(inputs[2], 1)])
        add_input(desc, [Tensor(inputs[3], 1), Tensor(inputs[4], 1)])
        return Operation(desc)
    end
    if node_def.op ∈ ("ConcatOffset", "Concat")
        input, port = parse_port_name(node_def.input[1])
        input_node = get_node_by_name(input) |> get
        add_input(desc, Tensor(input_node, port))
        inputs = Tensor[]
        for idx in 2:length(node_def.input)
            input, port = parse_port_name(node_def.input[2])
            input_node = get_node_by_name(input) |> get
            push!(inputs, Tensor(input_node, port))
        end
        add_input(desc, inputs)
        return Operation(desc)
    end
    if node_def.op ∈ ("AddN", "ShapeN")
        inputs = Tensor[]
        for input in node_def.input
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)|>get
            push!(inputs, Tensor(input_node, port))
        end
        add_input(desc, inputs)
    else
        for (input_idx, input) in enumerate(node_def.input)
            if input[1] == '^'
                continue
            end
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)
            if isnull(input_node)
                warn("Could not find name $input")
            end
            add_input(desc, Tensor(input_node |> get, port))
        end
    end
    if isdefined(node_def, :attr)  # TODO: complete this
        for (attr_name, attr) in node_def.attr
            if attr_name ∈ ("dtype", "T", "DstT", "SrcT")
                ccall(:TF_SetAttrType, Void, (Ptr{Void}, Cstring, Cint), desc.ptr, attr_name, attr._type)
            elseif attr_name == "value"
                dtype = attr.tensor.dtype
                dim = (Int[_.size for _ in attr.tensor.tensor_shape.dim]...)
                if dtype == tensorflow._DataType.DT_FLOAT
                    val = attr.tensor.float_val
                elseif dtype == tensorflow._DataType.DT_INT32
                    val = attr.tensor.int_val
                elseif dtype == tensorflow._DataType.DT_DOUBLE
                    val = attr.tensor.double_val
                else
                    warn("Unrecognized datatype $dtype")
                end
                # Sometimes Tensorflow stores the tensor content in the 'tensor_content' byte array,
                # and sometimes in a typed field. Haven't figured out the rational yet.
                if length(attr.tensor.tensor_content) > 0
                    val = reinterpret(eltype(val), attr.tensor.tensor_content)
                end
                if length(val) == 0
                    desc["value"] = RawTensor(zeros(eltype(val),0))
                elseif length(dim) == 0
                    desc["value"] = RawTensor(val[1])
                else
                    desc["value"] = RawTensor(reshape(val, dim))
                end
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
            else
                warn("Unrecognized attribute $attr_name")
            end
        end
    end
    Operation(desc)
end

"""
`node_name(node::AbstractOperation)`

Returns the name of a node in the computation graph.
"""
node_name(node::AbstractOperation) = ccall((:TF_NodeName), Cstring, (Ptr{Void},), Operation(node).ptr) |> unsafe_string


function get_attr_value_proto(node::Operation, attr_name)
    buf = Buffer()
    status = Status()
    ccall(:TF_NodeGetAttrValueProto, Void, (Ptr{Void}, Cstring, Ptr{Void}, Ptr{Void}), node.ptr, attr_name, buf.ptr, status.ptr)
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

"""
`eltype(node::AbstractOperation)`

Returns the type of the tensor the given operation will return when executed.
"""
function Base.eltype(node::AbstractOperation)
    node = Operation(node)
    dtype = nothing
    try
        dtype = node["dtype"]._type
    catch
        try
            dtype = node["T"]._type
        catch
            error("eltype called on node with no type information")
        end
    end
    return proto_type_map[dtype]
end

abstract AbstractTensor

"""
Represents the output of an operation in the computation graph
"""
type Tensor <: AbstractTensor
    op::Operation
    value_index::Int
end

function Base.show(io::IO, t::Tensor)
    local dtype
    try
        dtype = eltype(t)
    catch
        dtype = "?"
    end
    print(io, "<Tensor $(node_name(t.op)):$(t.value_index) dtype=$(dtype)>")
end

node_name(t::AbstractTensor) = node_name(Tensor(t).op)

Tensor(op::Operation) = Tensor(op, 1)

Base.eltype(t::AbstractTensor) = eltype(Tensor(t).op)

immutable Port
    node_ptr::Ptr{Void}
    index::Int
end

Port(t::Tensor) = Port(t.op.ptr, t.value_index-1)
Port(op::Operation) = Port(Tensor(op))
Port(port::Port) = port

function add_input(desc::NodeDescription, input::Union{Tensor, Operation})
    ccall((:TF_AddInput), Void, (Ptr{Void}, Port), desc.ptr, Port(input))
end

function add_input(desc::NodeDescription, inputs::Vector{Tensor})
    inputs = map(Port, inputs)
    ccall((:TF_AddInputList), Void, (Ptr{Void}, Ptr{Void}, Cint), desc.ptr, inputs, length(inputs))
end

function setindex!(desc::NodeDescription, tensor::RawTensor, attr_name)
    status = Status()
    ccall((:TF_SetAttrTensor), Void, (Ptr{Void}, Cstring, Ptr{Void}, Ptr{Void}), desc.ptr, attr_name, tensor.ptr, status.ptr)
    check_status(status)
end

function setindex!(desc::NodeDescription, tensors::Vector{RawTensor}, attr_name)
    status = Status()
    ccall(:TF_SetAttrTensorList, Void, (Ptr{Void}, Cstring, Ptr{Ptr{Void}}, Cint, Ptr{Void}), desc.ptr, attr_name, [_.ptr for _ in tensors], length(tensors), status.ptr)
    check_status(status)
end

function setindex!(desc::NodeDescription, dtype::DataType, attr_name)
    ccall((:TF_SetAttrType), Void, (Ptr{Void}, Cstring, TF_DataType), desc.ptr, attr_name, dtype|>jl_to_df_type)
end

function setindex!(desc::NodeDescription, value::Int, attr_name)
    ccall((:TF_SetAttrInt), Void, (Ptr{Void}, Cstring, Int64), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::Tuple, attr_name)
    dims = Int[value...]
    ccall(:TF_SetAttrShape, Void, (Ptr{Void}, Cstring, Ptr{Int64}, Cint), desc.ptr, attr_name, dims, length(dims))
end

function setindex!(desc::NodeDescription, value::Bool, attr_name)
    ccall(:TF_SetAttrBool, Void, (Ptr{Void}, Cstring, Cuchar), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::Float32, attr_name)
    ccall(:TF_SetAttrFloat, Void, (Ptr{Void}, Cstring, Cfloat), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::AbstractString, attr_name)
    value = String(value)
    ccall(:TF_SetAttrString, Void, (Ptr{Void}, Cstring, Ptr{Void}, Cint), desc.ptr, attr_name, value.data, sizeof(value))
end

function set_attr_list(desc::NodeDescription, attr_name, list::Vector{Int})
    ccall(:TF_SetAttrIntList, Void, (Ptr{Void}, Cstring, Ptr{Int64}, Cint), desc.ptr, attr_name, list, length(list))
end

function run(sess::Session, inputs, input_values, outputs, targets)
    status = Status()
    output_values = fill(C_NULL, length(outputs))
    input_tensors = [RawTensor(_) for _ in input_values]
    ccall((:TF_SessionRun), Void,
    (Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Cint, Ptr{Void}, Ptr{Ptr{Void}}, Cint, Ptr{Void}, Cint, Ptr{Void}, Ptr{Void}),
        sess.ptr,
        C_NULL,
        inputs,
        [_.ptr for _ in input_tensors],
        length(input_tensors),
        outputs,
        output_values,
        length(output_values),
        targets,
        length(targets),
        C_NULL,
        status.ptr)
    check_status(status)
    as_native = tensor->begin
        if ndims(tensor) == 0
            if eltype(tensor) == String
                String(tensor)
            else
                Number(tensor)
            end
        else
            Array(tensor)
        end
    end
    return [as_native(RawTensor(_)) for _ in output_values]
end

function run(sess::Session, outputs::AbstractVector, input_dict)
    inputs = Port[]
    input_values = []
    for (input, value) in input_dict
        push!(inputs, Port(input))
        push!(input_values, map(eltype(input), value))
    end
    output_ports = map(Port, outputs)
    run(sess, inputs, input_values, output_ports, [])
end

"""
Compute the result of one of more operations in the computation graph.
"""
function run(sess::Session, output::Tensor, input_dict)
    res = run(sess, [output], input_dict)
    if length(res)==1
        return res[1]
    else
        return res
    end
end

run(sess::Session, outputs) = run(sess, outputs, Dict())

function Base.eltype(t::RawTensor)
    tf_type = ccall((:TF_TensorType), TF_DataType, (Ptr{Void},), t.ptr)
    tf_to_jl_type(tf_type)
end

const type_map = Dict(TF_UINT8=>UInt8, TF_FLOAT=>Float32, TF_INT32=>Int32, TF_INT64=>Int64, TF_DOUBLE=>Float64, TF_STRING=>String, TF_BOOL=>Bool)
const inv_type_map = Dict(v=>k for (k, v) in type_map)

function tf_to_jl_type(dt::TF_DataType)
    return type_map[dt]
end

function jl_to_df_type(dt)
    return inv_type_map[dt]
end

function Base.convert(::Type{Array}, t::RawTensor)
    dims = ndims(t)
    data = ccall(:TF_TensorData, Ptr{eltype(t)}, (Ptr{Void},), t.ptr)
    if eltype(t) == String
        array = unsafe_wrap(Array, convert(Ptr{UInt8}, data), sizeof(t))
        b = IOBuffer(array)
        seekstart(b)
        offset = read(b, UInt64)
        len = varint_decode(b)
        raw_data = read(b, UInt8, len)
        [String(raw_data)]
    else
        if dims > 0
            convert_major_order(unsafe_wrap(Array, data, size(t)|>reverse))
        else
            unsafe_wrap(Array, data, size(t))
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
    ccall(:TF_GraphToGraphDef, Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}), graph.ptr, output.ptr, status.ptr)
    check_status(status)
    convert(Array, output)
end

function get_proto(node::Operation)
    output = Buffer()
    status = Status()
    ccall(:TF_NodeToNodeDef, Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}), node.ptr, output.ptr, status.ptr)
    check_status(status)
    convert(Array, output)
end

get_def_type(::Type{Operation}) = tensorflow.NodeDef
get_def_type(::Type{Graph}) = tensorflow.GraphDef

"""
Returns the definition of the given operation or graph, in returns of its properties
with respect to the computation graph.
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

function Base.show(io::IO, desc::tensorflow.NodeDef)
    # TODO: complete this
    println(io, "name: ", desc.name)
    println(io, "op: ", desc.op)
    for input_name in desc.input
        println(io, "input: ", input_name)
    end
    for (attr_name, attr_value) in desc.attr
        println(io, "attr {")
        println(io, "  key: ", attr_name)
        println(io, "  value {")
        print(io, "    ")
        if has_field(attr_value, :_type)
            println(io, "type: $(proto_type_map[attr_value._type])")
        elseif has_field(attr_value, :s)
            println(io, "string: $(String(attr_value.s))")
        elseif has_field(attr_value, :i)
            println(io, "int: $(attr_value.i)")
        elseif has_field(attr_value, :b)
            println(io, "bool: $(attr_value.b)")
        elseif has_field(attr_value, :f)
            println(io, "float: $(attr_value.f)")
        elseif has_field(attr_value, :tensor)
            t = attr_value.tensor
            println(io, "dtype: $(proto_type_map[t.dtype])")
            sep = "    "
            print(io, sep, "shape: ")
            println(io, [_.size for _ in t.tensor_shape.dim])
            print(io, sep, "content: ")
            show_tensor = k->begin
                f = getfield(t, k)
                if length(f) > 0
                    println(io, f)
                    return true
                end
                return false
            end
            for v in [:float_val, :double_val, :int_val, :int64_val, :bool_val, :half_val, :string_val, :tensor_content]
                if show_tensor(v)
                    break
                end
            end
        elseif has_field(attr_value, :shape)
            print(io, "shape: ")
            println(io, [_.size for _ in attr_value.shape.dim])
        end
        println(io, "  }")
        println(io, "}")
    end
end

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
    node_ptr = ccall(:TF_GraphNodeByName, Ptr{Void}, (Ptr{Void}, Cstring), graph.ptr, name)
    if node_ptr == C_NULL
        return Nullable{Operation}()
    else
        return Nullable(Operation(node_ptr))
    end
end

get_node_by_name(name) = get_node_by_name(get_def_graph(), name)

include("shape_inference.jl")

"""
Runs shape inference to return the shape of the tensor produced by the given operation.

Returns -1 if shape inference cannot infer a shape.

Note this runs *statically*. Use the `shape` operation to dynamically get the shape of an operation.
"""
function get_shape(n::AbstractTensor)
    t = Tensor(n)
    op = t.op
    fillin_operation(op)
    if op.op_name ∈ keys(shape_inferer)
        return shape_inferer[op.op_name](op)[t.value_index]
    else
        return -1
    end
end

const py_proc = Ref{Int}()

function spawn_py_process()
    addprocs(1)
    py_proc[] = nprocs()
    eval(Main, :(@everywhere using TensorFlow))
    path = joinpath(dirname(@__FILE__), "py.jl")
    remotecall_wait(py_proc[]) do
        eval(TensorFlow, quote
            include($path)
        end)
    end
    nothing
end

function gradients(y, x::AbstractArray)
    x_names = [node_name(_) for _ in x]
    y_name = node_name(y)
    graph_proto = get_def_graph() |> get_proto
    node_protos, grad_names = remotecall_fetch(py_proc[]) do
        py_gradients(graph_proto, x_names, y_name)
    end
    extend_graph(get_def_graph(), node_protos)
    return [Tensor(get_node_by_name(_)|>get, 1) for _ in grad_names]
end

gradients(y, x) = gradients(y, [x])[1]

function get_num_outputs(op::Operation)
    ccall(:TF_OperationNumOutputs, Cint, (Ptr{Void},), op.ptr) |> Int
end

function get_device(op::Operation)
    ccall(:TF_OperationDevice, Cstring, (Ptr{Void},), op.ptr) |> String
end

get_device(t::Tensor) = get_device(t.op)
