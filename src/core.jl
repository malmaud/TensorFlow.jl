using ProtoBuf

import Base: setindex!, getindex, run

const LIB_BASE = joinpath(dirname(@__FILE__), "..", "deps")

@static if is_apple()
    Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local-fastbuild", "bin", "tensorflow", "libtensorflow"), Libdl.RTLD_GLOBAL)
    Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local-fastbuild", "bin", "tensorflow", "c", "libc_api"), Libdl.RTLD_GLOBAL)
end

@static if is_linux()
    Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local_linux-fastbuild", "bin", "tensorflow", "libtensorflow"), Libdl.RTLD_GLOBAL)
    Libdl.dlopen(joinpath(LIB_BASE, "bazel-out", "local_linux-fastbuild", "bin", "tensorflow", "c", "libc_api"), Libdl.RTLD_GLOBAL)
end

include("py.jl")

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

function get_collection(g::Graph, name)
    g.collections[name]
end

add_to_collection(name, node) = add_to_collection(get_def_graph(), name, node)
get_collection(name) = get_collection(get_def_graph(), name)

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

get_def_graph() = def_graph

function set_def_graph(g)
    global def_graph
    def_graph = g
end

set_def_graph(Graph())

function as_default(f, g::Graph)
    old_def = get_def_graph()
    set_def_graph(g)
    f()
    set_def_graph(old_def)
end

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


type Tensor
    ptr::Ptr{Void}
    data::Array  # To avoid underlying data being GCed

    Tensor() = new()

    function Tensor(data::Array)
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

    function Tensor(data::Number)
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

    function Tensor(ptr::Ptr)
        this = new(ptr)
        finalizer(this, this->begin
            ccall((:TF_DeleteTensor), Void, (Ptr{Void},), this.ptr)
        end)
        return this
    end
end

Tensor(t::Tensor) = t

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

function Tensor(data::String)
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
    t = Tensor()
    t.data = [data]
    t.ptr = ptr
    return t
end


function Base.show(io::IO, t::Tensor)
    print(io, "Tensor: ")
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

function Base.ndims(t::Tensor)
    ccall((:TF_NumDims), Cint, (Ptr{Void},), t.ptr) |> Int
end

function Base.size(t::Tensor, dim::Integer)
    n = ndims(t)
    dim -= 1
    @assert dim < n
    ccall((:TF_Dim), Clonglong, (Ptr{Void}, Cint), t.ptr, dim)
end

function Base.size(t::Tensor)
    d = (size(t,_) for _ in 1:ndims(t))
    (d...)
end

function Base.sizeof(t::Tensor)
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

get_graph(desc::NodeDescription) = Nullable(desc.graph)

abstract AbstractNode

type Node <: AbstractNode
    ptr::Ptr{Void}
    graph::Nullable{Graph}

    function Node(desc::NodeDescription)
        status = Status()
        ptr = ccall((:TF_FinishNode), Ptr{Void}, (Ptr{Void}, Ptr{Void}), desc.ptr, status.ptr)
        check_status(status)
        new(ptr, Nullable(desc.graph))
    end

    Node(ptr::Ptr) = new(ptr, Nullable{Graph}())
end

get_graph(n::AbstractNode) = Node(n).graph

function Base.show(io::IO, n::Node)
    print(io, "<Node '$(node_name(n))' dtype=$(eltype(n))>")
end

# Replace this entire function once we can import protobufs into a graph
function Node(node_def::tensorflow.NodeDef)
    graph = get_def_graph()
    desc = NodeDescription(graph, node_def.op, node_def.name)
    if node_def.op == "DynamicStitch"
        inputs = []
        for input in node_def.input
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)|>get
            push!(inputs, input_node)
        end
        add_input(desc, [Port(inputs[1], 1), Port(inputs[2], 1)])
        add_input(desc, [Port(inputs[3], 1), Port(inputs[4], 1)])
        return Node(desc)
    end
    if node_def.op == "AddN"
        inputs = []
        for input in node_def.input
            input, port = parse_port_name(input)
            input_node = get_node_by_name(graph, input)|>get
            push!(inputs, input_node)
        end
        add_input(desc, [Port(_, 1) for _ in inputs])
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
            add_input(desc, Port(input_node |> get, port))
        end
    end
    if isdefined(node_def, :attr)  # TODO: complete this
        for (attr_name, attr) in node_def.attr
            if attr_name ∈ ("dtype", "T")
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
                    desc["value"] = Tensor(zeros(eltype(val),0))
                elseif length(dim) == 0
                    desc["value"] = Tensor(val[1])
                else
                    desc["value"] = Tensor(reshape(val, dim))
                end
            elseif attr_name == "keep_dims"
                desc["keep_dims"] = attr.b
            elseif attr_name == "N"
                desc["N"] = attr.i
            elseif attr_name == "transpose_a"
                desc["transpose_a"] = attr.b
            elseif attr_name == "transpose_b"
                desc["transpose_b"] = attr.b
            else
                warn("Unrecognized attribute $attr_name")
            end
        end
    end
    Node(desc)
end

node_name(node::AbstractNode) = ccall((:TF_NodeName), Cstring, (Ptr{Void},), Node(node).ptr) |> unsafe_string

function get_attr_value_proto(node::Node, attr_name)
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

Base.getindex(node::Node, attr_name) = get_attr_value_proto(node, attr_name)

const dt = tensorflow._DataType
const proto_type_map = Dict(dt.DT_FLOAT=>Float32, dt.DT_INT32=>Int32, dt.DT_DOUBLE=>Float64, dt.DT_INT64=>Int64, dt.DT_STRING=>String)

function Base.eltype(node::AbstractNode)
    node = Node(node)
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
    dt = tensorflow._DataType
    type_map = Dict(dt.DT_FLOAT=>Float32, dt.DT_INT32=>Int32, dt.DT_DOUBLE=>Float64, dt.DT_INT64=>Int64, dt.DT_STRING=>String)
    return proto_type_map[dtype]
end

immutable Port
    node_ptr::Ptr{Void}
    index::Int
end

Port(node::Node, index=1) = Port(node.ptr, index-1)  # Convert between 1-based (Julia) and 0-based (Python) indexing of port numbers


function add_input(desc::NodeDescription, input::Port)
    ccall((:TF_AddInput), Void, (Ptr{Void}, Port), desc.ptr, input)
end

add_input(desc::NodeDescription, node::Node) = add_input(desc, Port(node))

function add_input(desc::NodeDescription, inputs::Vector{Port})
    ccall((:TF_AddInputList), Void, (Ptr{Void}, Ptr{Void}, Cint), desc.ptr, inputs, length(inputs))
end

function setindex!(desc::NodeDescription, tensor::Tensor, attr_name)
    status = Status()
    ccall((:TF_SetAttrTensor), Void, (Ptr{Void}, Cstring, Ptr{Void}, Ptr{Void}), desc.ptr, attr_name, tensor.ptr, status.ptr)
    check_status(status)
end

function setindex!(desc::NodeDescription, tensors::Vector{Tensor}, attr_name)
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
    input_tensors = [Tensor(_) for _ in input_values]
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
    return [as_native(Tensor(_)) for _ in output_values]
end

function run(sess::Session, outputs::AbstractVector, input_dict)
    inputs = map(input->Port(input, 1), keys(input_dict))
    input_values = collect(values(input_dict))
    output_ports = map(output->Port(output, 1), outputs)
    run(sess, inputs, input_values, output_ports, [])
end

run(sess::Session, output::Node, input_dict) = run(sess, [output], input_dict)[1]

run(sess::Session, outputs) = run(sess, outputs, Dict())

function Base.eltype(t::Tensor)
    tf_type = ccall((:TF_TensorType), TF_DataType, (Ptr{Void},), t.ptr)
    tf_to_jl_type(tf_type)
end

const type_map = Dict(TF_UINT8=>UInt8, TF_FLOAT=>Float32, TF_INT32=>Int32, TF_INT64=>Int64, TF_DOUBLE=>Float64, TF_STRING=>String)
const inv_type_map = Dict(v=>k for (k, v) in type_map)

function tf_to_jl_type(dt::TF_DataType)
    return type_map[dt]
end

function jl_to_df_type(dt)
    return inv_type_map[dt]
end

function Base.convert(::Type{Array}, t::Tensor)
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

function Base.convert{T<:Union{Number, String}}(::Type{T}, t::Tensor)
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

function get_proto(node::Node)
    output = Buffer()
    status = Status()
    ccall(:TF_NodeToNodeDef, Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}), node.ptr, output.ptr, status.ptr)
    check_status(status)
    convert(Array, output)
end

get_def_type(::Type{Node}) = tensorflow.NodeDef
get_def_type(::Type{Graph}) = tensorflow.GraphDef

function get_def(n::Union{Node, Graph})
    p = get_proto(n)
    b = IOBuffer()
    write(b, p)
    seekstart(b)
    desc = get_def_type(typeof(n))()
    readproto(b, desc)
    return desc
end

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

function get_node_by_name(graph::Graph, name::AbstractString)
    name, port = parse_port_name(name)
    node_ptr = ccall(:TF_GraphNodeByName, Ptr{Void}, (Ptr{Void}, Cstring), graph.ptr, name)
    if node_ptr == C_NULL
        return Nullable{Node}()
    else
        return Nullable(Node(node_ptr))
    end
end

get_node_by_name(name) = get_node_by_name(get_def_graph(), name)

function get_shape(n::AbstractNode)
    n = Node(n)
    local shape
    try
        shape = n["shape"]
    catch
        try
            shape = n["value"].tensor.tensor_shape
        catch
            return -1
        end
    end
    return [_.size for _ in shape.dim]
end
