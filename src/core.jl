using ProtoBuf
using PyCall
using Compat
using Compat.Iterators
using MacroTools
using AutoHashEquals
using Libdl

import Base: setindex!, getindex, run, ==

const LIB_BASE = joinpath(dirname(@__FILE__), "..", "deps")
include("py.jl")

const LIBTF_PTR = Base.RefValue(C_NULL)

macro tfcall(sym, ret, args, vals...)
    quote
        tf_path = get(ENV,
            "LIBTENSORFLOW",
            joinpath(LIB_BASE, "usr", "bin", "libtensorflow"))
        if LIBTF_PTR[] == C_NULL
            LIBTF_PTR[] = Libdl.dlopen(expanduser(tf_path))
        end
        func = Libdl.dlsym(LIBTF_PTR[], $sym)
        ccall(func, $(esc(ret)), $(esc(args)), $(esc.(vals)...))
    end
end

"""
    @required(keywords...)

Macro that raises an error if any of the passed-in symbols equal 'nothing'.
Useful for marking keyword arguments as required.
"""
macro required(keywords...)
    blocks = []
    for keyword in keywords
        push!(blocks, quote
            err_msg = string($(string(keyword)), " is required")
            $(esc(keyword)) === nothing && error(err_msg)
        end)
    end
    quote
        $(blocks...)
    end
end


mutable struct Status
    ptr::Ptr{Cvoid}
    function Status()
        ptr = @tfcall(:TF_NewStatus, Ptr{Cvoid}, ())
        this = new(ptr)
        finalizer(this) do status
            @tfcall(:TF_DeleteStatus, Cvoid, (Ptr{Cvoid},), status.ptr)
        end
        this
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, s::Status) = s.ptr

function get_code(s::Status)
    code = @tfcall(:TF_GetCode, Cint, (Ptr{Cvoid},), s.ptr)
    return TF_Code(code)
end

@auto_hash_equals struct DevicePart{IndexType}
    kind::Vector{String}
    index::IndexType
end

device_index_from_zero(part::DevicePart{Int}) = "$(join(part.kind, ":")):$(part.index-1)"
device_index_from_zero(part::DevicePart) = "$(join(part.kind, ":")):$(part.index)"

@auto_hash_equals struct Device
    parts::Vector{DevicePart}
end

Device() = Device(DevicePart[])

function DevicePart(s::AbstractString)
    parts = split(s, ":")
    length(parts) >= 2 || error("Invalid device: $s")
    kind = String.(parts[1:end-1])
    index_part = String(parts[end])
    maybe_index = tryparse(Int, index_part)
    index = something(maybe_index, index_part)
    DevicePart(kind, index)
end

function device_index_from_zero(device::Device)
    b = IOBuffer()
    for part in device.parts
        print(b, "/")
        print(b, device_index_from_zero(part))
    end
    String(take!(b))
end

Base.show(io::IO, part::DevicePart) = print(io, "$(join(part.kind, ":")):$(part.index)")

function Device(s::AbstractString)
    device = Device()
    for part in split(s, "/")
        isempty(part) && continue
        push!(device.parts, DevicePart(part))
    end
    device
end

function Base.show(io::IO, device::Device)
    print(io, "/")
    join(io, device.parts, "/")
end

macro device_str(s)
    Device(s)
end


"""
    with_device(function, device)

Specifies the default device to use for ops created in `function`.

In contrast to the Python version, devices use 1-based indexing (eg, "gpu:1"
is the first GPU).

Intended to be used with `do` syntax:

```
with_device("gpu:2") do  # Use the second GPU
    x = constant(1.0)
end
```
"""
function with_device(f, device::Device)
    g = get_def_graph()
    push!(g.op_context.devices, device)
    try
        f()
    finally
        pop!(g.op_context.devices)
    end
end

with_device(f, device) = with_device(f, Device(device))

struct OperationContext
    control_ops::Vector{Vector{Any}}  # Can't make Operation to break type cycle
    names::Vector{String}
    while_context::Vector{tensorflow.WhileContextDef}
    devices::Vector{Device}
    is_top_level::Ref{Bool}
end

@auto_hash_equals struct TensorShape
    dims::Vector{Union{Int, Missing}}
    rank_unknown::Bool
end

TensorShape(dims::Vector) = TensorShape(dims, false)
TensorShape(::Missing) = TensorShape([], true)

function Base.copy(shape::TensorShape)
    TensorShape(copy(shape.dims), shape.rank_unknown)
end

"""
A TensorFlow computation graph
"""
mutable struct Graph
    ptr::Ptr{Cvoid}
    collections::Dict{Symbol, Any}
    name_idx::Dict{String, Int}
    op_context::OperationContext

    function Graph()
        ptr = @tfcall(:TF_NewGraph, Ptr{Cvoid}, ())
        collections = Dict{Symbol, Any}()
        collections[:Variables] = []
        collections[:TrainableVariables] = []
        collections[:Summaries] = []
        collections[:QueueRunners] = []
        collections[:while_context] = []
        self = new(ptr, collections, Dict{String, Int}(), OperationContext(Vector{Operation}[], String[], tensorflow.WhileContextDef[], Device[], Ref(false)))
        finalizer(self) do self
            @tfcall(:TF_DeleteGraph, Cvoid, (Ptr{Cvoid},), self.ptr)
        end
        self
    end
end

function Base.show(io::IO, g::Graph)
    print(io, "Graph($(g.ptr))")
end


function with_def_graph(ex)
    ex = longdef(ex)
    (@capture ex begin
        function f_(args__; kwargs__)
            body_
        end
    end) ||
    (@capture ex begin
        function f_(args__)
            body_
        end
    end) ||
    error("Improper use of with_def_graph")
    (kwargs === nothing) && (kwargs = [])
    new_args = args[2:end]

    quote
        function $f($(new_args...); $(kwargs...))
            $f(TensorFlow.get_def_graph(), $(new_args...); $(kwargs...))
        end
    end
end

"""
    @with_def_graph

Defaults the first parameter of the given function to `get_def_graph`.
"""
macro with_def_graph(ex)
    new_func = with_def_graph(ex)
    quote
        @Base.__doc__($(esc(ex)))
        $(esc(new_func))
    end
end

@with_def_graph function add_to_collection(g::Graph, name, node)
    if !haskey(g.collections, name)
        g.collections[name] = []
    end
    push!(g.collections[name], node)
end

"""
    get_collection(g::Graph, name)

Returns a collection attached to the graph `g` named `name`
"""
function get_collection end

@with_def_graph function get_collection(g::Graph, name)
    if !haskey(g.collections, name)
        return []
    end
    return g.collections[name]
end

const DEBUG_EXTEND_GRAPH = false

function Base.convert(::Type{tensorflow.NodeDef}, proto::DenseVector{UInt8})
    b = IOBuffer()
    write(b, proto)
    seekstart(b)
    node_def = tensorflow.NodeDef()
    readproto(b, node_def)
    node_def
end

@with_def_graph function extend_graph(graph::Graph, node_def_bytes)
    new_graph = tensorflow.GraphDef()
    setproperty!(new_graph, :node, tensorflow.NodeDef[])
    import_options = GraphImportOptions()
    ph_names = Set{String}()
    for node_def in convert.(tensorflow.NodeDef, node_def_bytes)
        if get_node_by_name(graph, node_def.name) === nothing
            # Hack to deal with imported nodes which have
            # colocation dependencies on existing nodes
            if has_field(node_def, :attr) && haskey(node_def.attr, "_class")
                classes = node_def.attr["_class"].list.s
                inds = Int[]
                for (ind, val) in enumerate(classes)
                    m = match(r"^loc:@(.*)", String(val))
                    if m !== nothing
                        loc_name = m[1]
                        if get_node_by_name(graph, loc_name) !== nothing
                            push!(inds, ind)
                        end
                    end
                end
                deleteat!(classes, inds)
            end
            push!(new_graph.node, node_def)
            for (i, input) in enumerate(node_def.input)
                name, dest_port = parse_port_name(input)
                is_control = name[1] == '^'
                if is_control
                    name = name[2:end]
                    dest_port = 0
                    source_port = 0
                else
                    source_port = 1
                end
                existing_node = get_node_by_name(graph, name)
                if existing_node !== nothing
                    local new_name
                    for name_id in Iterators.countfrom()
                        new_name = "$(name)__placeholder__$(name_id)_$dest_port"
                        get_node_by_name(graph, new_name) === nothing && break
                    end
                    if is_control
                        input_name = string("^", new_name)
                    else
                        input_name = new_name
                    end
                    node_def.input[i] = input_name

                    import_options.input_mapping[(new_name, source_port)] = Tensor(existing_node, dest_port)
                    new_ph = tensorflow.NodeDef()
                    setproperty!(new_ph, :name, new_name)
                    if is_control
                        setproperty!(new_ph, :op, "NoOp")
                    else
                        setproperty!(new_ph, :op, "Placeholder")
                        setproperty!(new_ph, :attr, Dict{AbstractString, tensorflow.AttrValue}())
                        new_ph.attr["dtype"] = tensorflow.AttrValue()
                        source_type = tensorflow._DataType.DT_FLOAT
                        for key in ["T", "SrcT"]
                            if key ∈ keys(node_def.attr)
                                source_type = node_def.attr[key]._type
                                break
                            end
                        end
                        setproperty!(new_ph.attr["dtype"], :_type, source_type)
                    end
                    if new_name ∉ ph_names
                        push!(new_graph.node, new_ph)
                        push!(ph_names, new_name)
                    end
                end
            end
        end
    end
    import_graph_def(graph, new_graph, import_options)
end

@with_def_graph function extend_graph(graph::Graph, node_def::tensorflow.NodeDef)
    extend_graph(graph, [node_def])
end


mutable struct SessionOptions
    ptr::Ptr{Cvoid}

    function SessionOptions()
        ptr = @tfcall(:TF_NewSessionOptions, Ptr{Cvoid}, ())
        self = new(ptr)
        set_tf_finalizer(self)
        self
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, o::SessionOptions) = o.ptr

function set_tf_finalizer(options::SessionOptions)
    finalizer(options) do options
        @tfcall(:TF_DeleteSessionOptions, Cvoid, (Ptr{Cvoid},), options.ptr)
    end
    options
end

struct TFException <: Exception
    status::Status
end

function check_status(status)
    if get_code(status) ≠ TF_OK
        throw(TFException(status))
    end
    nothing
end

const def_graph = Ref{Graph}()

const upgrade_check_needed = Ref(true)

function upgrade_check(v)
    if upgrade_check_needed[]
        if tf_version() < v
            @warn("You are using an old version version of the TensorFlow binary library. It is recommened that you upgrade with Pkg.build(\"TensorFlow\") or various
            errors may be encountered.\n You have $(tf_version()) and the new version is $v.")
        end
        upgrade_check_needed[] = false
    end
end

"""
    get_def_graph()

Returns the default computation graph, an object of type `Graph`.

See also `as_default` for setting the default graph
"""
function get_def_graph()
    upgrade_check(v"1.2.0")  # This is here instead of in __init__ to avoid issues
                             # with precompilation.
    has_def_graph() || (def_graph[] = Graph())
    def_graph[]
end

has_def_graph() = isdefined(def_graph, :x)

"""
    set_def_graph(g)

Sets the default computation graph to `g`.

See also `get_def_graph`, `as_default`
"""
function set_def_graph(g)
    def_graph[] = g
end

"""
    as_default(f, g::Graph)

For the duration of the function `f`
temporarily sets the default computational graph to `g`.

Suggested usage is via a do-block:
```julia
    as_default(graph1) do
        x = constant(5)
        y = 2*x
    end
```

In that example the nodes `x` and `y` were added to the Graph `graph1`.

see also See also `get_def_graph`
"""
function as_default(f, g::Graph)
    old_def = get_def_graph()
    set_def_graph(g)
    try
        f()
    finally
        set_def_graph(old_def)
    end
end

"""
A TensorFlow session.
"""
mutable struct Session
    ptr::Ptr{Cvoid}
    graph::Graph

    function Session(graph, config=nothing; target=nothing)
        set_def_graph(graph)
        options = SessionOptions()
        if config !== nothing
            b = IOBuffer()
            writeproto(b, config)
            seekstart(b)
            proto = read(b)
            config_status = Status()
            @tfcall(:TF_SetConfig, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}), options.ptr, proto, sizeof(proto), config_status.ptr)
            check_status(config_status)
        end
        if target !== nothing
            @tfcall(:TF_SetTarget, Cvoid, (Ptr{Cvoid}, Cstring), options, target)
        end
        status = Status()
        ptr = @tfcall(:TF_NewSession, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), graph.ptr, options.ptr, status.ptr)
        this = new(ptr, graph)
        check_status(status)
        finalizer(this) do self
            close(self)
        end
        return this
    end

    function Session(;config=nothing, allow_growth=false, graph=get_def_graph(), target=nothing)
        if config === nothing
            config = tensorflow.ConfigProto()
            gpu_config = tensorflow.GPUOptions()
            setproperty!(gpu_config, :allow_growth, allow_growth)
            setproperty!(config, :gpu_options, gpu_config)
        end
        Session(graph, config, target=target)
    end
end
Base.unsafe_convert(::Type{Ptr{Cvoid}}, s::Session) = s.ptr

"""
    close(sess::Session)

Closes the TensorFlow session, freeing the associated computational resources.
"""
function Base.close(sess::Session)
    if sess.ptr != C_NULL
        status = Status()
        @tfcall(:TF_DeleteSession, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), sess.ptr, status.ptr)
        check_status(status)
        sess.ptr = C_NULL
    end
    return nothing
end

mutable struct DeviceList
    ptr::Ptr{Cvoid}
    count::Int
end

function DeviceList(s::Session)
    status = Status()
    ptr = @tfcall(:TF_SessionListDevices, Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}), s, status)
    check_status(status)
    count = @tfcall(:TF_DeviceListCount, Cint, (Ptr{Cvoid},),
        ptr)
    this = DeviceList(ptr, count)
    finalizer(this) do self
        close(self)
    end
    this
end

struct DeviceInfo
    name::String
    device_type::String
    membytes::Int64
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, devices::DeviceList) = devices.ptr
Base.close(devices::DeviceList) = @tfcall(:TF_DeleteDeviceList, Cvoid, (Ptr{Cvoid},), devices)

Base.length(devices::DeviceList) = devices.count
function Base.iterate(devices::DeviceList, idx::UInt=UInt(0))
    idx >= length(devices) && return nothing
    status = Status()
    name = @tfcall(:TF_DeviceListName, Ptr{UInt8}, (Ptr{Cvoid}, Cint, Ptr{Cvoid}), devices, idx, status)
    check_status(status)
    device_type = @tfcall(:TF_DeviceListType, Ptr{UInt8}, (Ptr{Cvoid}, Cint, Ptr{Cvoid}), devices, idx, status)
    check_status(status)
    membytes = @tfcall(:TF_DeviceListMemoryBytes, Int64, (Ptr{Cvoid}, Cint, Ptr{Cvoid}), devices, idx, status)
    check_status(status)
    DeviceInfo(unsafe_string(name), unsafe_string(device_type), membytes), idx + 1
end

mutable struct Buffer
    ptr::Ptr{Cvoid}

    function Buffer(s::Vector{UInt8})
        self = new()
        self.ptr = @tfcall(:TF_NewBufferFromString, Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t), pointer(s), sizeof(s))
        set_tf_finalizer(self)
        return self
    end

    function Buffer()
        self = new()
        self.ptr = @tfcall(:TF_NewBuffer, Ptr{Cvoid}, ())
        set_tf_finalizer(self)
        return self
    end

    Buffer(ptr) = new(ptr)
end

function set_tf_finalizer(buffer::Buffer)
    finalizer(buffer) do buffer
        @tfcall(:TF_DeleteBuffer, Cvoid, (Ptr{Cvoid},), buffer.ptr)
    end
end

struct BufferStruct
    data::Ptr{UInt8}
    len::Csize_t
    deallocator::Ptr{Cvoid}
end

function getindex(b::Buffer)
    @tfcall(:TF_GetBuffer, BufferStruct, (Ptr{Cvoid},), b.ptr)
end

function Base.convert(::Type{Array}, buf::Buffer)
    struct_ = buf[]
    array = unsafe_wrap(Array, struct_.data, (struct_.len,))
    copy(array)
end

const c_deallocator = Ref{Ptr}()

"""
    convert_major_order(array)

Convert from row-major to column-major or vice-versa
"""
function convert_major_order(array)
    permutedims(array, length(size(array)):-1:1)
end

struct EmptyTensorError <: Exception
end

function Base.show(io::IO, err::EmptyTensorError)
    print(io, "Creating tensors from empty arrays is not allowed")
end

mutable struct RawTensor
    ptr::Ptr{Cvoid}
    data::Array  # To avoid underlying data being GCed

    RawTensor() = new()

    function RawTensor(data::Array)
        isempty(data) && throw(EmptyTensorError())
        dims = [size(data)...]
        dt = jl_to_df_type(eltype(data))
        data = convert_major_order(data)
        ptr = @tfcall(:TF_NewTensor, Ptr{Cvoid}, (Cint, Ptr{Cint}, Cint, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Int(dt),
            pointer(dims),
            length(dims),
            pointer(data),
            sizeof(data),
            c_deallocator[],
            C_NULL)
        self = new(ptr, data)
        set_tf_finalizer(self)
        return self
    end

    function RawTensor(data::Number)
        dims = Cint[]
        dt = jl_to_df_type(eltype(data))
        data_boxed = [data]
        ptr = @tfcall(:TF_NewTensor, Ptr{Cvoid}, (Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Int(dt),
            pointer(dims),
            length(dims),
            pointer(data_boxed),
            sizeof(data_boxed),
            c_deallocator[],
            C_NULL)
        self = new(ptr, data_boxed)
        set_tf_finalizer(self)
        return self
    end

    function RawTensor(ptr::Ptr)
        self = new(ptr)
        set_tf_finalizer(self)
        return self
    end
end

function set_tf_finalizer(tensor::RawTensor)
    finalizer(tensor) do tensor
        @tfcall(:TF_DeleteTensor, Cvoid, (Ptr{Cvoid},), tensor.ptr)
    end
end

RawTensor(data::AbstractArray) = RawTensor(collect(data))

RawTensor(t::RawTensor) = t

Base.unsafe_convert(::Type{Ptr{Cvoid}}, t::RawTensor) = t.ptr

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

function tf_string_encode(src::DenseVector{UInt8})
    dest_length = @tfcall(:TF_StringEncodedSize, Csize_t, (Csize_t,), length(src)) |> Int
    dest = Vector{UInt8}(undef, dest_length)
    status = Status()
    @tfcall(:TF_StringEncode, Csize_t,
        (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}),
        src, length(src), dest, length(dest), status.ptr)
    check_status(status)
    dest
end

tf_string_encode(src) = tf_string_encode(codeunits(src))

function tf_string_decode(src::DenseVector{UInt8})
    status = Status()
    dst = Ref{Ptr{UInt8}}()
    dst_len = Ref{Csize_t}()
    @tfcall(:TF_StringDecode, Csize_t,
        (Ptr{Cvoid}, Csize_t, Ref{Ptr{UInt8}}, Ref{Csize_t}, Ptr{Cvoid}),
        src, length(src), dst, dst_len, status.ptr)
    check_status(status)
    copy(unsafe_wrap(Array, dst[], Int(dst_len[])))
end

tf_string_decode(src) = tf_string_decode(codeunits(src))
tf_string_decode(T, src) = T(tf_string_decode(src))

# cf this section of c_api.h in upstream tensorflow/c_api.h
#=
// --------------------------------------------------------------------------
// TF_Tensor holds a multi-dimensional array of elements of a single data type.
// For all types other than TF_STRING, the data buffer stores elements
// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
//   element 0:   index (0, ..., 0)
//   element 1:   index (0, ..., 1)
//   ...
//
// The format for TF_STRING tensors is:
//   start_offset: array[uint64]
//   data:         byte[...]
//
//   The string length (as a varint), followed by the contents of the string
//   is encoded at data[start_offset[i]]]. TF_StringEncode and TF_StringDecode
//   facilitate this encoding.

=#
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
    data = map(tf_string_encode ,data)
    encoded_buf = IOBuffer()
    pos = 0
    for str in data
        write(encoded_buf, UInt64(pos))
        pos += length(str)
    end
    for str in data
        write(encoded_buf, str)
    end
    data_encoded = take!(encoded_buf)
    dt = jl_to_df_type(String)
    ptr = @tfcall(:TF_NewTensor, Ptr{Cvoid}, (Cint, Ptr{Int64}, Cint, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
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

function RawTensor(data::Array{Vector{UInt8}})
    RawTensor(String.(data))
end


function Base.ndims(t::RawTensor)
    @tfcall(:TF_NumDims, Cint, (Ptr{Cvoid},), t.ptr) |> Int
end

function Base.size(t::RawTensor, dim::Integer)
    n = ndims(t)
    dim -= 1
    @assert dim < n
    @tfcall(:TF_Dim, Clonglong, (Ptr{Cvoid}, Cint), t.ptr, dim)
end

function Base.size(t::RawTensor)
    d = (size(t, x) for x in 1:ndims(t))
    (d...,)
end

function Base.sizeof(t::RawTensor)
    @tfcall(:TF_TensorByteSize, Csize_t, (Ptr{Cvoid},), t.ptr) |> Int
end

function set_device(node_desc, device)
    @tfcall(:TF_SetDevice, Cvoid,
        (Ptr{Cvoid}, Cstring),
        node_desc.ptr, device)
end

set_device(node_desc, device::Device) = set_device(node_desc, device_index_from_zero(device))

mutable struct NodeDescription
    ptr::Ptr{Cvoid}
    graph::Graph

    function NodeDescription(graph, op_type, full_name)
        desc = @tfcall(:TF_NewOperation, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring, Cstring), graph.ptr, op_type, full_name)
        self = new(desc, graph)
        for control_op_set in graph.op_context.control_ops
            for control_op in control_op_set
                add_control_input(self, control_op)
            end
        end
        isempty(graph.op_context.devices) || set_device(self, graph.op_context.devices[end])
        self
    end

end

NodeDescription(op_type, node_name) = NodeDescription(get_def_graph(), op_type, node_name)

function get_cur_node_name()
    join(get_def_graph().op_context.names, "/")
end

function NodeDescription(op_type)
    name = get_cur_node_name()
    NodeDescription(op_type, name)
end

get_graph(desc::NodeDescription) = desc.graph

abstract type AbstractOperation end

"""
An operation in the computation graph.
"""
mutable struct Operation <: AbstractOperation
    ptr::Ptr{Cvoid}
    graph::Union{Graph, Nothing}
    op_name::String
    name::String
    Operation() = new()
end

==(op1::Operation, op2::Operation) = op1.ptr == op2.ptr
Base.hash(op::Operation, h::UInt) = hash(Operation, hash(op.ptr, h))

struct Port
    node_ptr::Ptr{Cvoid}
    index::Int
end

function get_num_inputs(op::Operation)
    @tfcall(:TF_OperationNumInputs, Cint, (Ptr{Cvoid},), op.ptr) |> Int
end

struct InputOutOfRangeError <: Exception
    op::Operation
    index::Int
end

function Base.show(io::IO, err::InputOutOfRangeError)
    fillin(err.op)
    num_inputs = get_num_inputs(err.op)
    print(io, "Index $(err.index) is out of range. Operation '$(err.op.op_name)' only has $num_inputs inputs.")
end

function get_input(op::Operation, idx)
    num_inputs = get_num_inputs(op)
    if idx < 1 || idx > num_inputs
        throw(InputOutOfRangeError(op, idx))
    end
    port = Port(op.ptr, idx-1)
    in_port = @tfcall(:TF_OperationInput, Port, (Port,), port)
    out_tensor = Tensor(in_port)
    out_op = get_op(out_tensor)
    out_op.graph = op.graph
    fillin(out_op)
    out_tensor
end

function get_num_control_inputs(op::Operation)
    @tfcall(:TF_OperationNumControlInputs, Cint, (Ptr{Cvoid},), op.ptr) |> Int
end

function get_control_inputs(op::Operation)
    N = get_num_control_inputs(op)
    ptrs = Vector{Ptr{Cvoid}}(undef, N)
    N_out = @tfcall(:TF_OperationGetControlInputs, Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Cint),
                    op.ptr, ptrs, N)
    out = Vector{Operation}()
    for n in 1:N_out
        op_out = Operation(ptrs[n])
        fillin(op_out)
        op_out.graph = op.graph
        push!(out, op_out)
    end
    out
end

function get_input_list_length(op::Operation, arg_name)
    status = Status()
    out = @tfcall(:TF_OperationInputListLength, Cint, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), op.ptr, arg_name, status.ptr)
    check_status(status)
    Int(out)
end

struct AttrMetadata
    is_list::Bool
    list_size::Int64
    _type::Int32
    total_size::Int64
end

function get_attr_metadata(op::Operation, attr)
    status = Status()
    out = @tfcall(:TF_OperationGetAttrMetadata, AttrMetadata, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), op.ptr, attr, status.ptr)
    check_status(status)
    out
end

function get_attr(op::Operation, attr, ::Type{Int})
    out = Ref{Int}()
    status = Status()
    @tfcall(:TF_OperationGetAttrInt, Cvoid, (Ptr{Cvoid}, Cstring, Ref{Int}, Ptr{Cvoid}), op.ptr, attr, out, status.ptr)
    check_status(status)
    out[]
end

function get_attr(op::Operation, attr, ::Type{Array})
    out = Ref{Ptr{Cvoid}}()
    status = Status()
    @tfcall(:TF_OperationGetAttrTensor, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), op.ptr, attr, out, status.ptr)
    check_status(status)
    convert(Array, RawTensor(out[]))
end

function get_attr(op::Operation, attr, ::Type{Bool})
    out = Ref{Bool}()
    status = Status()
    @tfcall(:TF_OperationGetAttrBool, Cvoid, (Ptr{Cvoid}, Cstring, Ref{Bool}, Ptr{Cvoid}), op.ptr, attr, out, status.ptr)
    check_status(status)
    out[]
end

function get_attr(op::Operation, attr, ::Type{Vector{Int}})
    meta = get_attr_metadata(op, attr)
    out = Vector{Int}(undef, meta.list_size)
    status = Status()
    @tfcall(:TF_OperationGetAttrIntList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Int}, Cint, Ptr{Cvoid}), op.ptr, attr, out, length(out), status.ptr)
    check_status(status)
    out
end

function get_attr(op::Operation, attr, ::Type{String})
    meta = get_attr_metadata(op, attr)
    out = Vector{UInt8}(undef, meta.total_size)
    status = Status()
    @tfcall(:TF_OperationGetAttrString, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{UInt8}, Cint, Ptr{Cvoid}), op.ptr, attr, out, length(out), status.ptr)
    check_status(status)
    String(out)
end

function fillin(op::Operation)
    op.name = @tfcall(:TF_OperationName, Cstring, (Ptr{Cvoid},), op.ptr) |> unsafe_string
    op.op_name = @tfcall(:TF_OperationOpType, Cstring, (Ptr{Cvoid},), op.ptr) |> unsafe_string
end

function with_op_name(f, name, def_name="Node")
    if name === nothing
        name = get_name(def_name)
    end
    g = get_def_graph()
    push!(g.op_context.names, name)
    try
        f()
    finally
        pop!(g.op_context.names)
    end
end


"""
     with_op_control(f, control_ops)

Any ops declared inside `f` will not execute until after all op listed in `control_ops`.
This enforces order of execution.
It is useful if the op in `f` may depend on the execution of one or more of the `control_ops` first.
see also [Python Docs](https://www.tensorflow.org/versions/r0.12/api_docs/python/framework/core_graph_data_structures#Graph.control_dependencies)
"""
function with_op_control(f, control_ops)
    g = get_def_graph()
    push!(g.op_context.control_ops, control_ops)
    try
        f()
    finally
        pop!(g.op_context.control_ops)
    end
end

function with_top_level(f)
    g = get_def_graph()
    is_top_level = g.op_context.is_top_level
    old_level = is_top_level[]
    is_top_level[] = true
    try
        with_no_op_control() do
            f()
        end
    finally
        is_top_level[] = old_level
    end
end

function with_no_op_control(f)
    g = get_def_graph()
    control_ops = deepcopy(g.op_context.control_ops)
    empty!(g.op_context.control_ops)
    try
        f()
    finally
        append!(g.op_context.control_ops, control_ops)
    end
end

function Operation(desc::NodeDescription)
    self = Operation()
    status = Status()
    ptr = @tfcall(:TF_FinishOperation, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), desc.ptr, status.ptr)
    check_status(status)
    self.ptr = ptr
    graph = desc.graph
    self.graph = graph
    fillin(self)

    if graph.op_context.is_top_level[]
        add_to_collection(graph, :TopLevel, self)
    end

    return self
end

function Operation(ptr::Ptr)
    self = Operation()
    self.ptr = ptr
    self.graph = nothing
    fillin(self)
    return self
end

struct NodeNameNotFound <: Exception
    name::String
end

function Base.show(io::IO, err::NodeNameNotFound)
    print(io, "Node $(err.name) not found in graph")
end

get_graph(n::AbstractOperation) = Operation(n).graph

function load_proto(tensor::tensorflow.TensorProto)
    dtype = tensor.dtype
    dim = (Int[x.size for x in tensor.tensor_shape.dim]...,)
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
    elseif dtype == tensorflow._DataType.DT_BOOL
        val = tensor.bool_val
    else
        #warn("Unrecognized datatype $dtype")
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
        RawTensor(zeros(eltype(val),0))
    elseif length(dim) == 0
        RawTensor(val[1])
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
        RawTensor(reshape(val, reverse(dim)) |> convert_major_order)
    end
end

function load_proto(shape::tensorflow.TensorShapeProto)
    if shape.unknown_rank
        ShapeInference.TensorShape(nothing)
    else
        ShapeInference.TensorShape(shape.dim)
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
    elseif has_field(value, :_type)
        type_ = value._type
        get(proto_type_map, type_) do
            @warn "Unrecognized type. Defaulting to Float32." type_
            Float32
        end
    end
end

"""
    node_name(node::Operation)

Returns the name of a node in the computation graph.
"""
node_name(node::Operation) = @tfcall(:TF_OperationName, Cstring, (Ptr{Cvoid},), node.ptr) |> unsafe_string


function get_attr_value_proto(node::Operation, attr_name)
    buf = Buffer()
    status = Status()
    @tfcall(:TF_OperationGetAttrValueProto, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Ptr{Cvoid}), node.ptr, attr_name, buf.ptr, status.ptr)
    check_status(status)
    proto = convert(Array, buf)
    b = IOBuffer()
    write(b, proto)
    seekstart(b)
    val = tensorflow.AttrValue()
    readproto(b, val)
    return val
end

Base.getindex(node::Operation, attr_name) = get_attr_value_proto(node, attr_name)

const dt = tensorflow._DataType
const proto_type_map = Dict(
    dt.DT_FLOAT=>Float32,
    dt.DT_INT32=>Int32,
    dt.DT_DOUBLE=>Float64,
    dt.DT_INT64=>Int64,
    dt.DT_STRING=>String,
    dt.DT_BOOL=>Bool,
    dt.DT_UINT8=>UInt8,
    dt.DT_COMPLEX64=>ComplexF32,
    dt.DT_COMPLEX128=>ComplexF64)

abstract type AbstractTensor{T} end

"""
Represents the output of an operation in the computation graph
"""
@auto_hash_equals struct Tensor{T} <: AbstractTensor{T}
    op::Operation
    value_index::Int
end

function Tensor(op::Operation, value_index::Int)
    base_tensor = Tensor{Any}(op, value_index)
    Tensor{get_output_type(base_tensor)}(op, value_index)
end

Tensor(op::Operation) = Tensor(op, 1)

Tensor(value) = convert(Tensor, value)
Tensor{T}(value) where {T}  = convert(Tensor{T}, value)

Base.convert(::Type{Tensor{T}}, value::AbstractTensor) where {T} = convert(Tensor{T}, convert(Tensor, value))
Base.convert(::Type{Tensor}, value) = constant(value)
Base.convert(::Type{Tensor}, value::Tensor) = value

Base.convert(::Type{Tensor{T}}, value::Tensor{R}) where {T, R} = cast(value, T)
Base.convert(::Type{Tensor{T}}, value::Tensor{T}) where {T} = value
Base.convert(::Type{Tensor{Any}}, value::Tensor{R}) where {R} = value

Base.convert(::Type{Tensor{T}}, value) where {T} =  convert(Tensor{T}, constant(value))

get_graph(t::AbstractTensor) = Tensor(t).op.graph

node_name(t::AbstractTensor) = (node_name(Tensor(t).op), Tensor(t).value_index)

function operation_output_type(port::Port)
    @tfcall(:TF_OperationOutputType, TF_DataType, (Port,), port)
end

function get_output_type(t::AbstractTensor)
    tf_type = operation_output_type(Port(Tensor(t)))
    if !haskey(type_map, tf_type)
        local dtype
        try
            dtype = get_op(t)["T"]._type
        catch err1
            err1 isa TFException || rethrow(err1)
            try
                dtype = get_op(t)["dtype"]._type
            catch err2
                err2 isa TFException || rethrow(err2)
                return Any
            end
        end
        return proto_type_map[dtype]
    else
        return tf_to_jl_type(tf_type)
    end
end

Base.eltype(::Type{<:AbstractTensor{T}}) where {T} = T

Port(t::Tensor) = Port(t.op.ptr, t.value_index-1)
Port(t::AbstractTensor) = Port(Tensor(t))
Port(op::Operation) = Port(Tensor(op))
Port(port::Port) = port

Tensor(p::Port) = Tensor(Operation(p.node_ptr), p.index+1)

function add_input(desc::NodeDescription, input::Union{Tensor, Operation})
    @tfcall(:TF_AddInput, Cvoid, (Ptr{Cvoid}, Port), desc.ptr, Port(input))
end

function add_input(desc::NodeDescription, inputs::Vector)
    inputs = map(Port, inputs)
    @tfcall(:TF_AddInputList, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Cint), desc.ptr, inputs, length(inputs))
end

function add_control_input(desc::NodeDescription, op::Operation)
    @tfcall(:TF_AddControlInput, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), desc.ptr, op.ptr)
end

add_control_input(desc::NodeDescription, t::Tensor) = add_control_input(desc, t.op)

function setindex!(desc::NodeDescription, tensor::RawTensor, attr_name)
    status = Status()
    @tfcall(:TF_SetAttrTensor, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Ptr{Cvoid}), desc.ptr, attr_name, tensor.ptr, status.ptr)
    check_status(status)
end

function setindex!(desc::NodeDescription, tensors::Vector{RawTensor}, attr_name)
    status = Status()
    @tfcall(:TF_SetAttrTensorList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Ptr{Cvoid}}, Cint, Ptr{Cvoid}), desc.ptr, attr_name, [x.ptr for x in tensors], length(tensors), status.ptr)
    check_status(status)
end

function setindex!(desc::NodeDescription, dtype::DataType, attr_name)
    @tfcall(:TF_SetAttrType, Cvoid, (Ptr{Cvoid}, Cstring, TF_DataType), desc.ptr, attr_name, dtype|>jl_to_df_type)
end

function setindex!(desc::NodeDescription, value::Integer, attr_name)
    value = Int64(value)
    @tfcall(:TF_SetAttrInt, Cvoid, (Ptr{Cvoid}, Cstring, Int64), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::TensorShape, attr_name)
    if value.rank_unknown
        dims = Int[]
    else
        dims = Int[ismissing(dim) ? -1 : dim for dim in value.dims]
    end
    @tfcall(:TF_SetAttrShape, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Int64}, Cint), desc.ptr, attr_name, dims, length(dims))
end

function setindex!(desc::NodeDescription, value::Bool, attr_name)
    @tfcall(:TF_SetAttrBool, Cvoid, (Ptr{Cvoid}, Cstring, Cuchar), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::AbstractFloat, attr_name)
    value = Float32(value)
    @tfcall(:TF_SetAttrFloat, Cvoid, (Ptr{Cvoid}, Cstring, Cfloat), desc.ptr, attr_name, value)
end

function setindex!(desc::NodeDescription, value::AbstractString, attr_name)
    value = String(value)
    @tfcall(:TF_SetAttrString, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Cint), desc.ptr, attr_name, Vector{UInt8}(value), sizeof(value))
end

function setindex!(desc::NodeDescription, value::Vector, attr_name)
    set_attr_list(desc, attr_name, value)
end

function set_attr_list(desc::NodeDescription, attr_name, list::Vector{<:Integer})
    list = Int64[Int64(x) for x in list]
    @tfcall(:TF_SetAttrIntList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Int64}, Cint), desc.ptr, attr_name, list, length(list))
end

function set_attr_list(desc::NodeDescription, attr_name, list::Vector{<:AbstractFloat})
    list = Float32[Float32(x) for x in list]
    @tfcall(:TF_SetAttrFloatList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Float32}, Cint), desc.ptr, attr_name, list, length(list))
end

function set_attr_list(desc::NodeDescription, attr_name, list::Vector{<:DataType})
    list = map(jl_to_df_type, list)
    @tfcall(:TF_SetAttrTypeList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Cint), desc.ptr, attr_name, list, length(list))
end

function set_attr_shape_list(desc::NodeDescription, attr_name, list::Vector)
    dims = Vector{Int64}[]
    for shape in list
        push!(dims, Int64[shape...])
    end
    @tfcall(:TF_SetAttrShapeList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Ptr{Int64}}, Ptr{Cint}, Cint),
        desc.ptr,
        attr_name,
        dims,
        Cint[length(x) for x in dims],
        length(dims))
end

function Base.eltype(t::RawTensor)
    tf_type = @tfcall(:TF_TensorType, TF_DataType, (Ptr{Cvoid},), t.ptr)
    tf_to_jl_type(tf_type)
end

const type_map = Dict(TF_INT8=>Int8, TF_UINT8=>UInt8,
                      TF_INT16=>Int16, TF_UINT16=>UInt16,
                      TF_INT32=>Int32, TF_UINT32=>UInt32,
                      TF_INT64=>Int64, TF_UINT64=>UInt64,
                      TF_FLOAT=>Float32, TF_DOUBLE=>Float64,
                      TF_BOOL=>Bool, TF_STRING=>String,
                      TF_COMPLEX64=>ComplexF32, TF_COMPLEX128=>ComplexF64)
const inv_type_map = Dict(v=>k for (k, v) in type_map)

function tf_to_jl_type(dt::TF_DataType)
    return type_map[dt]
end

function jl_to_df_type(dt)
    return inv_type_map[dt]
end

function Base.convert(::Type{Array}, t::RawTensor)
    dims = ndims(t)
    data = @tfcall(:TF_TensorData, Ptr{Cvoid}, (Ptr{Cvoid},), t.ptr)
    data = convert(Ptr{eltype(t)}, data)
    if eltype(t) == String
        d = size(t)
        out = String[]
        array = unsafe_wrap(Array, convert(Ptr{UInt8}, data), sizeof(t))
        b = IOBuffer(array)
        seekstart(b)
        read(b, sizeof(Int64)*prod(d))  # The offsets
        for i in 1:prod(d)
            len = varint_decode(b)
            raw_data = read(b, len)
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

function Base.convert(::Type{<:Union{Number, String}}, t::RawTensor)
    @assert ndims(t)==0
    return convert(Array, t)[]
end

function get_proto(graph::Graph)
    output = Buffer()
    status = Status()
    @tfcall(:TF_GraphToGraphDef, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), graph.ptr, output.ptr, status.ptr)
    check_status(status)
    convert(Array, output)
end

function get_proto(node::Operation)
    output = Buffer()
    status = Status()
    @tfcall(:TF_OperationToNodeDef, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), node.ptr, output.ptr, status.ptr)
    check_status(status)
    convert(Array, output)
end

get_proto(tensor::AbstractTensor) = get_proto(get_op(tensor))

function get_proto(w::tensorflow.WhileContextDef)
    b = IOBuffer()
    writeproto(b, w)
    take!(b)
end

get_def_type(::Type{Operation}) = tensorflow.NodeDef
get_def_type(::Type{Graph}) = tensorflow.GraphDef

"""
    get_def(n)

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
    if m === nothing
        return (name, 1)
    else
        port = parse(Int, m[2]) + 1
        return (m[1], port)
    end
end

"""
    get_node_by_name(graph::Graph, name::AbstractString)

Returns an operation by searching for its name in the given graph.
"""
@with_def_graph function get_node_by_name(graph::Graph, name::AbstractString)
    name, port = parse_port_name(name)
    node_ptr = @tfcall(:TF_GraphOperationByName, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), graph.ptr, name)
    if node_ptr == C_NULL
        return nothing
    else
        return Operation(node_ptr)
    end
end

"""
    get_tensor_by_name([graph,], name)

Returns the tensor with name `name` (in name:port format) in the given graph.

Throws a `NodeNameNotFound` exception if there is no such tensor.
"""
@with_def_graph function get_tensor_by_name(graph::Graph, full_name)
    name, port = parse_port_name(full_name)
    maybe_node = get_node_by_name(graph, name)
    maybe_node === nothing && throw(NodeNameNotFound(full_name))
    return Tensor(maybe_node, port)
end

# Indexing based name Access
Base.getindex(graph::Graph, name) = get_tensor_by_name(graph, name)
Base.values(graph::Graph) = get_operations(graph)
Base.keys(graph::Graph) = (node_name(op) for op in values(graph))
Base.haskey(graph::Graph, name) = get_node_by_name(graph, name) !== nothing



node_name(::Cvoid) = nothing
node_name(xs::AbstractVector)=node_name.(xs)

"""
    gradients(ys, xs, grad_ys=nothing)

Constructs symbolic partial derivatives of sum of ys w.r.t. x in xs.

ys and xs are each a Tensor or a list of tensors. grad_ys is a list of Tensor, holding the gradients received by the ys. The list must be the same length as ys.

gradients() adds ops to the graph to output the partial derivatives of ys with respect to xs. It returns a list of Tensor of length len(xs) where each tensor is the sum(dy/dx) for y in ys.

`grad_ys` is a tensor or list of tensors which holds the initial gradients for each y in ys.
If `ys` is a single tensor `grad_ys` must be a single tensor; if `ys` is a list of tensors then likewise `grad_ys` must be a list of the same size.
When `grad_ys` is `nothing`, it is effectively defaulted to a tensor of '1's of the shape of y for each y in ys.
`grad_ys` can be partialy specified, with some gradients given and others left to default, py passing their values as  `nothing`.
A user can provide their own initial `grad_ys` to compute the derivatives using a different initial gradient for each y (e.g., if one wanted to weight the gradient differently for each value in each y).

`grad_ys` must be a `Tensor` (or an `Operation`), if a plain julia value to be used, it should be wrapped into a `constant`.

see: [Python Docs](https://www.tensorflow.org/versions/master/api_docs/python/tf/gradients)
"""
gradients(y, x::AbstractArray, dx=nothing) = add_gradients_py(y, x, dx)
gradients(y, x, grad_y=nothing) = gradients(y, [x], grad_y)[1]

function add_gradients_py(y, x::AbstractArray, grad_y=nothing)
    x_names = node_name(x)
    y_names = node_name(y)
    grad_y_names = node_name(grad_y)

    meta_graph = train.export_meta_graph()
    b = IOBuffer()
    writeproto(b, meta_graph)
    graph_proto = take!(b)
    node_protos, grad_names = fetch(@py_proc py_gradients($graph_proto, $x_names, $y_names, $grad_y_names))
    extend_graph(node_protos)
    out = []
    for name in grad_names
        if isa(name, String)
            push!(out, get_tensor_by_name(name))
        elseif isa(name, Cvoid)
            push!(out, nothing)
        else
            push!(out, IndexedSlices(get_tensor_by_name(name[1]), get_tensor_by_name(name[2])+1))
        end
    end
    return out
end

add_gradients_c(y, x, dx=nothing) = add_gradients_c([y], x, dx)
function add_gradients_c(y::AbstractArray, x::AbstractArray, dx=nothing)
    status = Status()
    graph = get_def_graph()
    grads = Array{Port}(length(x))
    if dx isa AbstractArray{Cvoid} || dx isa Cvoid
        c_dx = C_NULL
    else
        c_dx = Port.(Tensor.(dx))
    end
    @tfcall(:TF_AddGradients, Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        graph.ptr, Port.(y), length(y), Port.(x), length(x), c_dx, status.ptr, grads
    )
    check_status(status)
    return Tensor.(grads)
end



function get_num_outputs(op::Operation)
    @tfcall(:TF_OperationNumOutputs, Cint, (Ptr{Cvoid},), op.ptr) |> Int
end

function get_device(op::Operation)
    @tfcall(:TF_OperationDevice, Cstring, (Ptr{Cvoid},), op.ptr) |> String
end

get_device(t::Tensor) = get_device(t.op)

function num_outputs(op::Operation)
    @tfcall(:TF_OperationNumOutputs, Cint, (Ptr{Cvoid},), op.ptr) |> Int
end

get_op(op::Operation) = op
get_op(t::AbstractTensor) = Tensor(t).op

struct IndexedSlices{ValueT, IndexT}
    values::ValueT
    indices::IndexT
end

Base.eltype(::Type{IndexedSlices{ValueT, IndexT}}) where {ValueT, IndexT} = eltype(ValueT)

struct IndexedSlicesValue{ValueT, IndexT}
    values::ValueT
    indices::IndexT
end

mutable struct GraphImportOptions
    input_mapping::Dict{Tuple{String, Int}, Tensor}
    return_output::Vector{Tuple{String, Int}}
    control_dependencies::Vector{Operation}
    prefix::String
    GraphImportOptions() = new(Dict{Tuple{String, Int}, Tensor}(), Vector{Tuple{String, Int}}(), Vector{Operation}(), "")
end

@with_def_graph function import_graph_def(graph::Graph, graph_def::Vector{UInt8}, options=GraphImportOptions())
    version_check(v"1.0.0-rc1")
    options_ptr = @tfcall(:TF_NewImportGraphDefOptions, Ptr{Cvoid}, ())

    ## BEGIN SET OPTIONS
    for ((input_name, input_port), tensor) in options.input_mapping
        @tfcall(:TF_ImportGraphDefOptionsAddInputMapping, Cvoid,
            (Ptr{Cvoid}, Cstring, Cint, Port),
            options_ptr, input_name, input_port-1, Port(tensor))
    end

    for (output_name, output_port) in options.return_output
        @tfcall(:TF_ImportGraphDefOptionsAddReturnOutput, Cvoid,
            (Ptr{Cvoid}, Cstring, Cint),
            options_ptr, output_name, output_port-1)
    end

    for operation in options.control_dependencies
        @tfcall(:TF_ImportGraphDefOptionsAddControlDependency, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}),
            options_ptr, operation.ptr)
    end

    @tfcall(:TF_ImportGraphDefOptionsSetPrefix, Cvoid,
        (Ptr{Cvoid}, Cstring),
        options_ptr, options.prefix)
    ## END SET OPTIONS

    status = Status()
    buffer = Buffer(graph_def)
    output_ports = Vector{Port}(undef, length(options.return_output))

    @tfcall(:TF_GraphImportGraphDefWithReturnOutputs, Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}),
        graph.ptr, buffer.ptr, options_ptr, output_ports, length(output_ports), status.ptr)

    check_status(status)

    @tfcall(:TF_DeleteImportGraphDefOptions, Cvoid, (Ptr{Cvoid},), options_ptr)
    output_tensors = map(Tensor, output_ports)
    output_tensors
end

@with_def_graph function import_graph_def(graph::Graph, graph_def::tensorflow.GraphDef, options=GraphImportOptions())
    b = IOBuffer()
    writeproto(b, graph_def)
    data = take!(b)
    import_graph_def(graph, data, options)
end

struct OperationIterator
    graph::Graph
end

struct OperationIteratorState
    next_op::Union{Operation, Nothing}
    pos::Int
end

function Base.iterate(iter::OperationIterator, state=_next(iter, OperationIteratorState(nothing, 0))[2])
    value, new_state = _next(iter, state)
    state.next_op === nothing ? nothing : (value, new_state)
end

function _next(iter::OperationIterator, state)
    pos = Ref{Csize_t}(state.pos)
    op_ptr = @tfcall(:TF_GraphNextOperation, Ptr{Cvoid},
        (Ptr{Cvoid}, Ref{Csize_t}),
        iter.graph.ptr, pos)
    if op_ptr == C_NULL
        next_op = nothing
    else
        op = Operation()
        op.ptr = op_ptr
        op.graph = iter.graph
        next_op = op
    end
    next_state = OperationIteratorState(next_op, pos[])
    (state.next_op, next_state)
end

#function Base.next(iter::OperationIterator, state)
#    value, new_state = _next(iter, state)
#    (get(value), new_state)
#end

#Base.done(iter::OperationIterator, state) = isnull(state.next_op)

Base.IteratorSize(::Type{OperationIterator}) = Base.SizeUnknown()
Base.eltype(::Type{OperationIterator}) = Operation

@with_def_graph function get_operations(g::Graph)
    OperationIterator(g)
end

get_name(t::Tensor) = "$(get_name(t.op)):$(t.value_index-1)"
get_name(op::Operation) = get_def(op).name
get_name(i::IndexedSlices) = get_name(i.values)

const op_list = Dict{String, tensorflow.OpDef}()

function get_all_op_list()
    !isempty(op_list) && return op_list
    buf_ptr = @tfcall(:TF_GetAllOpList, Ptr{Cvoid}, ())
    buffer = Buffer(buf_ptr)
    data = convert(Array, buffer)
    b = IOBuffer(data)
    ops = tensorflow.OpList()
    readproto(b, ops)
    for op in ops.op
        op_list[op.name] = op
    end
    op_list
end


get_op_def(x::AbstractString) = get_all_op_list()[x]

function Operation(node_def::tensorflow.NodeDef)
    function get_tensor(full_name)
        name, port = parse_port_name(full_name)
        get_tensor_by_name("$name:$(port-1)")
    end

    op = get_all_op_list()[node_def.op]
    desc = NodeDescription(node_def.op, node_def.name)
    inputs = String[]

    # Process control inputs
    for input in node_def.input
        if input[1:1] == "^"
            name = input[2:end]
            control_op = get_node_by_name(name)
            add_control_input(desc, control_op)
        else
            push!(inputs, input)
        end
    end

    # Add regular inputs
    input_idx = 1
    for input in op.input_arg
        number_attr = input.number_attr
        if isempty(number_attr)
            add_input(desc, get_tensor(inputs[input_idx]))
            input_idx += 1
        else
            N = load_proto(node_def.attr[number_attr])
            tensors = []
            for n in 1:N
                push!(tensors, get_tensor(inputs[input_idx]))
                input_idx += 1
            end
            add_input(desc, tensors)
        end
    end

    # Add attributes
    for (key, attr) in node_def.attr
        value = load_proto(attr)
        desc[key] = value
    end

    Operation(desc)
end

function set_tensor_shape(tensor::AbstractTensor, dims)
    output = Port(tensor)
    g = get_graph(tensor)
    status = Status()
    dims = [ismissing(dim) ? -1 : dim for dim in dims]
    @tfcall(:TF_GraphSetTensorShape, Cvoid, (Ptr{Cvoid}, Port, Ptr{Cvoid}, Cint, Ptr{Cvoid}), g.ptr, output, dims, length(dims), status.ptr)
    check_status(status)
end

function get_tensor_num_dims(tensor::AbstractTensor)
    if tensor isa Tensor && tensor.op.op_name == "NoOp"
        # Work  around https://github.com/malmaud/TensorFlow.jl/issues/466
        # In short libtensorflow segfaults when you ask for the shape
        # of a `no_op`. Which includes most control flow statements like `group`
        # which are used e.g. in `minimize`
        return 0
    end
    
    output = Port(tensor)
    g = get_graph(tensor)
    status = Status()
    num_dims = @tfcall(:TF_GraphGetTensorNumDims, Cint, (Ptr{Cvoid}, Port, Ptr{Cvoid}), g.ptr, output, status.ptr)
    check_status(status)
    num_dims = Int(num_dims)
    if num_dims == -1
        return missing
    else
        return num_dims
    end
end

function get_tensor_shape(tensor::AbstractTensor)
    output = Port(tensor)
    g = get_graph(tensor)
    status = Status()
    num_dims = get_tensor_num_dims(tensor)
    if num_dims == 0 || ismissing(num_dims)
        return Int64[]
    end
    dims = Vector{Int64}(undef, num_dims)
    @tfcall(:TF_GraphGetTensorShape, Cvoid, (Ptr{Cvoid}, Port, Ptr{Int64}, Cint, Ptr{Cvoid}), g.ptr, output, dims, length(dims), status.ptr)
    check_status(status)
    missing_dims = [dim==-1 ? missing : dim for dim in dims]
    return missing_dims
end

function get_shape(tensor::AbstractTensor)
    # TODO understand/fix why gradient tensors have 'nothing' for their graph
    if get_graph(tensor) === nothing || ismissing(get_tensor_num_dims(tensor))
        return TensorShape([], true)
    else
        return TensorShape(get_tensor_shape(tensor), false)
    end
end

function get_shape(tensor::AbstractTensor, dim)
    shape = get_shape(tensor)
    if shape.rank_unknown
        error("Rank is unknown")
    end
    if length(shape.dims) < dim
        error("$dim is more than shape's rank of $(length(shape.dims))")
    end
    return shape.dims[dim]
end
