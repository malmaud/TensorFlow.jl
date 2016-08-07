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

    function Graph()
        ptr = ccall((:TF_NewGraph), Ptr{Void}, ())
        self = new(ptr)
        finalizer(self, self->begin
            ccall((:TF_DeleteGraph), Void, (Ptr{Void},), self.ptr)
        end)
        self
    end
end

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

type Session
    ptr::Ptr{Void}
    graph::Graph

    function Session(graph=Graph())
        global def_graph = graph
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

type Tensor
    ptr::Ptr{Void}
    data::Array  # To avoid underlying data being GCed
    function Tensor(data::Array)
        dims = [size(data)...]
        dt = jl_to_df_type(eltype(data))

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

function Base.show(io::IO, t::Tensor)
    print(io, "Tensor: ")
    if ndims(t) == 0
        show(io, Number(t))
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

    function NodeDescription(graph, op_type, node_name)
        desc = ccall((:TF_NewNode), Ptr{Void}, (Ptr{Void}, Cstring, Cstring), graph.ptr, op_type, node_name)
        new(desc)
    end

end

type Node
    ptr::Ptr{Void}

    function Node(desc::NodeDescription)
        status = Status()
        ptr = ccall((:TF_FinishNode), Ptr{Void}, (Ptr{Void}, Ptr{Void}), desc.ptr, status.ptr)
        check_status(status)
        new(ptr)
    end

    Node(ptr::Ptr) = new(ptr)
end

function Node(node_def::tensorflow.NodeDef)
    graph = get_def_graph()
    desc = NodeDescription(graph, node_def.op, node_def.name)
    for input in node_def.input
        input, port = parse_port_name(input)
        input_node = get_node_by_name(graph, input)
        if isnull(input_node)
            warn("Could not find name $input")
        end
        add_input(desc, Port(input_node |> get, port))
    end
    if isdefined(node_def, :attr)  # TODO: complete this
        for (attr_name, attr) in node_def.attr
            if attr_name == "dtype"
                ccall(:TF_SetAttrType, Void, (Ptr{Void}, Cstring, Cint), desc.ptr, attr_name, attr._type)
            elseif attr_name == "value"
                info("Got value ", attr.tensor)
                desc["value"] = Tensor(attr.tensor.float_val[1])
            elseif attr_name == "keep_dims"
                desc["keep_dims"] = attr.b
            else
            end
        end
    end
    Node(desc)
end

node_name(node::Node) = ccall((:TF_NodeName), Cstring, (Ptr{Void},), node.ptr) |> String

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
    nothing
end

function setindex!(desc::NodeDescription, dtype::DataType, attr_name)
    ccall((:TF_SetAttrType), Void, (Ptr{Void}, Cstring, TF_DataType), desc.ptr, attr_name, dtype|>jl_to_df_type)
    nothing
end

function setindex!(desc::NodeDescription, value::Int, attr_name)
    ccall((:TF_SetAttrType), Void, (Ptr{Void}, Cstring, Int64), desc.ptr, attr_name, value)
    nothing
end

function setindex!(desc::NodeDescription, value::Tuple, attr_name)
    dims = Int[value...]
    ccall(:TF_SetAttrShape, Void, (Ptr{Void}, Cstring, Ptr{Int64}, Cint), desc.ptr, attr_name, dims, length(dims))
    nothing
end

function setindex!(desc::NodeDescription, value::Bool, attr_name)
    ccall(:TF_SetAttrBool, Void, (Ptr{Void}, Cstring, Cuchar), desc.ptr, attr_name, value)
    nothing
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
    return [Tensor(_) for _ in output_values]
end

function run(sess::Session, outputs::AbstractVector{Node}, input_dict)
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

const type_map = Dict(TF_FLOAT=>Float32, TF_INT32=>Int32, TF_INT64=>Int64, TF_DOUBLE=>Float64)
const inv_type_map = Dict(v=>k for (k, v) in type_map)

function tf_to_jl_type(dt::TF_DataType)
    return type_map[dt]
end

function jl_to_df_type(dt)
    return inv_type_map[dt]
end

function Base.convert(::Type{Array}, t::Tensor)
    dims = ndims(t)
    data = ccall((:TF_TensorData), Ptr{eltype(t)}, (Ptr{Void},), t.ptr)
    unsafe_wrap(Array, data, size(t))
end

function Base.convert(::Type{Number}, t::Tensor)
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
