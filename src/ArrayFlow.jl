module ArrayFlow

import Base: +, -, *, /, setindex!, run

include("constants.jl")
include("tensorflow.jl")

const LIB = joinpath(dirname(@__FILE__), "..", "deps", "libc_api.so")
const LIB_BASE = "/Users/malmaud/tensorflow"

Libdl.dlopen(joinpath(LIB_BASE, "bazel-bin", "tensorflow", "libtensorflow.so"), Libdl.RTLD_GLOBAL)
Libdl.dlopen(joinpath(LIB_BASE, "bazel-bin", "tensorflow", "c", "libc_api.so"), Libdl.RTLD_GLOBAL)

type Status
    ptr::Ptr{Void}
    function Status()
        ptr = ccall((:TF_NewStatus), Ptr{Void}, ())
        this = new(ptr)
        finalizer(this, this->begin
            ccall((:TF_DeleteStatus), Void, (Ptr{Void},), this.ptr)
        end)
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

type Session
    ptr::Ptr{Void}
    graph::Graph

    function Session(graph=Graph(), options=SessionOptions())
        status = Status()
        ptr = ccall((:TF_NewSessionWithGraph), Ptr{Void}, (Ptr{Void}, Ptr{Void}, Ptr{Void}), graph.ptr, options.ptr, status.ptr)
        this = new(ptr, graph)
        check_status(status)
        finalizer(this, self->begin
            status = Status()
            ccall((:TF_DeleteSessionWithGraph), Void, (Ptr{Void}, Ptr{Void}), self.ptr, status)
        end)
        return this
    end
end

function extend_graph(sess::Session, proto::Vector{UInt8})
    status = Status()
    ccall((:TF_ExtendGraph), Void, (Ptr{Void}, Ptr{Void}, Csize_t, Ptr{Void}), sess.ptr, pointer(proto), sizeof(proto), status.ptr)
    return status
end

extend_graph(sess::Session, proto::String) = extend_graph(sess, proto.data)


type Buffer
    ptr::Ptr{Void}
end

function Buffer(s::Vector{UInt8})
    ptr = ccall((:TF_NewBufferFromString), Ptr{Void}, (Ptr{Void}, Csize_t), pointer(s), sizeof(s))
    return Buffer(ptr)
end

function deallocator(data, len, arg)

end

c_deallocator = cfunction(deallocator, Void, (Ptr{Void}, Csize_t, Ptr{Void}))

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
end

node_name(node::Node) = ccall((:TF_NodeName), Cstring, (Ptr{Void},), node.ptr)

immutable Port
    node_ptr::Ptr{Void}
    index::Int
end

Port(node::Node, index) = Port(node.ptr, index)

function add_input(desc::NodeDescription, input::Port)
    ccall((:TF_AddInput), Void, (Ptr{Void}, Port), desc.ptr, input)
end

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

function Base.run(sess::Session, inputs, input_values, outputs, targets)
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
#
# Base.run(sess::Session, output_name, feed_dict::Associative) = run(sess, [output_name], feed_dict)[1]
# Base.run(sess::Session, output_name) = run(sess, [output_name])[1]
#
# function Base.run(sess::Session, output_names::Vector, feed_dict::Associative)
#     input_names = map(get_name, collect(keys(feed_dict)))
#     inputs = collect(values(feed_dict))
#     run(sess, input_names, inputs, map(get_name, output_names, []))
# end
#
# Base.run(sess::Session, output_names::Vector) = run(sess, output_names, Dict())

function Base.run(sess::Session)
end

function Base.eltype(t::Tensor)
    tf_type = ccall((:TF_TensorType), TF_DataType, (Ptr{Void},), t.ptr)
    tf_to_jl_type(tf_type)
end

const type_map = Dict(TF_FLOAT=>Float32, TF_INT32=>Int32)
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

end
