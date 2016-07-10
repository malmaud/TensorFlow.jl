module ArrayFlow

import Base: +, -, *, /
using PyCall
@pyimport tensorflow as tf

include("constants.jl")

const LIB = joinpath(dirname(@__FILE__), "..", "deps", "libtensorflow.so")

type Status
    ptr::Ptr{Void}
    function Status()
        ptr = ccall((:TF_NewStatus, LIB), Ptr{Void}, ())
        this = new(ptr)
        finalizer(this, this->begin
            ccall((:TF_DeleteStatus, LIB), Void, (Ptr{Void},), this.ptr)
        end)
        this
    end
end

function Base.show(io::IO, s::Status)
    msg = ccall((:TF_Message, LIB), Cstring, (Ptr{Void},), s.ptr) |> unsafe_string
    print(io, @sprintf("Status: %s", msg))
end

function get_code(s::Status)
    code = ccall((:TF_GetCode, LIB), Cint, (Ptr{Void},), s.ptr)
    return TF_Code(code)
end

type SessionOptions
    ptr::Ptr{Void}

    function SessionOptions()
        ptr = ccall((:TF_NewSessionOptions, LIB), Ptr{Void}, ())
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
    function Session(options=SessionOptions())
        status = Status()
        ptr = ccall((:TF_NewSession, LIB), Ptr{Void}, (Ptr{Void}, Ptr{Void}), options.ptr, status.ptr)
        this = new(ptr)
        check_status(status)
        return this
    end
end

function extend_graph(sess::Session, proto::Vector{UInt8})
    status = Status()
    ccall((:TF_ExtendGraph, LIB), Void, (Ptr{Void}, Ptr{Void}, Csize_t, Ptr{Void}), sess.ptr, pointer(proto), sizeof(proto), status.ptr)
    return status
end


type Buffer
    ptr::Ptr{Void}
end

function Buffer(s::Vector{UInt8})
    ptr = ccall((:TF_NewBufferFromString, LIB), Ptr{Void}, (Ptr{Void}, Csize_t), pointer(s), sizeof(s))
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

        ptr = ccall((:TF_NewTensor, LIB), Ptr{Void}, (Cint, Ptr{Cint}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
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
        ptr = ccall((:TF_NewTensor, LIB), Ptr{Void}, (Cint, Ptr{Void}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}),
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
            ccall((:TF_DeleteTensor, LIB), Void, (Ptr{Void},), this.ptr)
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
    ccall((:TF_NumDims, LIB), Cint, (Ptr{Void},), t.ptr) |> Int
end

function Base.size(t::Tensor, dim::Integer)
    n = ndims(t)
    dim -= 1
    @assert dim < n
    ccall((:TF_Dim, LIB), Clonglong, (Ptr{Void}, Cint), t.ptr, dim)
end

function Base.size(t::Tensor)
    d = (size(t,_) for _ in 1:ndims(t))
    (d...)
end

function Base.sizeof(t::Tensor)
    ccall((:TF_TensorByteSize, LIB), Csize_t, (Ptr{Void},), t.ptr) |> Int
end

function Base.run(sess::Session, input_names, inputs, output_names)
    status = Status()
    output_ptr = fill(C_NULL, length(output_names))
    input_tensors = [Tensor(_) for _ in inputs]
    ccall((:TF_Run, LIB), Void,
    (Ptr{Void}, Ptr{Void}, Ptr{Cstring}, Ptr{Void}, Cint, Ptr{Cstring}, Ptr{Ptr{Void}}, Cint, Ptr{Void}, Cint, Ptr{Void}, Ptr{Void}),
        sess.ptr,
        C_NULL,
        input_names,
        [_.ptr for _ in input_tensors],
        length(input_tensors),
        output_names,
        output_ptr,
        length(output_ptr),
        C_NULL,
        0,
        C_NULL,
        status.ptr)
    check_status(status)
    return [Tensor(_) for _ in output_ptr]
end

Base.run(sess::Session, output_name, feed_dict::Associative) = run(sess, [output_name], feed_dict)[1]

function Base.run(sess::Session, output_names::Vector, feed_dict::Associative)
    input_names = map(get_name, collect(keys(feed_dict)))
    inputs = collect(values(feed_dict))
    run(sess, input_names, inputs, map(get_name, output_names))
end

Base.run(sess::Session, output_names::Vector) = run(sess, output_names, Dict())

function Base.eltype(t::Tensor)
    tf_type = ccall((:TF_TensorType, LIB), TF_DataType, (Ptr{Void},), t.ptr)
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
    data = ccall((:TF_TensorData, LIB), Ptr{eltype(t)}, (Ptr{Void},), t.ptr)
    unsafe_wrap(Array, data, size(t))
end

function Base.convert(::Type{Number}, t::Tensor)
    @assert ndims(t)==0
    return convert(Array, t)[]
end


function extend_graph(sess::Session)
    proto = tf.get_default_graph()[:as_graph_def]()[:SerializeToString]()
    extend_graph(sess, proto.data)
end

immutable Node
    o::PyObject

    function Node(o::PyObject)
        this = new(o)
    end
end

function placeholder(dt::TF_DataType)
    Node(tf.placeholder(Int(dt)))
end

function placeholder(dt::Type)
    placeholder(jl_to_df_type(dt))
end

for (jl_op, tf_op) in [(:+, :add), (:-, :sub), (:*, :mul), (:/, :div)]
    @eval function $jl_op(n1::Node, n2::Node)
        Node(tf.$tf_op(n1.o, n2.o))
    end
end

get_name(s::String) = s
get_name(n::Node) = n.o[:name]

end
