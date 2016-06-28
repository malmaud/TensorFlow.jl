module ArrayFlow

include("constants.jl")



const LIB = joinpath(homedir(), "Dropbox/code/arrayflow/deps", "libtensorflow.so")

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

function Base.string(s::Status)
    ccall((:TF_Message, LIB), Cstring, (Ptr{Void},), s.ptr) |> unsafe_string
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

function Base.show(io::IO, s::Status)
    println(io, @sprintf("Tensorflow error: %s", string(s)))
end

type Session
    ptr::Ptr{Void}
    function Session(options=SessionOptions())
        status = Status()
        ptr = ccall((:TF_NewSession, LIB), Ptr{Void}, (Ptr{Void}, Ptr{Void}), options.ptr, status.ptr)
        this = new(ptr)
        if get_code(status) ≠ TF_OK
            throw(TFException(status))
        end
        return this
    end
end

function extend_graph(sess::Session, proto::Vector{UInt8})
    status = Status()
    ccall((:TF_ExtendGraph, LIB), Void, (Ptr{Void}, Ptr{Void}, Csize_t, Ptr{Void}), sesss.ptr, pointer(proto), sizeof(proto), status.ptr)
    return status
end

type Buffer
    ptr::Ptr{Void}
end

function Buffer(s::Vector{UInt8})
    ptr = ccall((:TF_NewBufferFromString, LIB), Ptr{Void}, (Ptr{Void}, Csize_t), pointer(s), sizeof(s))
    return Buffer(ptr)
end

type Tensor
    ptr::Ptr{Void}
    function Tensor(dt::TF_DataType, dims::Vector{Int}, data::Vector)
        ptr = ccall((:TF_NewTensor, LIB), Ptr{Void}, (Cint, Ptr{Void}, Cint, Ptr{Void}, Csize_t, Ptr{Void}, Ptr{Void}), Int(dt), pointer(dims), length(dims), pointer(data), sizeof(data), C_NULL, C_NULL)
        return new(ptr)
    end
end




end
