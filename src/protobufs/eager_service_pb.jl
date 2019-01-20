# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow

mutable struct RemoteTensorHandle <: ProtoType
    op_id::Int64
    output_num::Int32
    RemoteTensorHandle(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RemoteTensorHandle

mutable struct Operation_AttrsEntry <: ProtoType
    key::AbstractString
    value::tensorflow.AttrValue
    Operation_AttrsEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Operation_AttrsEntry (mapentry)

mutable struct Operation <: ProtoType
    id::Int64
    name::AbstractString
    inputs::Base.Vector{RemoteTensorHandle}
    control_op_ids::Base.Vector{Int64}
    attrs::Base.Dict{AbstractString,tensorflow.AttrValue} # map entry
    device::AbstractString
    Operation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Operation
const __pack_Operation = Symbol[:control_op_ids]
meta(t::Type{Operation}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_Operation, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct QueueItem <: ProtoType
    handle_to_decref::RemoteTensorHandle
    operation::Operation
    QueueItem(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct QueueItem
const __oneofs_QueueItem = Int[1,1]
const __oneof_names_QueueItem = [Symbol("item")]
meta(t::Type{QueueItem}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_QueueItem, __oneof_names_QueueItem, ProtoBuf.DEF_FIELD_TYPES)

mutable struct QueueResponse <: ProtoType
    shape::Base.Vector{tensorflow.TensorShapeProto}
    QueueResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct QueueResponse

mutable struct CreateContextRequest <: ProtoType
    server_def::tensorflow.ServerDef
    async::Bool
    keep_alive_secs::Int64
    version_def::tensorflow.VersionDef
    rendezvous_id::Int64
    CreateContextRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateContextRequest

mutable struct CreateContextResponse <: ProtoType
    context_id::UInt64
    device_attributes::Base.Vector{tensorflow.DeviceAttributes}
    CreateContextResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateContextResponse
const __wtype_CreateContextResponse = Dict(:context_id => :fixed64)
meta(t::Type{CreateContextResponse}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_CreateContextResponse, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct EnqueueRequest <: ProtoType
    context_id::UInt64
    queue::Base.Vector{QueueItem}
    EnqueueRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct EnqueueRequest
const __fnum_EnqueueRequest = Int[1,3]
const __wtype_EnqueueRequest = Dict(:context_id => :fixed64)
meta(t::Type{EnqueueRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_EnqueueRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_EnqueueRequest, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct EnqueueResponse <: ProtoType
    queue_response::Base.Vector{QueueResponse}
    EnqueueResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct EnqueueResponse

mutable struct WaitQueueDoneRequest <: ProtoType
    context_id::UInt64
    op_id::Base.Vector{Int64}
    WaitQueueDoneRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct WaitQueueDoneRequest
const __pack_WaitQueueDoneRequest = Symbol[:op_id]
const __wtype_WaitQueueDoneRequest = Dict(:context_id => :fixed64)
meta(t::Type{WaitQueueDoneRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_WaitQueueDoneRequest, __wtype_WaitQueueDoneRequest, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct WaitQueueDoneResponse <: ProtoType
    WaitQueueDoneResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct WaitQueueDoneResponse

mutable struct KeepAliveRequest <: ProtoType
    context_id::UInt64
    KeepAliveRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct KeepAliveRequest
const __wtype_KeepAliveRequest = Dict(:context_id => :fixed64)
meta(t::Type{KeepAliveRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_KeepAliveRequest, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct KeepAliveResponse <: ProtoType
    KeepAliveResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct KeepAliveResponse

mutable struct CloseContextRequest <: ProtoType
    context_id::UInt64
    CloseContextRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CloseContextRequest
const __wtype_CloseContextRequest = Dict(:context_id => :fixed64)
meta(t::Type{CloseContextRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_CloseContextRequest, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct CloseContextResponse <: ProtoType
    CloseContextResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CloseContextResponse

mutable struct RegisterFunctionRequest <: ProtoType
    context_id::UInt64
    function_def::tensorflow.FunctionDef
    RegisterFunctionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RegisterFunctionRequest
const __wtype_RegisterFunctionRequest = Dict(:context_id => :fixed64)
meta(t::Type{RegisterFunctionRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_RegisterFunctionRequest, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct RegisterFunctionResponse <: ProtoType
    RegisterFunctionResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RegisterFunctionResponse

mutable struct SendTensorRequest <: ProtoType
    context_id::UInt64
    op_id::Int64
    tensors::Base.Vector{tensorflow.TensorProto}
    device_name::AbstractString
    SendTensorRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SendTensorRequest
const __wtype_SendTensorRequest = Dict(:context_id => :fixed64)
meta(t::Type{SendTensorRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_SendTensorRequest, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct SendTensorResponse <: ProtoType
    SendTensorResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SendTensorResponse

export RemoteTensorHandle, Operation_AttrsEntry, Operation, QueueItem, QueueResponse, CreateContextRequest, CreateContextResponse, EnqueueRequest, EnqueueResponse, WaitQueueDoneRequest, WaitQueueDoneResponse, KeepAliveRequest, KeepAliveResponse, CloseContextRequest, CloseContextResponse, RegisterFunctionRequest, RegisterFunctionResponse, SendTensorRequest, SendTensorResponse
# mapentries: "Operation_AttrsEntry" => ("AbstractString", "tensorflow.AttrValue")
