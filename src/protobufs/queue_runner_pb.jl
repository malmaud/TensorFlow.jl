# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow

mutable struct QueueRunnerDef <: ProtoType
    queue_name::AbstractString
    enqueue_op_name::Base.Vector{AbstractString}
    close_op_name::AbstractString
    cancel_op_name::AbstractString
    queue_closed_exception_types::Base.Vector{Int32}
    QueueRunnerDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct QueueRunnerDef
const __pack_QueueRunnerDef = Symbol[:queue_closed_exception_types]
meta(t::Type{QueueRunnerDef}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_QueueRunnerDef, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export QueueRunnerDef
