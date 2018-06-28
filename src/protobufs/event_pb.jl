# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct __enum_LogMessage_Level <: ProtoEnum
    UNKNOWN::Int32
    DEBUG::Int32
    INFO::Int32
    WARN::Int32
    ERROR::Int32
    FATAL::Int32
    __enum_LogMessage_Level() = new(0,10,20,30,40,50)
end #type __enum_LogMessage_Level
const LogMessage_Level = __enum_LogMessage_Level()

mutable struct LogMessage
    level::Int32
    message::AbstractString
    LogMessage(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type LogMessage
hash(v::LogMessage) = ProtoBuf.protohash(v)
isequal(v1::LogMessage, v2::LogMessage) = ProtoBuf.protoisequal(v1, v2)
==(v1::LogMessage, v2::LogMessage) = ProtoBuf.protoeq(v1, v2)

mutable struct __enum_SessionLog_SessionStatus <: ProtoEnum
    STATUS_UNSPECIFIED::Int32
    START::Int32
    STOP::Int32
    CHECKPOINT::Int32
    __enum_SessionLog_SessionStatus() = new(0,1,2,3)
end #type __enum_SessionLog_SessionStatus
const SessionLog_SessionStatus = __enum_SessionLog_SessionStatus()

mutable struct SessionLog
    status::Int32
    checkpoint_path::AbstractString
    msg::AbstractString
    SessionLog(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SessionLog
hash(v::SessionLog) = ProtoBuf.protohash(v)
isequal(v1::SessionLog, v2::SessionLog) = ProtoBuf.protoisequal(v1, v2)
==(v1::SessionLog, v2::SessionLog) = ProtoBuf.protoeq(v1, v2)

mutable struct TaggedRunMetadata
    tag::AbstractString
    run_metadata::Array{UInt8,1}
    TaggedRunMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TaggedRunMetadata
hash(v::TaggedRunMetadata) = ProtoBuf.protohash(v)
isequal(v1::TaggedRunMetadata, v2::TaggedRunMetadata) = ProtoBuf.protoisequal(v1, v2)
==(v1::TaggedRunMetadata, v2::TaggedRunMetadata) = ProtoBuf.protoeq(v1, v2)

mutable struct Event
    wall_time::Float64
    step::Int64
    file_version::AbstractString
    graph_def::Array{UInt8,1}
    summary::Summary
    log_message::LogMessage
    session_log::SessionLog
    tagged_run_metadata::TaggedRunMetadata
    Event(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type Event
const __oneofs_Event = Int[0,0,1,1,1,1,1,1]
const __oneof_names_Event = [Symbol("what")]
meta(t::Type{Event}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_Event, __oneof_names_Event)
hash(v::Event) = ProtoBuf.protohash(v)
isequal(v1::Event, v2::Event) = ProtoBuf.protoisequal(v1, v2)
==(v1::Event, v2::Event) = ProtoBuf.protoeq(v1, v2)

export Event, LogMessage_Level, LogMessage, SessionLog_SessionStatus, SessionLog, TaggedRunMetadata
