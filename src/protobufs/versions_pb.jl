# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct VersionDef <: ProtoType
    producer::Int32
    min_consumer::Int32
    bad_consumers::Base.Vector{Int32}
    VersionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct VersionDef
const __pack_VersionDef = Symbol[:bad_consumers]
meta(t::Type{VersionDef}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_VersionDef, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export VersionDef
