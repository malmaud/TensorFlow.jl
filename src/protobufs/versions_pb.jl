# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct VersionDef
    producer::Int32
    min_consumer::Int32
    bad_consumers::Array{Int32,1}
    VersionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type VersionDef
const __pack_VersionDef = Symbol[:bad_consumers]
meta(t::Type{VersionDef}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_VersionDef, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::VersionDef) = ProtoBuf.protohash(v)
isequal(v1::VersionDef, v2::VersionDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::VersionDef, v2::VersionDef) = ProtoBuf.protoeq(v1, v2)

export VersionDef
