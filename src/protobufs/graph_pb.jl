# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct GraphDef
    node::Array{NodeDef,1}
    versions::VersionDef
    version::Int32
    library::FunctionDefLibrary
    GraphDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type GraphDef
const __fnum_GraphDef = Int[1,4,3,2]
meta(t::Type{GraphDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GraphDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::GraphDef) = ProtoBuf.protohash(v)
isequal(v1::GraphDef, v2::GraphDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::GraphDef, v2::GraphDef) = ProtoBuf.protoeq(v1, v2)

export GraphDef
