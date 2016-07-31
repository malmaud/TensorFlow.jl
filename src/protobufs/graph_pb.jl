# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type NodeDef_AttrEntry
    key::AbstractString
    value::AttrValue
    NodeDef_AttrEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NodeDef_AttrEntry (mapentry)
hash(v::NodeDef_AttrEntry) = ProtoBuf.protohash(v)
isequal(v1::NodeDef_AttrEntry, v2::NodeDef_AttrEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::NodeDef_AttrEntry, v2::NodeDef_AttrEntry) = ProtoBuf.protoeq(v1, v2)

type NodeDef
    name::AbstractString
    op::AbstractString
    input::Array{AbstractString,1}
    device::AbstractString
    attr::Dict{AbstractString,AttrValue} # map entry
    NodeDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NodeDef
hash(v::NodeDef) = ProtoBuf.protohash(v)
isequal(v1::NodeDef, v2::NodeDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::NodeDef, v2::NodeDef) = ProtoBuf.protoeq(v1, v2)

type GraphDef
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

export GraphDef, NodeDef_AttrEntry, NodeDef
# mapentries: Pair{AbstractString,Tuple{AbstractString,AbstractString}}("NodeDef_AttrEntry",("AbstractString","AttrValue"))
