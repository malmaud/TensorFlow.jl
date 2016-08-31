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

export NodeDef_AttrEntry, NodeDef
# mapentries: Pair{AbstractString,Tuple{AbstractString,AbstractString}}("NodeDef_AttrEntry",("AbstractString","AttrValue"))
