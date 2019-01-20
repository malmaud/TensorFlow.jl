# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct GraphDef <: ProtoType
    node::Base.Vector{NodeDef}
    versions::VersionDef
    version::Int32
    library::FunctionDefLibrary
    GraphDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphDef
const __fnum_GraphDef = Int[1,4,3,2]
meta(t::Type{GraphDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GraphDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export GraphDef
