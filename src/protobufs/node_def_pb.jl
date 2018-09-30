# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct NodeDef_AttrEntry <: ProtoType
    key::AbstractString
    value::AttrValue
    NodeDef_AttrEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NodeDef_AttrEntry (mapentry)

mutable struct NodeDef <: ProtoType
    name::AbstractString
    op::AbstractString
    input::Base.Vector{AbstractString}
    device::AbstractString
    attr::Base.Dict{AbstractString,AttrValue} # map entry
    NodeDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NodeDef

export NodeDef_AttrEntry, NodeDef
# mapentries: "NodeDef_AttrEntry" => ("AbstractString", "AttrValue")
