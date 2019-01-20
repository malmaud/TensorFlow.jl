# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct InterconnectLink <: ProtoType
    device_id::Int32
    _type::AbstractString
    strength::Int32
    InterconnectLink(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct InterconnectLink

mutable struct LocalLinks <: ProtoType
    link::Base.Vector{InterconnectLink}
    LocalLinks(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LocalLinks

mutable struct DeviceLocality <: ProtoType
    bus_id::Int32
    numa_node::Int32
    links::LocalLinks
    DeviceLocality(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceLocality

mutable struct DeviceAttributes <: ProtoType
    name::AbstractString
    device_type::AbstractString
    memory_limit::Int64
    locality::DeviceLocality
    incarnation::UInt64
    physical_device_desc::AbstractString
    DeviceAttributes(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceAttributes
const __fnum_DeviceAttributes = Int[1,2,4,5,6,7]
const __wtype_DeviceAttributes = Dict(:incarnation => :fixed64)
meta(t::Type{DeviceAttributes}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DeviceAttributes, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_DeviceAttributes, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export InterconnectLink, LocalLinks, DeviceLocality, DeviceAttributes
