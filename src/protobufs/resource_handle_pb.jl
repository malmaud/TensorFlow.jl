# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct ResourceHandleProto <: ProtoType
    device::AbstractString
    container::AbstractString
    name::AbstractString
    hash_code::UInt64
    maybe_type_name::AbstractString
    ResourceHandleProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ResourceHandleProto

export ResourceHandleProto
