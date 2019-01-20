# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct _Any <: ProtoType
    type_url::AbstractString
    value::Array{UInt8,1}
    _Any(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct _Any

export _Any
