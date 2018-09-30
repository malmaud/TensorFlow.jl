# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct IteratorStateMetadata <: ProtoType
    version::AbstractString
    keys::Base.Vector{AbstractString}
    IteratorStateMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct IteratorStateMetadata

export IteratorStateMetadata
