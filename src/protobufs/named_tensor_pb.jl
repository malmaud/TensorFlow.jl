# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct NamedTensorProto <: ProtoType
    name::AbstractString
    tensor::TensorProto
    NamedTensorProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NamedTensorProto

export NamedTensorProto
