# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct TensorShapeProto_Dim <: ProtoType
    size::Int64
    name::AbstractString
    TensorShapeProto_Dim(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorShapeProto_Dim

mutable struct TensorShapeProto <: ProtoType
    dim::Base.Vector{TensorShapeProto_Dim}
    unknown_rank::Bool
    TensorShapeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorShapeProto
const __fnum_TensorShapeProto = Int[2,3]
meta(t::Type{TensorShapeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorShapeProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export TensorShapeProto_Dim, TensorShapeProto
