# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct TensorSliceProto_Extent <: ProtoType
    start::Int64
    length::Int64
    TensorSliceProto_Extent(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorSliceProto_Extent
const __oneofs_TensorSliceProto_Extent = Int[0,1]
const __oneof_names_TensorSliceProto_Extent = [Symbol("has_length")]
meta(t::Type{TensorSliceProto_Extent}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_TensorSliceProto_Extent, __oneof_names_TensorSliceProto_Extent, ProtoBuf.DEF_FIELD_TYPES)

mutable struct TensorSliceProto <: ProtoType
    extent::Base.Vector{TensorSliceProto_Extent}
    TensorSliceProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorSliceProto

export TensorSliceProto_Extent, TensorSliceProto
